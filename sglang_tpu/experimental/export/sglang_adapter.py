"""Exposes an SGLang model running on TPU through the exporter's model protocol.

The exporter (`exporter.export`) lowers `forward`, `compute_logits` and the samplers
of an `api.ExportableTextOnlyLLM`. This module provides one for a model that SGLang
loaded on TPU (`load_model_runner`): the forward is SGLang's own model forward over
its Pallas ragged paged attention (RPA v3) backend, with the artifact's control
inputs turned into the backend's `PallasForwardMetadata` and its KV caches swapped
into the token-to-KV pool for the call.

The artifact's inputs map onto the RPA metadata one to one:

| artifact input            | `PallasForwardMetadata` field            |
|---------------------------|------------------------------------------|
| `kv_lens[max_num_seqs]`   | `kv_lens` (KV length after this step)    |
| `block_tables[max_pages]` | `page_indices` (`pages_per_seq` per row) |
| `query_start_loc[S + 1]`  | `cu_q_lens`                              |
| `req_dist[3]`             | `distribution` (decode, prefill, mixed)  |

`cu_kv_lens` is the rectangular page-table stride, a constant of the graph, and
`active_num_tokens` is `query_start_loc[req_dist[2]]`. Each layer's KV cache is
the fused `[num_blocks, page_size, 2 * kv_heads / packing, packing, head_dim]`
buffer of `TpuFusedMHATokenToKVPool`, which the RPA kernel writes in place.

Supported: dense text decoders on the RPA v3 backend at tensor parallel size 1.
"""

from __future__ import annotations

import contextlib
import dataclasses
import logging
import math
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn

from sglang_tpu.experimental.export import api, rope_patch, sampling, tpu_lowering

logger = logging.getLogger(__name__)

# The exporter and the checkpoint writer look an MTP draft up under this
# attribute; SGLang models are exported without one.
MTP_DRAFT_ATTR = "_export_mtp_draft"

# Buffers become graph constants; anything larger belongs in the checkpoint.
_MAX_CONSTANT_ELEMENTS = 1024
_MIN_REQ_BUCKET = 8


def mtp_draft_tensors(model: nn.Module) -> dict[str, Any]:
    """The MTP draft's own tensors by export name; SGLang exports carry none."""
    del model
    return {}


def default_req_buckets(max_num_seqs: int) -> tuple[int, ...]:
    """Powers of two from 8 up to, and always ending with, `max_num_seqs`."""
    buckets = []
    size = _MIN_REQ_BUCKET
    while size < max_num_seqs:
        buckets.append(size)
        size *= 2
    buckets.append(max_num_seqs)
    return tuple(buckets)


def load_model_runner(server_args: Any) -> Any:
    """Loads the model of `server_args` on this TPU without a server or scheduler.

    The single-rank steps of `sglang.benchmark.one_batch.load_model`: resolve the
    server arguments, publish this process as rank 0, initialize the TPU worker
    and the parallel runtime, then build the `ModelRunner` with its KV pool and
    attention backends. No graph runner is built: the exporter traces the
    forward itself.
    """
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.distributed import bootstrap
    from sglang.srt.entrypoints.engine import _set_envs_and_config
    from sglang.srt.layers.moe import initialize_moe_config
    from sglang.srt.layers.quantization.fp4_utils import initialize_fp4_gemm_config
    from sglang.srt.layers.quantization.fp8_utils import initialize_fp8_gemm_config
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.arg_groups.overrides import resolving_view
    from sglang.srt.runtime_context import (
        SpawnRanks,
        get_device,
        publish,
        spawn_world_rank,
    )
    from sglang.srt.server_args import PortArgs
    from sglang.srt.utils.patch_torch import initialize_torch_tpu_worker

    server_args.resolve_once()
    _set_envs_and_config(server_args)
    port_args = PortArgs.init_new(server_args)
    publish(
        server_args,
        role="scheduler",
        ranks=SpawnRanks(
            world_rank=spawn_world_rank(server_args, tp_rank=0, pp_rank=0),
            gpu_id=0,
        ),
    )
    initialize_moe_config()
    initialize_fp8_gemm_config()
    initialize_fp4_gemm_config()

    initialize_torch_tpu_worker(server_args, 0)
    cfg = resolving_view(server_args)
    model_config = ModelConfig.from_server_args(server_args)
    bootstrap.init_parallel_runtime(
        server_args=server_args,
        device=get_device().device,
        dist_port=port_args.nccl_port,
    )
    bootstrap.init_layer_runtime(model_config=model_config)
    runner = ModelRunner(
        model_config=model_config,
        mem_fraction_static=cfg.mem_fraction_static,
        gpu_id=0,
        nccl_port=port_args.nccl_port,
        server_args=server_args,
    )
    runner.alloc_memory_pool()
    runner.init_attention_backends()
    return runner


def _chip_version() -> str:
    """The TPU generation as the artifact names it (`v7x`, `v6e`, ...)."""
    try:
        import jax

        kind = jax.devices()[0].device_kind.lower()
    except Exception:  # noqa: BLE001 - JAX may run CPU-only under torch_tpu
        kind = ""
    for name, tag in (("7x", "v7x"), ("v6", "v6e"), ("v5p", "v5p"), ("v5", "v5e")):
        if name in kind:
            return tag
    return "v7x"


@dataclasses.dataclass(frozen=True)
class ExportOptions:
    """Export choices that SGLang's server arguments do not carry."""

    token_buckets: tuple[int, ...]
    req_buckets: tuple[int, ...] | None = None
    target_tpu: str | None = None


class SglangExportableLLM(api.ExportableTextOnlyLLM):
    """A loaded SGLang `ModelRunner` as an exportable text-only LLM."""

    def __init__(self, model_runner: Any, options: ExportOptions) -> None:
        self._runner = model_runner
        self._model: nn.Module = model_runner.model
        self._backend = model_runner.attn_backend
        self._pool = model_runner.token_to_kv_pool
        self._check_supported()
        self._export_config = self._build_export_config(options)
        self._patched_rope_modules = rope_patch.patch_in_graph_rotary_embedding(
            self._model
        )
        self._forward_batches: dict[int, Any] = {}

    # ------------------------------------------------------------------ setup

    def _check_supported(self) -> None:
        from sglang_tpu.attention_pallas_rpa_base import BaseTpuPallasRpaAttnBackend

        if not isinstance(self._backend, BaseTpuPallasRpaAttnBackend):
            raise NotImplementedError(
                f"export supports the Pallas RPA attention backends, got "
                f"{type(self._backend).__name__}; pass --attention-backend "
                "pallas_tpu_rpa_v3"
            )
        if self._backend.kernel_spec.is_batched:
            raise NotImplementedError(
                "export supports the RPA v3 kernel (its distribution layout is the "
                "artifact's req_dist); pass --attention-backend pallas_tpu_rpa_v3"
            )
        tp = int(self._runner.server_args.tp_size)
        if tp != 1:
            raise NotImplementedError(
                f"export supports tensor parallel size 1, got {tp}"
            )
        state = self._pool.get_state()
        if not isinstance(state, list) or not state:
            raise NotImplementedError(
                f"export needs a fused per-layer KV pool, got "
                f"{type(self._pool).__name__}"
            )

    def _build_export_config(self, options: ExportOptions) -> api.TextOnlyLLMExportConfig:
        runner = self._runner
        cfg = runner.model_config
        hf = cfg.hf_config
        server_args = runner.server_args
        kv = self._pool.get_state()[0]
        page_size = int(kv.shape[1])
        if page_size != int(runner.page_size):
            raise ValueError(f"KV page size {page_size} != runner page size {runner.page_size}")
        max_num_seqs = int(runner.req_to_token_pool.size)
        max_model_len = int(cfg.context_len)
        eos = getattr(hf, "eos_token_id", None)
        if isinstance(eos, (list, tuple)):
            eos = eos[0]
        return api.TextOnlyLLMExportConfig(
            model_name=str(server_args.served_model_name or server_args.model_path),
            target_tpu=options.target_tpu or _chip_version(),
            num_replicas=1,
            num_partitions=1,
            use_spmd_partitioning=True,
            vocab_size=int(cfg.vocab_size),
            hidden_size=int(cfg.hidden_size),
            num_layers=len(self._pool.get_state()),
            num_q_heads=int(cfg.num_attention_heads),
            num_kv_heads=int(cfg.get_total_num_kv_heads()),
            head_dim=int(cfg.head_dim),
            block_size=page_size,
            num_blocks=int(kv.shape[0]),
            max_num_seqs=max_num_seqs,
            max_model_len=max_model_len,
            eos_token_id=int(eos) if eos is not None else 0,
            dtype=tpu_lowering.dtype_name(cfg.dtype),
            token_buckets=tuple(options.token_buckets),
            req_buckets=tuple(options.req_buckets or default_req_buckets(max_num_seqs)),
            hf_checkpoint_dir=str(server_args.model_path),
        )

    # --------------------------------------------------------------- protocol

    @property
    def model(self) -> nn.Module:
        return self._model

    @property
    def pallas_interpret(self) -> bool:
        return False

    @property
    def export_config(self) -> api.TextOnlyLLMExportConfig:
        return self._export_config

    def abstract_params(self) -> Mapping[str, api.ShapeDtypeStruct]:
        """Parameters, then buffers (constants), plus a tied LM head alias.

        At tensor parallel size 1 nothing is sharded.
        """
        from sglang_tpu.experimental.export import weight_mapper

        params: dict[str, api.ShapeDtypeStruct] = {}
        for name, param in self._model.named_parameters():
            params[name] = api.ShapeDtypeStruct(
                shape=tuple(param.shape), dtype=tpu_lowering.dtype_name(param.dtype)
            )
        for name, buf in self._model.named_buffers():
            params.setdefault(
                name,
                api.ShapeDtypeStruct(
                    shape=tuple(buf.shape), dtype=tpu_lowering.dtype_name(buf.dtype)
                ),
            )
        tied = weight_mapper.tied_lm_head_alias(params)
        if tied:
            params[tied[0]] = params[tied[1]]
        return params

    def constant_names(self) -> set[str]:
        """Names in `abstract_params()` that are buffers rather than parameters."""
        params = {name for name, _ in self._model.named_parameters()}
        return {name for name, _ in self._model.named_buffers() if name not in params}

    def constant_axis_values(self, name: str) -> list[Any]:
        """The buffer's value on every tensor-parallel rank (one rank here)."""
        buf = dict(self._model.named_buffers())[name]
        if buf.numel() > _MAX_CONSTANT_ELEMENTS:
            raise ValueError(
                f"buffer {name} has {buf.numel()} elements; constants are limited to "
                f"{_MAX_CONSTANT_ELEMENTS}"
            )
        local = buf.detach().cpu().flatten().tolist()
        return [local[0] if buf.dim() == 0 else local]

    def abstract_states(self) -> Sequence[api.ShapeDtypeStruct]:
        """Each layer's fused KV buffer, with `num_blocks` from the export config."""
        num_blocks = self._export_config.num_blocks
        return [
            api.ShapeDtypeStruct(
                shape=(num_blocks, *tuple(t.shape[1:])),
                dtype=tpu_lowering.dtype_name(t.dtype),
                sharding=(None,) * t.dim(),
            )
            for t in self._pool.get_state()
        ]

    def fit_num_blocks(self, num_blocks: int) -> None:
        """Declares `num_blocks` KV blocks per layer instead of SGLang's pool size."""
        if num_blocks < 2:
            raise ValueError(f"num_blocks must be >= 2 (page 0 is padding), got {num_blocks}")
        self._export_config = dataclasses.replace(self._export_config, num_blocks=num_blocks)

    # ---------------------------------------------------------------- forward

    def _pages_per_seq(self) -> int:
        cfg = self._export_config
        return max(1, math.ceil(cfg.max_model_len / cfg.block_size))

    def _forward_batch(self, num_tokens: int) -> Any:
        """A mixed (EXTEND) batch at `num_tokens` rows over `max_num_seqs` requests.

        Only its mode and shapes matter: the attention metadata comes from the
        artifact's inputs, and the model reads no other batch tensor on the
        dense decoder path.
        """
        cached = self._forward_batches.get(num_tokens)
        if cached is not None:
            return cached
        from sglang.srt.model_executor.forward_batch_info import (
            CaptureHiddenMode,
            ForwardBatch,
            ForwardMode,
        )

        bs = self._export_config.max_num_seqs
        device = self._runner.device
        zeros_bs = torch.zeros(bs, dtype=torch.int64, device=device)
        batch = ForwardBatch(
            forward_mode=ForwardMode.EXTEND,
            batch_size=bs,
            input_ids=torch.zeros(num_tokens, dtype=torch.int64, device=device),
            req_pool_indices=zeros_bs,
            req_pool_indices_cpu=torch.zeros(bs, dtype=torch.int64),
            seq_lens=torch.ones(bs, dtype=torch.int64, device=device),
            seq_lens_sum=bs,
            orig_seq_lens=torch.ones(bs, dtype=torch.int64, device=device),
            out_cache_loc=torch.zeros(num_tokens, dtype=torch.int64, device=device),
            seq_lens_cpu=torch.ones(bs, dtype=torch.int64),
            positions=torch.zeros(num_tokens, dtype=torch.int64, device=device),
            return_logprob=False,
            extend_num_tokens=num_tokens,
            extend_seq_lens=torch.ones(bs, dtype=torch.int32, device=device),
            extend_prefix_lens=torch.zeros(bs, dtype=torch.int32, device=device),
            extend_start_loc=torch.arange(bs, dtype=torch.int32, device=device),
            extend_seq_lens_cpu=[1] * bs,
            extend_prefix_lens_cpu=[0] * bs,
            extend_logprob_start_lens_cpu=[0] * bs,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            spec_algorithm=getattr(self._runner, "spec_algorithm", None),
        )
        prepare = getattr(self._model, "prepare_forward_batch", None)
        if callable(prepare):
            prepare(batch)
        self._forward_batches[num_tokens] = batch
        return batch

    def _metadata(
        self,
        *,
        kv_lens: Any,
        block_tables: Any,
        query_start_loc: Any,
        req_dist: Any,
    ) -> Any:
        from sglang_tpu.attention_pallas_rpa_base import PallasForwardMetadata

        cfg = self._export_config
        stride = self._pages_per_seq() * cfg.block_size
        cu_kv_lens = (
            torch.arange(cfg.max_num_seqs + 1, dtype=torch.int32, device=kv_lens.device)
            * stride
        )
        active = query_start_loc.index_select(0, req_dist[2:3].to(torch.int64)).reshape(())
        return PallasForwardMetadata(
            kv_lens=kv_lens,
            cu_q_lens=query_start_loc,
            cu_kv_lens=cu_kv_lens,
            page_indices=block_tables,
            distribution=req_dist,
            active_num_tokens=active,
        )

    @contextlib.contextmanager
    def _bound(self, weights: Mapping[str, Any], kv_caches: Sequence[Any]) -> Iterator[None]:
        """Binds `weights` into the model and `kv_caches` into the KV pool."""
        from torch.nn.utils.stateless import _reparametrize_module

        call_weights = {
            k: v
            for k, v in weights.items()
            if not k.startswith("lm_head.") and isinstance(v, torch.Tensor)
        }
        saved_state = self._pool.get_state()
        self._pool.set_state(list(kv_caches))
        try:
            with _reparametrize_module(self._model, call_weights, strict=False):
                yield
        finally:
            self._pool.set_state(saved_state)

    def forward(
        self,
        weights: Mapping[str, Any],
        kv_caches: Sequence[Any],
        *,
        input_ids: Any,
        positions: Any,
        kv_lens: Any,
        query_start_loc: Any,
        req_dist: Any,
        block_tables: Any = None,
        **extra_controls: Any,
    ) -> tuple[Sequence[Any], Any]:
        """One continuous-batching step: SGLang's decoder over the artifact's inputs."""
        if extra_controls:
            raise NotImplementedError(
                f"SGLang export takes no extra controls, got {sorted(extra_controls)}"
            )
        from sglang.srt.model_executor.forward_context import (
            ForwardContext,
            forward_context,
        )

        num_tokens = int(input_ids.shape[0])
        batch = self._forward_batch(num_tokens)
        metadata = self._metadata(
            kv_lens=kv_lens,
            block_tables=block_tables,
            query_start_loc=query_start_loc,
            req_dist=req_dist,
        )
        backbone = self._model.model
        with (
            self._bound(weights, kv_caches),
            self._backend.bind_forward_metadata(metadata, batch.forward_mode),
            forward_context(ForwardContext(attn_backend=self._backend)),
        ):
            hidden = backbone(
                input_ids.to(torch.int64), positions.to(torch.int64), batch
            )
            if isinstance(hidden, tuple):
                hidden = hidden[0]
            updated = list(self._pool.get_state())
        return updated, hidden

    def compute_logits(self, hidden_states: Any, lm_head_weight: Any) -> Any:
        """Logits as SGLang's `LogitsProcessor` computes them, trimmed to the vocabulary.

        Runs the processor's own LM head projection (fp32 when the model asks for
        it) with `lm_head_weight` bound to the head, then its logit scale and final
        soft-capping.
        """
        from torch.nn.utils.stateless import _reparametrize_module

        processor = self._model.logits_processor
        head = self._model.lm_head
        with _reparametrize_module(head, {"weight": lm_head_weight}):
            logits = processor._compute_lm_head_rows(hidden_states, head)
        if processor.logit_scale is not None:
            logits = logits * processor.logit_scale
        cap = processor.final_logit_softcapping
        if cap:
            logits = cap * torch.tanh(logits / cap)
        return logits[..., : self._export_config.vocab_size]

    def sample_tokens(self, logits: Any, **kwargs: Any) -> Any:
        """Samples token ids `i64[R, 1]` from `logits[R, vocab_size]`."""
        return sampling.dispatch_sample_tokens(logits, **kwargs)
