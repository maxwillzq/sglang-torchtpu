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

Under tensor (and expert) parallelism every rank runs this module in its own
process (`cli`): each traces its own shard-local program, collectives included,
and the artifact records each tensor's sharding over the `model` mesh axis
(`abstract_params`, `abstract_states`). The LM head's vocabulary shards are
gathered in `compute_logits`, so every rank returns full logits.

Hybrid models (Gated DeltaNet linear-attention layers between full-attention
layers, e.g. Qwen3.5) run SGLang's `HybridLinearAttnBackend`: the full layers'
KV caches stay as above, and each linear layer adds its conv and recurrent state
`[num_state_slots, ...]`, null slot 0, selected per request by the
`mamba_state_indices` input. The artifact lists the states per layer in layer
order (`[L0.conv, L0.rec, ..., L3.kv, ...]`, as vLLM's export does); SGLang keeps
them as the full KV caches, then every conv, then every recurrent state, so the
adapter permutes between the two.

Supported: text decoders (dense, MoE, or hybrid GDN without speculative
decoding) on the RPA v3 backend.
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


def prepare_server_args(server_args: Any) -> Any:
    """Resolves `server_args` and picks the ports, once, in the launching process.

    As SGLang's engine does before it starts its schedulers: every rank then gets
    the same resolved arguments and the same `PortArgs`.
    """
    from sglang.srt.entrypoints.engine import _set_envs_and_config
    from sglang.srt.server_args import PortArgs

    server_args.resolve_once()
    _set_envs_and_config(server_args)
    return PortArgs.init_new(server_args)


def load_model_runner(server_args: Any, tp_rank: int = 0, port_args: Any = None) -> Any:
    """Loads the model of `server_args` on this TPU without a server or scheduler.

    The single-rank steps of `sglang.benchmark.one_batch.load_model`: resolve the
    server arguments, publish this process as rank 0, initialize the TPU worker
    and the parallel runtime, then build the `ModelRunner` with its KV pool and
    attention backends. No graph runner is built: the exporter traces the
    forward itself. Under tensor parallelism every rank calls this with its own
    `tp_rank` and the same `server_args.nccl_port`.
    """
    from sglang.srt.arg_groups.overrides import resolving_view
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.distributed import bootstrap
    from sglang.srt.layers.moe import initialize_moe_config
    from sglang.srt.layers.quantization.fp4_utils import initialize_fp4_gemm_config
    from sglang.srt.layers.quantization.fp8_utils import initialize_fp8_gemm_config
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.runtime_context import (
        SpawnRanks,
        get_device,
        publish,
        spawn_world_rank,
    )
    from sglang.srt.utils.patch_torch import initialize_torch_tpu_worker

    if port_args is None:
        port_args = prepare_server_args(server_args)
    publish(
        server_args,
        role="scheduler",
        ranks=SpawnRanks(
            world_rank=spawn_world_rank(server_args, tp_rank=tp_rank, pp_rank=0),
            gpu_id=tp_rank,
        ),
    )
    initialize_moe_config()
    initialize_fp8_gemm_config()
    initialize_fp4_gemm_config()

    initialize_torch_tpu_worker(server_args, tp_rank)
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
        gpu_id=tp_rank,
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
        self._hybrid = _is_hybrid_backend(self._backend)
        if self._hybrid:
            from sglang_tpu.compilation.hybrid_linear_direct_compile import (
                hybrid_linear_kv_pool_state,
            )

            self._rpa = self._backend.full_attn_backend
            self._pool = hybrid_linear_kv_pool_state(
                kv_pool=model_runner.token_to_kv_pool,
                req_to_token_pool=model_runner.req_to_token_pool,
            )
        else:
            self._rpa = self._backend
            self._pool = model_runner.token_to_kv_pool
        self._layout = self._state_layout()
        self._check_supported()
        self._export_config = self._build_export_config(options)
        self._patched_rope_modules = rope_patch.patch_in_graph_rotary_embedding(
            self._model
        )
        self._patched_embeddings = _patch_vocab_shard_bounds(self._model)
        self._forward_batches: dict[int, Any] = {}

    # ------------------------------------------------------------------ setup

    def _check_supported(self) -> None:
        from sglang_tpu.attention_pallas_rpa_base import BaseTpuPallasRpaAttnBackend

        if not isinstance(self._rpa, BaseTpuPallasRpaAttnBackend):
            raise NotImplementedError(
                f"export supports the Pallas RPA attention backends, got "
                f"{type(self._rpa).__name__}; pass --attention-backend "
                "pallas_tpu_rpa_v3"
            )
        if self._rpa.kernel_spec.is_batched:
            raise NotImplementedError(
                "export supports the RPA v3 kernel (its distribution layout is the "
                "artifact's req_dist); pass --attention-backend pallas_tpu_rpa_v3"
            )
        state = self._pool.get_state()
        if not isinstance(state, list) or not state:
            raise NotImplementedError(
                f"export needs a fused per-layer KV pool, got "
                f"{type(self._pool).__name__}"
            )

    def _tp_size(self) -> int:
        return int(self._runner.server_args.tp_size)

    def _state_layout(self) -> list[tuple[str, int]]:
        """The artifact's states in order, each as (kind, index into SGLang's list).

        Kinds: `kv` (a full-attention layer's KV cache), `conv` and `rec` (a
        linear layer's conv and recurrent state). SGLang's list is the full KV
        caches, then each conv kind's states, then the recurrent states.
        """
        state = self._pool.get_state()
        if not self._hybrid:
            return [("kv", i) for i in range(len(state))]
        full = self._runner.token_to_kv_pool.full_attention_layer_id_mapping
        linear = self._runner.req_to_token_pool.mamba_map
        num_full, num_linear = len(full), len(linear)
        if len(state) != num_full + 2 * num_linear:
            raise NotImplementedError(
                f"export supports one conv state per linear layer (Gated DeltaNet); "
                f"got {len(state)} state tensors for {num_full} full and "
                f"{num_linear} linear layers"
            )
        layout: list[tuple[str, int]] = []
        for layer_id in sorted({*full, *linear}):
            if layer_id in full:
                layout.append(("kv", full[layer_id]))
            else:
                j = linear[layer_id]
                layout += [("conv", num_full + j), ("rec", num_full + num_linear + j)]
        return layout

    def _kv_tensor(self) -> torch.Tensor:
        state = self._pool.get_state()
        return next(state[i] for kind, i in self._layout if kind == "kv")

    def _build_export_config(
        self, options: ExportOptions
    ) -> api.TextOnlyLLMExportConfig:
        runner = self._runner
        cfg = runner.model_config
        hf = cfg.hf_config
        server_args = runner.server_args
        kv = self._kv_tensor()
        page_size = int(kv.shape[1])
        if page_size != int(runner.page_size):
            raise ValueError(
                f"KV page size {page_size} != runner page size {runner.page_size}"
            )
        max_num_seqs = int(runner.req_to_token_pool.size)
        max_model_len = int(cfg.context_len)
        eos = getattr(hf, "eos_token_id", None)
        if isinstance(eos, (list, tuple)):
            eos = eos[0]
        return api.TextOnlyLLMExportConfig(
            model_name=str(server_args.served_model_name or server_args.model_path),
            target_tpu=options.target_tpu or _chip_version(),
            num_replicas=1,
            num_partitions=self._tp_size(),
            use_spmd_partitioning=True,
            vocab_size=int(cfg.vocab_size),
            hidden_size=int(cfg.hidden_size),
            num_layers=sum(1 for kind, _ in self._layout if kind != "rec"),
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
        """Parameters (shard-local shapes, with their sharding), then buffers
        (constants), plus a tied LM head alias."""
        from sglang_tpu.experimental.export import weight_mapper

        tp = self._tp_size()
        named = list(self._model.named_parameters())
        has_ep = any(_is_expert_param(n) for n, _ in named) and self._ep_size() > 1
        params: dict[str, api.ShapeDtypeStruct] = {}
        for name, param in named:
            shape = tuple(int(d) for d in param.shape)
            sharding = _moe_sharding(self._model, name, shape, self._ep_size())
            if sharding is None:
                sharding = _attr_sharding(self._model, name, param, len(shape), tp)
            if sharding is None:
                sharding = weight_mapper.infer_weight_sharding(
                    name, shape, model=self._model, tp_size_hint=tp, has_ep_hint=has_ep
                )
            params[name] = api.ShapeDtypeStruct(
                shape=shape,
                dtype=tpu_lowering.dtype_name(param.dtype),
                sharding=sharding,
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

    def _ep_size(self) -> int:
        return int(getattr(self._runner.server_args, "ep_size", 1) or 1)

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
        value = local[0] if buf.dim() == 0 else local
        if self._tp_size() == 1:
            return [value]
        from sglang.srt.distributed import get_tp_group

        group = get_tp_group()
        values: list[Any] = [None] * group.world_size
        torch.distributed.all_gather_object(values, value, group=group.cpu_group)
        return values

    def abstract_states(self) -> Sequence[api.ShapeDtypeStruct]:
        """The artifact's state tensors in layer order (`_state_layout`).

        A KV cache is `[num_blocks, ...]`; under tensor parallelism each rank holds
        its KV heads, so the fused heads axis (dim 2) is sharded over `model`, as
        in vLLM's export. A linear layer's conv state `[num_state_slots, K-1, 1,
        conv_dim]` is sharded on its channels and its recurrent state
        `[num_state_slots, n_v, d_k, d_v]` on its value heads.
        """
        cfg = self._export_config
        tp = self._tp_size() > 1
        state = self._pool.get_state()
        specs = []
        for kind, i in self._layout:
            t = state[i]
            if kind == "kv":
                lead, shard = (
                    cfg.num_blocks,
                    (None, None, "model" if tp else None, None, None),
                )
            elif kind == "conv":
                lead, shard = (
                    cfg.num_state_slots,
                    (None, None, None, "model" if tp else None),
                )
            else:
                lead, shard = (
                    cfg.num_state_slots,
                    (None, "model" if tp else None, None, None),
                )
            specs.append(
                api.ShapeDtypeStruct(
                    shape=(lead, *tuple(t.shape[1:])),
                    dtype=tpu_lowering.dtype_name(t.dtype),
                    sharding=shard[: t.dim()],
                )
            )
        return specs

    def extra_user_inputs(self) -> list[tuple[str, tuple[int, ...]]]:
        """Hybrid models take each request's recurrent state slot after the states."""
        if not self._hybrid:
            return []
        return [("mamba_state_indices", (self._export_config.max_num_seqs,))]

    def fit_num_blocks(self, num_blocks: int) -> None:
        """Declares `num_blocks` KV blocks per layer instead of SGLang's pool size."""
        if num_blocks < 2:
            raise ValueError(
                f"num_blocks must be >= 2 (page 0 is padding), got {num_blocks}"
            )
        self._export_config = dataclasses.replace(
            self._export_config, num_blocks=num_blocks
        )

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
        active = query_start_loc.index_select(0, req_dist[2:3].to(torch.int64)).reshape(
            ()
        )
        return PallasForwardMetadata(
            kv_lens=kv_lens,
            cu_q_lens=query_start_loc,
            cu_kv_lens=cu_kv_lens,
            page_indices=block_tables,
            distribution=req_dist,
            active_num_tokens=active,
        )

    @contextlib.contextmanager
    def _bound(
        self, weights: Mapping[str, Any], kv_caches: Sequence[Any]
    ) -> Iterator[None]:
        """Binds `weights` into the model and `kv_caches` into the KV pool."""
        from torch.nn.utils.stateless import _reparametrize_module

        call_weights = {
            k: v
            for k, v in weights.items()
            if not k.startswith("lm_head.") and isinstance(v, torch.Tensor)
        }
        call_weights.update(_aliases(self._model, call_weights))
        saved_state = self._pool.get_state()
        self._pool.set_state(self._to_sglang_order(kv_caches))
        try:
            with contextlib.ExitStack() as stack:
                stack.enter_context(
                    _reparametrize_module(self._model, call_weights, strict=False)
                )
                if self._hybrid:
                    from sglang_tpu.compilation.hybrid_linear_direct_compile import (
                        bind_traced_conv_weights,
                    )

                    # The linear layers' conv_weights view conv1d.weight: re-derive
                    # them from the bound weight.
                    stack.enter_context(bind_traced_conv_weights(self._model))
                yield
        finally:
            self._pool.set_state(saved_state)

    def _to_sglang_order(self, states: Sequence[Any]) -> list[Any]:
        out: list[Any] = [None] * len(self._layout)
        for t, (_, i) in zip(states, self._layout, strict=True):
            out[i] = t
        return out

    def _from_sglang_order(self, states: Sequence[Any]) -> list[Any]:
        return [states[i] for _, i in self._layout]

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
        mamba_state_indices = extra_controls.pop("mamba_state_indices", None)
        if extra_controls:
            raise NotImplementedError(
                f"SGLang export takes no extra controls besides mamba_state_indices, "
                f"got {sorted(extra_controls)}"
            )
        if self._hybrid and mamba_state_indices is None:
            raise ValueError("a hybrid model's forward needs mamba_state_indices")
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
            self._bind_metadata(metadata, mamba_state_indices, batch.forward_mode),
            forward_context(ForwardContext(attn_backend=self._backend)),
        ):
            hidden = backbone(
                input_ids.to(torch.int64), positions.to(torch.int64), batch
            )
            if isinstance(hidden, tuple):
                hidden = hidden[0]
            updated = self._from_sglang_order(self._pool.get_state())
        return updated, hidden

    def _bind_metadata(
        self, rpa_metadata: Any, mamba_state_indices: Any, mode: Any
    ) -> Any:
        """Installs the attention metadata (and, for a hybrid model, the GDN's)."""
        if not self._hybrid:
            return self._rpa.bind_forward_metadata(rpa_metadata, mode)
        from sglang_tpu.attention_gdn_pallas import GdnTracedMetadata
        from sglang_tpu.compilation.hybrid_linear_direct_compile import (
            bind_traced_attention_metadata,
        )

        req_dist = rpa_metadata.distribution
        # Every request in the GDN's per-sequence segment: the backend runs one
        # kernel segment per forward (`prefill_only` outside decode), so a decode
        # range [0, req_dist[0]) would be skipped. Correct for decodes too, if
        # slower than the batched decode segment.
        # TODO: a mode of TpuGDNAttnBackend that runs both segments.
        zeros = torch.zeros(2, dtype=req_dist.dtype, device=req_dist.device)
        gdn = GdnTracedMetadata(
            distribution=torch.cat([zeros, req_dist[2:3]]),
            seq_lens=rpa_metadata.kv_lens,
            state_indices=mamba_state_indices.to(torch.int32),
            query_start_loc=rpa_metadata.cu_q_lens,
            track_indices=None,
            track_end=None,
            read_offsets=None,
        )
        return bind_traced_attention_metadata(self._backend, (rpa_metadata, gdn), mode)

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
        if self._tp_size() > 1:
            # Each rank projects onto its vocabulary shard; gather them, as the
            # processor's own gatherer does, so every rank returns full logits.
            from sglang.srt.distributed import tensor_model_parallel_all_gather

            logits = tensor_model_parallel_all_gather(logits, dim=-1)
        cap = processor.final_logit_softcapping
        if cap:
            logits = cap * torch.tanh(logits / cap)
        return logits[..., : self._export_config.vocab_size]

    def sample_tokens(self, logits: Any, **kwargs: Any) -> Any:
        """Samples token ids `i64[R, 1]` from `logits[R, vocab_size]`."""
        return sampling.dispatch_sample_tokens(logits, **kwargs)


_SCALE_SUFFIXES = ("_scale_inv", "_scale")
# Parameter attributes SGLang's parallel layers set on the dim they split.
_SHARD_ATTRS = (
    ("ep_dim", "ep"),
    ("tp_dim", "tp"),
    ("output_dim", "tp"),
    ("input_dim", "tp"),
)


def _attr_sharding(
    model: nn.Module, name: str, param: Any, rank: int, tp_size: int
) -> list[str | None] | None:
    """Sharding of an SGLang parallel-layer parameter, from its layer and attributes.

    SGLang keeps linear weights as `[out, in]` and sets both `output_dim` and
    `input_dim` on them, so the attribute alone does not say which one is split:
    the layer does. A column-parallel layer has `output_size_per_partition <
    output_size` (split `output_dim`), a row-parallel one `input_size_per_partition
    < input_size` (split `input_dim`); vocabulary embeddings and heads split
    `output_dim`. (vllm-torchtpu transposes its weights; `weight_mapper` flips the
    attributes for that and does not apply here.) None when the parameter has no
    shard attribute.
    """
    if rank == 0:
        return None
    dims = {attr: getattr(param, attr, None) for attr, _ in _SHARD_ATTRS}
    dims = {a: d for a, d in dims.items() if isinstance(d, int) and 0 <= d < rank}
    if not dims:
        return None
    sharding: list[str | None] = [None] * rank
    if "ep_dim" in dims:
        sharding[dims["ep_dim"]] = "ep"
        return sharding
    if tp_size <= 1:
        return sharding
    if "tp_dim" in dims:
        sharding[dims["tp_dim"]] = "tp"
        return sharding
    module = dict(model.named_modules()).get(name.rsplit(".", 1)[0])
    split = None
    in_full, in_part = (
        getattr(module, k, None) for k in ("input_size", "input_size_per_partition")
    )
    out_full, out_part = (
        getattr(module, k, None) for k in ("output_size", "output_size_per_partition")
    )
    if in_full is not None and in_part is not None and in_part < in_full:
        split = "input_dim"
    elif out_full is not None and out_part is not None and out_part < out_full:
        split = "output_dim"
    elif hasattr(module, "num_embeddings_per_partition") or hasattr(
        module, "org_vocab_size"
    ):
        split = "output_dim"
    if split in dims:
        sharding[dims[split]] = "tp"
    return sharding


def _is_expert_param(name: str) -> bool:
    leaf = name.rsplit(".", 1)[-1]
    in_experts = name.startswith("experts.") or ".experts." in name
    return in_experts and leaf.split("_", 1)[0] in ("w13", "w1", "w2", "w3")


def _moe_sharding(
    model: nn.Module, name: str, shape: tuple[int, ...], ep_size: int
) -> list[str | None] | None:
    """Sharding of a fused-MoE expert tensor, or None for anything else.

    SGLang keeps a MoE layer's experts as stacked `w13_*` (gate and up; `w1`/`w3`
    when separate) and `w2_*` (down) tensors, `[num_experts, ...]`, with their
    scales beside them. With expert parallelism (the layer's `moe_ep_size`) each
    rank holds a slice of the experts, dim 0. Otherwise tensor parallelism splits
    the intermediate dimension: the dim whose size is the layer's per-rank
    intermediate size (twice that for `w13`), found by size because the TPU MoE
    kernels may keep the weights transposed. A scale tensor is sharded like its
    weight.
    """
    del ep_size
    if not _is_expert_param(name) or len(shape) < 2:
        return None
    module_name, leaf = name.rsplit(".", 1)
    module = dict(model.named_modules()).get(module_name)
    if module is None:
        return None
    sharding: list[str | None] = [None] * len(shape)
    if int(getattr(module, "moe_ep_size", 1) or 1) > 1:
        sharding[0] = "ep"
        return sharding
    if int(getattr(module, "moe_tp_size", 1) or 1) <= 1:
        return sharding
    weight_leaf = leaf
    for suffix in _SCALE_SUFFIXES:
        if leaf.endswith(suffix):
            weight_leaf = leaf[: -len(suffix)]
            break
    weight = getattr(module, weight_leaf, None)
    weight_shape = tuple(weight.shape) if isinstance(weight, torch.Tensor) else shape
    per_rank = getattr(module, "intermediate_size_per_partition", None)
    if per_rank is None:
        return sharding
    target = 2 * int(per_rank) if weight_leaf.startswith("w13") else int(per_rank)
    dims = [d for d in range(1, len(weight_shape)) if weight_shape[d] == target]
    if len(dims) != 1:
        raise NotImplementedError(
            f"cannot tell which dim of {name} {shape} tensor parallelism splits "
            f"(per-rank intermediate size {per_rank})"
        )
    if dims[0] < len(shape):
        sharding[dims[0]] = "tp"
    return sharding


_VOCAB_BOUNDS = (
    "org_vocab_start_index",
    "org_vocab_end_index",
    "num_org_vocab_padding",
    "added_vocab_start_index",
    "added_vocab_end_index",
)


def _patch_vocab_shard_bounds(model: nn.Module) -> int:
    """Makes each vocab-parallel embedding read its shard bounds from buffers.

    SGLang keeps a rank's vocabulary range as Python ints (`shard_indices`), which
    tracing bakes into the graph, so rank 0's program would mask every rank's
    tokens with rank 0's range. The bounds become int32 scalar buffers instead:
    the exporter turns per-rank buffer values into a constant input sharded over
    the mesh (`constant_axis_values`), so each device reads its own. Only
    embeddings that look tokens up are patched; an LM head projects in
    `compute_logits` and needs no bounds.
    """
    from sglang.srt.layers.vocab_parallel_embedding import (
        ParallelLMHead,
        VocabParallelEmbedding,
        get_masked_input_and_mask,
    )

    patched = 0
    for _, module in model.named_modules():
        if not isinstance(module, VocabParallelEmbedding) or isinstance(
            module, ParallelLMHead
        ):
            continue
        if int(getattr(module, "tp_size", 1) or 1) <= 1:
            continue
        device = module.weight.device
        for bound in _VOCAB_BOUNDS:
            value = int(getattr(module.shard_indices, bound))
            module.register_buffer(
                bound,
                torch.tensor(value, dtype=torch.int32, device=device),
                persistent=False,
            )

        def embed_local_shard(
            input_: torch.Tensor, module: Any = module
        ) -> torch.Tensor:
            masked, mask = get_masked_input_and_mask(
                input_, *(getattr(module, b) for b in _VOCAB_BOUNDS)
            )
            out = module.quant_method.embedding(module, masked.long())
            out = out.masked_fill(mask.unsqueeze(-1), 0)
            if module.output_dtype is not None:
                out = out.to(module.output_dtype)
            return out

        module._embed_local_shard = embed_local_shard
        patched += 1
    return patched


def _is_hybrid_backend(backend: Any) -> bool:
    try:
        from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
            HybridLinearAttnBackend,
        )
    except ImportError:
        return False
    return isinstance(backend, HybridLinearAttnBackend)


def _aliases(model: nn.Module, weights: Mapping[str, Any]) -> dict[str, Any]:
    """`weights` under every other name the same parameter or buffer has.

    A module may hold another module's parameter as its own (SGLang's
    `RadixLinearAttention` keeps its layer's `A_log`, `dt_bias` and conv bias), so
    the tensor is registered twice; `named_parameters()` lists it once, and binding
    only that name would leave the other reading the eager tensor.
    """
    out: dict[str, Any] = {}
    for named in (model.named_parameters, model.named_buffers):
        canonical = {id(t): n for n, t in named()}
        for name, tensor in named(remove_duplicate=False):
            first = canonical.get(id(tensor))
            if (
                first is not None
                and first != name
                and first in weights
                and name not in weights
            ):
                out[name] = weights[first]
    return out
