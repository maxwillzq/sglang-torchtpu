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

With MTP speculative decoding (`--speculative-algorithm NEXTN`), SGLang's MTP
draft is loaded beside the target and exported as `draft_bucket{B}` with vLLM's
semantics (`mtp_draft`); the draft's own weights are named `mtp.*`, and the
target's embedding and LM head are shared. A hybrid target then keeps
`recurrent_window` consecutive state slots per request, as the artifact contract
lays them out: the forward reads each request's state at `mamba_state_indices +
mamba_slot_read_offsets[mamba_state_indices]` and writes verify checkpoint `t` to
`mamba_state_indices + t`, bypassing SGLang's request-row window layout.

Supported: text decoders (dense, MoE, hybrid GDN, with or without MTP drafting)
on the RPA v3 backend.
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


def mtp_draft_names(model: nn.Module) -> dict[str, str]:
    """Export name of each draft tensor, by the draft's own name.

    A tensor the draft shares with the target (its embedding, a tied or shared LM
    head) keeps the target's name; the draft's own tensors are `mtp.` plus their
    name without a leading `model.`, as vLLM's export names them.
    """
    draft = getattr(model, MTP_DRAFT_ATTR, None)
    if draft is None:
        return {}
    target: dict[int, str] = {}
    for named in (model.named_parameters, model.named_buffers):
        for n, t in named():
            target.setdefault(id(t), n)
    names: dict[str, str] = {}
    for named in (draft.named_parameters, draft.named_buffers):
        for n, t in named(remove_duplicate=False):
            names.setdefault(n, target.get(id(t), "mtp." + n.removeprefix("model.")))
    return names


def mtp_draft_tensors(model: nn.Module) -> dict[str, Any]:
    """The MTP draft's own tensors (not shared with the target), by export name."""
    draft = getattr(model, MTP_DRAFT_ATTR, None)
    if draft is None:
        return {}
    names = mtp_draft_names(model)
    out: dict[str, Any] = {}
    for named in (draft.named_parameters, draft.named_buffers):
        for n, t in named(remove_duplicate=False):
            if names[n].startswith("mtp."):
                out[names[n]] = t
    return out


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
    # MTP drafting: the draft worker is built before the target's pools and shares
    # them (page table and allocator), as SGLang's scheduler builds it.
    draft_worker = None
    if getattr(server_args, "speculative_algorithm", None):
        from sglang_tpu.experimental.speculative.mtp.mtp_draft_worker import (
            TpuMtpDraftWorker,
        )

        draft_worker = TpuMtpDraftWorker(runner)
    runner.alloc_memory_pool()
    if draft_worker is not None:
        draft_worker.alloc_memory_pool(
            memory_pool_config=runner.memory_pool_config,
            req_to_token_pool=runner.req_to_token_pool,
            token_to_kv_pool_allocator=runner.token_to_kv_pool_allocator,
        )
    runner.init_attention_backends()
    if draft_worker is not None:
        draft_worker.init_attention_backends()
        runner._export_draft_worker = draft_worker
        # Off the target's module tree, so the target's parameters stay its own.
        object.__setattr__(
            runner.model, MTP_DRAFT_ATTR, draft_worker.model_runner.model
        )
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
        self._draft_worker = getattr(model_runner, "_export_draft_worker", None)
        self._draft: nn.Module | None = None
        if self._draft_worker is not None:
            draft_runner = self._draft_worker.model_runner
            self._draft = draft_runner.model
            self._draft_backend = draft_runner.attn_backend
            # A hybrid target wraps the draft's one attention layer in a hybrid backend
            # and pool too; the draft owns only their full-attention part.
            self._draft_rpa = getattr(
                self._draft_backend, "full_attn_backend", self._draft_backend
            )
            pool = draft_runner.token_to_kv_pool
            self._draft_pool = getattr(pool, "full_kv_pool", pool)
        self._layout = self._state_layout()
        self._check_supported()
        self._export_config = self._build_export_config(options)
        self._patched_rope_modules = rope_patch.patch_in_graph_rotary_embedding(
            self._model
        )
        self._patched_embeddings = _patch_vocab_shard_bounds(self._model)
        if self._draft is not None:
            self._patched_rope_modules += rope_patch.patch_in_graph_rotary_embedding(
                self._draft
            )
            self._patched_embeddings += _patch_vocab_shard_bounds(self._draft)
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
        speculative_config = None
        max_req = max_num_seqs
        if self._draft is not None:
            k = int(server_args.speculative_num_steps)
            num_draft_tokens = getattr(
                server_args, "speculative_num_draft_tokens", None
            )
            if num_draft_tokens is not None and int(num_draft_tokens) != k + 1:
                raise ValueError(
                    f"MTP export drafts chains: --speculative-num-draft-tokens must be "
                    f"--speculative-num-steps + 1 ({k + 1}), got {num_draft_tokens}"
                )
            speculative_config = {"method": "mtp", "num_speculative_tokens": k}
            # compute_logits and the samplers see up to k + 1 rows per request.
            max_req = min(max_num_seqs * (k + 1), max(options.token_buckets))
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
            req_buckets=tuple(options.req_buckets or default_req_buckets(max_req)),
            hf_checkpoint_dir=str(server_args.model_path),
            speculative_config=speculative_config,
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
        params.update(self._mtp_abstract_params())
        return params

    def _mtp_abstract_params(self) -> dict[str, api.ShapeDtypeStruct]:
        """The draft's own parameters and buffers by their `mtp.` names; shardings
        come from the draft's own layers."""
        from sglang_tpu.experimental.export import weight_mapper

        if self._draft is None:
            return {}
        tp = self._tp_size()
        names = mtp_draft_names(self._model)
        params: dict[str, api.ShapeDtypeStruct] = {}
        for name, param in self._draft.named_parameters():
            if not names[name].startswith("mtp."):
                continue
            shape = tuple(int(d) for d in param.shape)
            sharding = _moe_sharding(self._draft, name, shape, self._ep_size())
            if sharding is None:
                sharding = _attr_sharding(self._draft, name, param, len(shape), tp)
            if sharding is None:
                sharding = weight_mapper.infer_weight_sharding(
                    name, shape, model=self._draft, tp_size_hint=tp
                )
            params[names[name]] = api.ShapeDtypeStruct(
                shape=shape,
                dtype=tpu_lowering.dtype_name(param.dtype),
                sharding=sharding,
            )
        for name, buf in self._draft.named_buffers():
            if names[name].startswith("mtp."):
                params.setdefault(
                    names[name],
                    api.ShapeDtypeStruct(
                        shape=tuple(buf.shape), dtype=tpu_lowering.dtype_name(buf.dtype)
                    ),
                )
        return params

    def _buffers(self) -> dict[str, Any]:
        """The target's buffers and the draft's own, by export name."""
        bufs = dict(self._model.named_buffers())
        bufs.update(
            {n: t for n, t in mtp_draft_tensors(self._model).items() if n not in bufs}
        )
        return bufs

    def _lm_head_name(self) -> str:
        """The export name of the LM head weight (a tied head's alias)."""
        from sglang_tpu.experimental.export import weight_mapper

        names = [n for n, _ in self._model.named_parameters()]
        tied = weight_mapper.tied_lm_head_alias(names)
        return tied[0] if tied else "lm_head.weight"

    def _ep_size(self) -> int:
        return int(getattr(self._runner.server_args, "ep_size", 1) or 1)

    def constant_names(self) -> set[str]:
        """Names in `abstract_params()` that are buffers rather than parameters."""
        params = {name for name, _ in self._model.named_parameters()}
        if self._draft is not None:
            names = mtp_draft_names(self._model)
            params |= {names[n] for n, _ in self._draft.named_parameters()}
        return {name for name in self._buffers() if name not in params}

    def constant_axis_values(self, name: str) -> list[Any]:
        """The buffer's value on every tensor-parallel rank (one rank here)."""
        buf = self._buffers()[name]
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

    def mtp_abstract_states(self) -> Sequence[api.ShapeDtypeStruct]:
        """The draft's paged KV caches, laid out like a target KV cache."""
        if self._draft is None:
            raise ValueError("the model has no MTP draft")
        cfg = self._export_config
        tp = self._tp_size() > 1
        return [
            api.ShapeDtypeStruct(
                shape=(cfg.num_blocks, *tuple(t.shape[1:])),
                dtype=tpu_lowering.dtype_name(t.dtype),
                sharding=(None, None, "model" if tp else None, None, None)[: t.dim()],
            )
            for t in self._draft_pool.get_state()
        ]

    def extra_user_inputs(self) -> list[tuple[str, tuple[int, ...]]]:
        """Hybrid models take each request's recurrent state slot after the states;
        with verify windows also the per-slot read offsets and the checkpoint rows
        (the latter declared for the contract; the split-state path derives them)."""
        if not self._hybrid:
            return []
        cfg = self._export_config
        inputs = [("mamba_state_indices", (cfg.max_num_seqs,))]
        if cfg.recurrent_window > 1:
            inputs += [
                ("mamba_slot_read_offsets", (cfg.num_state_slots,)),
                ("mamba_ckpt_indices", (cfg.max_num_seqs, cfg.recurrent_window)),
            ]
        return inputs

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
        read_offsets = extra_controls.pop("mamba_slot_read_offsets", None)
        # Checkpoint t of a request's window is its state slot + t: nothing to read.
        extra_controls.pop("mamba_ckpt_indices", None)
        if extra_controls:
            raise NotImplementedError(
                f"SGLang export takes no extra controls besides the recurrent ones, "
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
        # The attention kernel's distribution; a 6-element req_dist adds the
        # recurrent kernels' after it.
        metadata = self._metadata(
            kv_lens=kv_lens,
            block_tables=block_tables,
            query_start_loc=query_start_loc,
            req_dist=req_dist[:3],
        )
        backbone = self._model.model
        with (
            self._bound(weights, kv_caches),
            self._bind_metadata(
                metadata,
                req_dist,
                mamba_state_indices,
                read_offsets,
                batch.forward_mode,
            ),
            forward_context(ForwardContext(attn_backend=self._backend)),
        ):
            hidden = backbone(
                input_ids.to(torch.int64), positions.to(torch.int64), batch
            )
            if isinstance(hidden, tuple):
                hidden = hidden[0]
            updated = self._from_sglang_order(self._pool.get_state())
        return updated, hidden

    @contextlib.contextmanager
    def _bind_metadata(
        self,
        rpa_metadata: Any,
        req_dist: Any,
        mamba_state_indices: Any,
        read_offsets: Any,
        mode: Any,
    ) -> Iterator[None]:
        """Installs the attention metadata (and, for a hybrid model, the GDN's)."""
        if not self._hybrid:
            with self._rpa.bind_forward_metadata(rpa_metadata, mode):
                yield
            return
        from sglang_tpu.attention_gdn_pallas import GdnTracedMetadata
        from sglang_tpu.compilation.hybrid_linear_direct_compile import (
            bind_traced_attention_metadata,
        )

        window = self._export_config.recurrent_window
        # The recurrent kernels' distribution: its windowed segment covers decodes
        # and verify windows, its per-sequence segment the rest.
        gdn_dist = req_dist[3:6] if req_dist.shape[0] == 6 else req_dist[:3]
        base = mamba_state_indices.to(torch.int32)
        offsets = None
        if window > 1:
            if read_offsets is None:
                raise ValueError(
                    "a forward with verify windows needs mamba_slot_read_offsets"
                )
            offsets = read_offsets.to(torch.int32).index_select(
                0, base.clamp(min=0).to(torch.int64)
            )
        gdn = GdnTracedMetadata(
            distribution=gdn_dist,
            seq_lens=rpa_metadata.kv_lens,
            state_indices=base,
            query_start_loc=rpa_metadata.cu_q_lens,
            track_indices=None,
            track_end=None,
            read_offsets=None,
        )
        with (
            bind_traced_attention_metadata(self._backend, (rpa_metadata, gdn), mode),
            _gdn_artifact_layout(
                self._backend.linear_attn_backend, base, offsets, window - 1
            ),
        ):
            yield

    # -------------------------------------------------------------- MTP draft

    def mtp_draft(
        self,
        weights: Mapping[str, Any],
        kv_caches: Sequence[Any],
        *,
        input_ids: Any,
        positions: Any,
        kv_lens: Any,
        block_tables: Any,
        query_start_loc: Any,
        req_dist: Any,
        last_token_indices: Any,
        hidden_states: Any,
    ) -> tuple[Sequence[Any], Any]:
        """Drafts `k` tokens per request with the MTP draft, as vLLM's export does.

        The first pass runs the draft over every row of the step (`input_ids`
        shifted by one token, the target's `hidden_states`), writing the draft's KV
        for each row; each request's first draft comes from its row
        `last_token_indices`. Each further pass feeds every request its last draft
        and the draft's own hidden state, one token per request at the next
        position.
        """
        from sglang.srt.model_executor.forward_context import (
            ForwardContext,
            forward_context,
        )
        from torch.nn.utils.stateless import _reparametrize_module

        draft = self._draft
        if draft is None:
            raise ValueError("the model has no MTP draft")
        cfg = self._export_config
        k = int(cfg.speculative_config["num_speculative_tokens"])  # type: ignore[index]
        names = mtp_draft_names(self._model)
        call = {
            d: weights[e]
            for d, e in names.items()
            if e in weights and isinstance(weights[e], torch.Tensor)
        }
        call.update(_aliases(draft, call))
        lm_head = weights[self._lm_head_name()]

        def run(ids: Any, pos: Any, hidden: Any, metadata: Any) -> Any:
            batch = self._forward_batch(int(ids.shape[0]))
            with self._draft_rpa.bind_forward_metadata(metadata, batch.forward_mode):
                ids = ids.to(torch.int64)
                # Qwen3_5ForCausalLMMTP.forward without its spec_info plumbing.
                emb = draft.pre_fc_norm_embedding(draft.model.embed_tokens(ids))
                h = draft.fc(torch.cat([emb, draft.pre_fc_norm_hidden(hidden)], dim=-1))
                out = draft.model(ids, pos.to(torch.int64), batch, h)
                return out[0] if isinstance(out, tuple) else out

        def greedy(hidden: Any) -> Any:
            return self.compute_logits(hidden, lm_head).argmax(dim=-1).to(torch.int32)

        saved = self._draft_pool.get_state()
        self._draft_pool.set_state(list(kv_caches))
        try:
            with (
                _reparametrize_module(draft, call, strict=False),
                forward_context(ForwardContext(attn_backend=self._draft_backend)),
            ):
                hidden = run(
                    input_ids,
                    positions,
                    hidden_states,
                    self._metadata(
                        kv_lens=kv_lens,
                        block_tables=block_tables,
                        query_start_loc=query_start_loc,
                        req_dist=req_dist[:3],
                    ),
                )
                rows = last_token_indices.to(torch.int64)
                hidden = hidden.index_select(0, rows)
                drafts = [greedy(hidden)]
                if k > 1:
                    num_seqs = req_dist[2]
                    live = torch.arange(cfg.max_num_seqs, device=rows.device) < num_seqs
                    # KV index of each request's row `last_token_indices`: its KV
                    # length less the rows after it in the step.
                    last_kv = (
                        kv_lens - (query_start_loc[1:] - rows.to(kv_lens.dtype) - 1) - 1
                    )
                    one_per_seq = torch.clamp(
                        torch.arange(cfg.max_num_seqs + 1, device=rows.device),
                        max=num_seqs,
                    ).to(query_start_loc.dtype)
                    all_decode = torch.stack([num_seqs, num_seqs, num_seqs]).to(
                        req_dist.dtype
                    )
                    last_pos = positions.index_select(-1, rows)
                    for step in range(1, k):
                        metadata = self._metadata(
                            kv_lens=torch.where(live, last_kv + step + 1, 0).to(
                                kv_lens.dtype
                            ),
                            block_tables=block_tables,
                            query_start_loc=one_per_seq,
                            req_dist=all_decode,
                        )
                        hidden = run(drafts[-1], last_pos + step, hidden, metadata)
                        drafts.append(greedy(hidden))
                updated = list(self._draft_pool.get_state())
        finally:
            self._draft_pool.set_state(saved)
        return updated, torch.stack(drafts, dim=1)

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


@contextlib.contextmanager
def _gdn_artifact_layout(
    backend: Any, base: Any, offsets: Any, num_spec_tokens: int
) -> Iterator[None]:
    """Runs SGLang's GDN backend on the artifact's recurrent state layout.

    The artifact gives each request `num_spec_tokens + 1` consecutive state slots
    from `base` (`mamba_state_indices`) and per-request read offsets into them, the
    layout the GDN kernel addresses natively; SGLang's own backend instead maps
    request-pool rows to checkpoint windows (`_state_rows`) and runs one kernel
    segment per forward. For the trace, the backend reads `(base, offsets)` as is,
    verifies `num_spec_tokens` drafts per window, and runs both kernel segments
    (windowed decodes and verifies, then per-sequence prefills), so mixed batches
    need no host-side split.
    """
    had_rows = "_state_rows" in backend.__dict__
    saved_rows = backend.__dict__.get("_state_rows")
    saved_spec = backend._num_spec_tokens
    saved_op = backend._op

    def both_segments(*args: Any, **kwargs: Any) -> Any:
        kwargs["batched_only"] = False
        kwargs["prefill_only"] = False
        return saved_op(*args, **kwargs)

    backend._state_rows = lambda: (base, offsets)
    backend._num_spec_tokens = num_spec_tokens
    backend._op = both_segments
    try:
        yield
    finally:
        backend._op = saved_op
        backend._num_spec_tokens = saved_spec
        if had_rows:
            backend._state_rows = saved_rows
        else:
            del backend._state_rows
