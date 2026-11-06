"""Exporter export pipeline tests (CPU), ported from vllm-torchtpu's tests/experimental/export.

Tests that need vLLM were dropped; the SGLang adapter has its own tests.
"""

from __future__ import annotations

import dataclasses
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

import pytest
import torch
import torch.nn as nn

from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu.experimental.export import (
    api,
    checkpoint,
    exporter,
    sampling,
    tpu_lowering,
    validator,
    weight_mapper,
)

register_cpu_ci(est_time=60, suite="base-a-test-cpu")


def _mock_lower_torch_callable_to_mlir(
    fn: Any,
    arg_specs: Sequence[api.ShapeDtypeStruct],
    *,
    donate_argnums: Sequence[int] = (),
    module_name: str | None = None,
) -> str:
    """Test helper: traces `fn` on CPU tensors and emits matching StableHLO MLIR."""
    del module_name
    donated_set = {int(i) for i in donate_argnums}
    cpu_args = [
        torch.zeros(
            [int(d) for d in s.shape],
            dtype=tpu_lowering.torch_dtype(s.dtype),
            device="cpu",
        )
        for s in arg_specs
    ]
    with torch.no_grad():
        raw_out = fn(*[t.clone() for t in cpu_args])
    out_tensors = (
        [raw_out]
        if isinstance(raw_out, torch.Tensor)
        else [t for t in raw_out if isinstance(t, torch.Tensor)]
    )

    def _fmt(shape: Sequence[int], dt: Any) -> str:
        c = validator.canonical_dtype(tpu_lowering.dtype_name(dt))
        dims = "x".join(str(int(d)) for d in shape)
        return f"tensor<{dims}x{c}>" if dims else f"tensor<{c}>"

    arg_strs: list[str] = []
    for i, s in enumerate(arg_specs):
        attr = " {jax.buffer_donor = true}" if i in donated_set else ""
        arg_strs.append(f"%arg{i}: {_fmt(s.shape, s.dtype)}{attr}")
    out_strs = [_fmt(tuple(int(d) for d in t.shape), t.dtype) for t in out_tensors]
    ret_sig = f"({', '.join(out_strs)})" if len(out_strs) != 1 else out_strs[0]
    # Read every argument, as a traced graph reads every input it uses.
    body_lines: list[str] = [
        f'    %r{i} = "stablehlo.optimization_barrier"(%arg{i}) : ({t}) -> {t}'
        for i, t in enumerate(_fmt(s.shape, s.dtype) for s in arg_specs)
    ]
    ret_vals: list[str] = []
    for i, t_str in enumerate(out_strs):
        body_lines.append(f"    %c{i} = stablehlo.constant dense<0> : {t_str}")
        ret_vals.append(f"%c{i}")
    ret_stmt = (
        f"    return {', '.join(ret_vals)} : {', '.join(out_strs)}"
        if out_strs
        else "    return"
    )
    return "\n".join(
        [
            "module {",
            f"  func.func public @main({', '.join(arg_strs)}) -> {ret_sig} {{",
            *body_lines,
            ret_stmt,
            "  }",
            "}",
            "",
        ]
    )


class _TinyPyTorchBackbone(nn.Module):
    def __init__(
        self, vocab_size: int = 64, hidden_size: int = 32, num_layers: int = 2
    ) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size, bias=False) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: list[torch.Tensor] | None = None,
        attn_metadata: Any = None,
    ) -> torch.Tensor:
        del positions, kv_caches, attn_metadata
        x = self.embed_tokens(
            input_ids.clamp(min=0, max=self.embed_tokens.num_embeddings - 1)
        )
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class _TinyPyTorchCausalLM(nn.Module):
    def __init__(
        self,
        vocab_size: int = 64,
        hidden_size: int = 32,
        num_layers: int = 2,
        mrope_section: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.model = _TinyPyTorchBackbone(vocab_size, hidden_size, num_layers)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        if mrope_section is not None:
            from tests.experimental.export.test_rope_patch import (
                _SyntheticRotaryEmbedding,
            )

            self.rotary_emb = _SyntheticRotaryEmbedding(
                head_size=32,
                rotary_dim=32,
                mrope_section=mrope_section,
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: list[torch.Tensor] | None = None,
        attn_metadata: Any = None,
    ) -> torch.Tensor:
        # As a vLLM *ForCausalLM, whose forward the export calls, runs its decoder.
        return self.model(input_ids, positions, kv_caches, attn_metadata)


class _TinyTorchExportableLLM(api.ExportableTextOnlyLLM):
    """Minimal PyTorch-backed ExportableTextOnlyLLM for protocol testing."""

    def __init__(self, cfg: api.TextOnlyLLMExportConfig) -> None:
        self._cfg = cfg

    @property
    def pallas_interpret(self) -> bool:
        return False

    @property
    def export_config(self) -> api.TextOnlyLLMExportConfig:
        return self._cfg

    def abstract_params(self) -> Mapping[str, api.ShapeDtypeStruct]:
        cfg = self._cfg
        return {
            "model.embed_tokens.weight": api.ShapeDtypeStruct(
                (cfg.vocab_size, cfg.hidden_size), cfg.dtype
            ),
            "model.layers.0.self_attn.qkv_proj.weight": api.ShapeDtypeStruct(
                (cfg.hidden_size, cfg.hidden_size), cfg.dtype
            ),
            "lm_head.weight": api.ShapeDtypeStruct(
                (cfg.hidden_size, cfg.vocab_size), cfg.dtype
            ),
        }

    def forward(
        self,
        weights: Mapping[str, torch.Tensor],
        kv_caches: Sequence[torch.Tensor],
        *,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_lens: torch.Tensor,
        block_tables: torch.Tensor,
        query_start_loc: torch.Tensor,
        req_dist: torch.Tensor,
    ) -> tuple[Sequence[torch.Tensor], torch.Tensor]:
        del positions, kv_lens, block_tables, query_start_loc, req_dist
        emb = weights["model.embed_tokens.weight"]
        proj = weights["model.layers.0.self_attn.qkv_proj.weight"]
        hidden = torch.matmul(emb[input_ids.to(torch.int64)], proj)
        return list(kv_caches), hidden

    def compute_logits(
        self, hidden_states: torch.Tensor, lm_head_weight: torch.Tensor
    ) -> torch.Tensor:
        return torch.matmul(hidden_states, lm_head_weight)

    def sample_tokens(
        self,
        logits: torch.Tensor,
        *,
        temperature: torch.Tensor | None = None,
        top_k: torch.Tensor | None = None,
        top_p: torch.Tensor | None = None,
        min_p: torch.Tensor | None = None,
        uniform: torch.Tensor | None = None,
        repetition_penalty: torch.Tensor | None = None,
        presence_penalty: torch.Tensor | None = None,
        frequency_penalty: torch.Tensor | None = None,
        history: torch.Tensor | None = None,
        output_start: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return sampling.dispatch_sample_tokens(
            logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            uniform=uniform,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            history=history,
            output_start=output_start,
        )


def test_exportable_protocol_completeness() -> None:
    cfg = api.TextOnlyLLMExportConfig(
        model_name="tiny-torch",
        vocab_size=32,
        hidden_size=16,
        num_layers=1,
        num_q_heads=2,
        num_kv_heads=2,
        head_dim=8,
        block_size=8,
        num_blocks=16,
        max_num_seqs=8,
        max_model_len=64,
        token_buckets=(8, 16),
        req_buckets=(4, 8),
        emit_byte_tokenizer=True,
    )
    model = _TinyTorchExportableLLM(cfg)
    assert isinstance(model, api.ExportableModel)
    assert isinstance(model, api.ExportableTextOnlyLLM)

    logits = torch.tensor(
        [[0.1, 2.5, -1.0, 0.4], [3.0, 0.2, 0.1, -0.5]], dtype=torch.float32
    )
    greedy = model.sample_tokens(logits)
    assert greedy.shape == (2, 1)
    assert greedy.dtype == torch.int64
    assert greedy[:, 0].tolist() == [1, 0]

    temp = torch.tensor([1.0, 1.0], dtype=torch.float32)
    top_k = torch.tensor([1, 1], dtype=torch.int32)
    top_p = torch.tensor([1.0, 1.0], dtype=torch.float32)
    min_p = torch.tensor([0.0, 0.0], dtype=torch.float32)
    unif = torch.tensor([0.5, 0.5], dtype=torch.float32)
    stoch = model.sample_tokens(
        logits,
        temperature=temp,
        top_k=top_k,
        top_p=top_p,
        min_p=min_p,
        uniform=unif,
    )
    assert stoch[:, 0].tolist() == [1, 0]

    pen = model.sample_tokens(
        logits,
        temperature=temp,
        top_k=top_k,
        top_p=top_p,
        min_p=min_p,
        uniform=unif,
        repetition_penalty=torch.tensor([1.0, 1.0], dtype=torch.float32),
        presence_penalty=torch.tensor([10.0, 0.0], dtype=torch.float32),
        frequency_penalty=torch.tensor([0.0, 0.0], dtype=torch.float32),
        history=torch.tensor([[1, -1, -1, -1], [0, -1, -1, -1]], dtype=torch.int32),
        output_start=torch.tensor([0, 0], dtype=torch.int32),
    )
    assert pen.shape == (2, 1)
    assert int(pen[0, 0].item()) == 3


def test_end_to_end_export_writes_no_weights(tmp_path) -> None:
    cfg = api.TextOnlyLLMExportConfig(
        model_name="tiny-torch",
        vocab_size=32,
        hidden_size=16,
        num_layers=1,
        num_q_heads=2,
        num_kv_heads=2,
        head_dim=8,
        block_size=8,
        num_blocks=16,
        max_num_seqs=8,
        max_model_len=64,
        token_buckets=(8, 16),
        req_buckets=(4, 8),
        emit_byte_tokenizer=True,
    )
    model = _TinyTorchExportableLLM(cfg)

    out_dir = str(tmp_path / "exported_torch")
    with mock.patch.object(
        tpu_lowering,
        "lower_torch_callable_to_mlir",
        side_effect=_mock_lower_torch_callable_to_mlir,
    ):
        exporter.export(model, out_dir)
    validator.validate_exported_model_dir(out_dir)

    assert os.path.isfile(os.path.join(out_dir, "metadata.json"))
    assert not os.path.exists(os.path.join(out_dir, "config.json"))
    assert os.path.isfile(os.path.join(out_dir, "tokenizer.json"))
    assert os.path.isfile(os.path.join(out_dir, "subgraphs", "abstract_params.mlir"))
    assert os.path.isfile(os.path.join(out_dir, "subgraphs", "forward_bucket8.mlir"))
    assert os.path.isfile(os.path.join(out_dir, "subgraphs", "forward_bucket16.mlir"))
    assert os.path.isfile(
        os.path.join(out_dir, "subgraphs", "compute_logits_req4.mlir")
    )
    assert os.path.isfile(os.path.join(out_dir, "subgraphs", "sample_tokens_req4.mlir"))
    assert os.path.isfile(
        os.path.join(out_dir, "subgraphs", "sample_tokens_temperature_req4.mlir")
    )
    assert os.path.isfile(
        os.path.join(out_dir, "subgraphs", "sample_tokens_stochastic_req4.mlir")
    )
    assert os.path.isfile(
        os.path.join(out_dir, "subgraphs", "sample_tokens_penalized_req4.mlir")
    )
    # Weights ship only as the checkpoint of `--export-mode checkpoint|all`.
    assert not os.path.exists(os.path.join(out_dir, "serving_checkpoint"))


class _TinyRecurrentExportableLLM(_TinyTorchExportableLLM):
    """A tiny hybrid model: one paged attention layer and one recurrent layer."""

    def abstract_states(self) -> Sequence[api.ShapeDtypeStruct]:
        cfg = self._cfg
        return [
            api.ShapeDtypeStruct(
                (cfg.num_blocks, cfg.block_size, 2 * cfg.num_kv_heads, cfg.head_dim),
                cfg.dtype,
            ),
            api.ShapeDtypeStruct(
                (cfg.num_state_slots, 3, 1, 8),
                "f32",
                sharding=[None, None, None, "model"],
            ),
            api.ShapeDtypeStruct(
                (cfg.num_state_slots, 2, 4, 4),
                "f32",
                sharding=[None, "model", None, None],
            ),
        ]

    def extra_user_inputs(self) -> list[tuple[str, tuple[int, ...]]]:
        return [("mamba_state_indices", (self._cfg.max_num_seqs,))]

    def forward(  # type: ignore[override]
        self,
        weights: Mapping[str, torch.Tensor],
        kv_caches: Sequence[torch.Tensor],
        *,
        mamba_state_indices: torch.Tensor,
        **controls: torch.Tensor,
    ) -> tuple[Sequence[torch.Tensor], torch.Tensor]:
        kvs, hidden = super().forward(weights, kv_caches, **controls)
        kv, conv, recurrent = kvs
        rows = mamba_state_indices.to(torch.int64)
        return [kv, conv.index_fill(0, rows, 1.0), recurrent], hidden


def test_end_to_end_export_of_a_recurrent_model(tmp_path: Path) -> None:
    cfg = api.TextOnlyLLMExportConfig(
        model_name="tiny-hybrid",
        vocab_size=32,
        hidden_size=16,
        num_layers=2,
        num_q_heads=2,
        num_kv_heads=2,
        head_dim=8,
        block_size=8,
        num_blocks=16,
        max_num_seqs=8,
        max_model_len=64,
        token_buckets=(8,),
        req_buckets=(4, 8),
        emit_byte_tokenizer=True,
    )
    out_dir = str(tmp_path / "exported_hybrid")
    with mock.patch.object(
        tpu_lowering,
        "lower_torch_callable_to_mlir",
        side_effect=_mock_lower_torch_callable_to_mlir,
    ):
        exporter.export(_TinyRecurrentExportableLLM(cfg), out_dir)
    validator.validate_exported_model_dir(out_dir)

    meta = json.loads(Path(out_dir, "metadata.json").read_text())
    forward = meta["functions"]["forward"]["8"]
    states = [t for t in forward["inputs"] if t["argument_type"] == "state"]
    assert [(t["shape"], t["dtype"], t["sharding"]) for t in states] == [
        ([16, 8, 4, 8], "bf16", [None, None, None, None]),
        ([9, 3, 1, 8], "f32", [None, None, None, "model"]),
        ([9, 2, 4, 4], "f32", [None, "model", None, None]),
    ]
    # Recurrent state pools hold one slot per sequence plus null slot 0.
    assert meta["serving_config"]["num_state_slots"] == 9
    # The recurrent control input follows the state, which keeps its indices.
    last = forward["inputs"][-1]
    assert last["name"] == "mamba_state_indices"
    assert last["argument_type"] == "user_io" and last["shape"] == [8]
    assert last["arg_index"] == states[-1]["arg_index"] + 1
    outputs = forward["outputs"]
    assert [t["argument_type"] for t in outputs] == ["state"] * 3 + ["intermediate"]
    assert outputs[-1]["name"] == "hidden_states" and outputs[-1]["arg_index"] == 3
    mlir = Path(out_dir, forward["file_path"]).read_text()
    assert f"%arg{last['arg_index']}: tensor<8xi32>" in mlir


def test_lm_head_name_finds_a_nested_language_model_head() -> None:
    params = {
        "visual.blocks.0.attn.qkv.weight": None,
        "language_model.model.embed_tokens.weight": None,
        "language_model.lm_head.weight": None,
    }
    assert exporter._lm_head_name(params) == "language_model.lm_head.weight"
    del params["language_model.lm_head.weight"]
    assert exporter._lm_head_name(params) == "language_model.model.embed_tokens.weight"


@pytest.mark.parametrize(
    ("names", "alias"),
    [
        # A tied head is named after the embedding, nested or not.
        (
            ["model.embed_tokens.weight"],
            ("lm_head.weight", "model.embed_tokens.weight"),
        ),
        (
            [
                "language_model.model.embed_tokens.weight",
                "visual.patch_embed.proj.weight",
                "mtp.embed_tokens.weight",
            ],
            (
                "language_model.lm_head.weight",
                "language_model.model.embed_tokens.weight",
            ),
        ),
        # An untied head has its own weight.
        (["model.embed_tokens.weight", "lm_head.weight"], None),
        (
            [
                "language_model.model.embed_tokens.weight",
                "language_model.lm_head.weight",
            ],
            None,
        ),
    ],
)
def test_tied_lm_head_alias(names: list[str], alias: tuple[str, str] | None) -> None:
    assert weight_mapper.tied_lm_head_alias(names) == alias


def test_a_nested_tied_lm_head_keeps_the_embedding_a_forward_weight() -> None:
    # Qwen3.5-0.8B: a multimodal model whose language model ties lm_head to the
    # embedding.
    model = nn.Module()
    model.language_model = nn.Module()
    model.language_model.model = nn.Module()
    embed = nn.Embedding(32, 8)
    model.language_model.model.embed_tokens = embed
    tensors = checkpoint.local_tensors(model)
    assert tensors["language_model.lm_head.weight"] is embed.weight
    # compute_logits projects with the alias, so forward keeps the embedding.
    assert exporter._lm_head_name(tensors) == "language_model.lm_head.weight"


class _FakeRecurrentLayer:
    def get_state_shape(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return (3, 1, 64), (4, 16, 16)

    def get_state_dtype(self) -> tuple[torch.dtype, ...]:
        return torch.float32, torch.bfloat16


def test_metadata_has_one_serving_config_with_what_the_runtime_reads(tmp_path) -> None:
    cfg = api.TextOnlyLLMExportConfig(
        model_name="tiny-torch",
        vocab_size=32,
        hidden_size=16,
        num_layers=1,
        num_q_heads=2,
        num_kv_heads=2,
        head_dim=8,
        block_size=8,
        num_blocks=16,
        max_num_seqs=8,
        max_model_len=64,
        token_buckets=(8,),
        req_buckets=(4, 8),
        emit_byte_tokenizer=True,
    )
    out_dir = str(tmp_path / "exported")
    with mock.patch.object(
        tpu_lowering,
        "lower_torch_callable_to_mlir",
        side_effect=_mock_lower_torch_callable_to_mlir,
    ):
        exporter.export(_TinyTorchExportableLLM(cfg), out_dir)
    meta = json.loads(Path(out_dir, "metadata.json").read_text())
    assert "config" not in meta
    serving = meta["serving_config"]
    assert set(serving) == {
        "vocab_size",
        "eos_token_id",
        "max_num_seqs",
        "max_num_pages",
        "max_cu_seqs",
        "block_size",
        "num_blocks",
        "max_model_len",
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "data_parallel_size",
        "tokenizer_config",
        "checkpoint_config",
    }
    # The checkpoint format every export writes (Zarr v2 arrays in an OCDBT database).
    assert meta["serving_config"]["checkpoint_config"] == {"format": "zarr2_ocdbt"}
    assert (serving["vocab_size"], serving["num_blocks"]) == (32, 16)


def test_metadata_declares_the_speculative_config_it_was_exported_with(
    tmp_path,
) -> None:
    speculative = {
        "method": "ngram",
        "num_speculative_tokens": 4,
        "prompt_lookup_max": 3,
        "prompt_lookup_min": 3,
    }
    cfg = api.TextOnlyLLMExportConfig(
        model_name="tiny-torch",
        vocab_size=32,
        hidden_size=16,
        num_layers=1,
        num_q_heads=2,
        num_kv_heads=2,
        head_dim=8,
        block_size=8,
        num_blocks=16,
        max_num_seqs=8,
        max_model_len=64,
        token_buckets=(8,),
        # 8 requests verifying 4 drafts each: 40 LM-head rows.
        req_buckets=(8, 40),
        emit_byte_tokenizer=True,
        speculative_config=speculative,
    )
    out_dir = str(tmp_path / "exported")
    with mock.patch.object(
        tpu_lowering,
        "lower_torch_callable_to_mlir",
        side_effect=_mock_lower_torch_callable_to_mlir,
    ):
        exporter.export(_TinyTorchExportableLLM(cfg), out_dir)
    meta = json.loads(Path(out_dir, "metadata.json").read_text())
    assert meta["serving_config"]["speculative_config"] == speculative


def _export_metadata(tmp_path: Path) -> dict[str, Any]:
    cfg = api.TextOnlyLLMExportConfig(
        model_name="tiny-torch",
        vocab_size=32,
        hidden_size=16,
        num_layers=1,
        num_q_heads=2,
        num_kv_heads=2,
        head_dim=8,
        block_size=8,
        num_blocks=16,
        max_num_seqs=8,
        max_model_len=64,
        token_buckets=(8,),
        req_buckets=(4, 8),
        emit_byte_tokenizer=True,
    )
    out_dir = str(tmp_path / "exported")
    with mock.patch.object(
        tpu_lowering,
        "lower_torch_callable_to_mlir",
        side_effect=_mock_lower_torch_callable_to_mlir,
    ):
        exporter.export(_TinyTorchExportableLLM(cfg), out_dir)
    # Strict JSON: Python would otherwise accept NaN and Infinity.
    return json.loads(
        Path(out_dir, "metadata.json").read_text(),
        parse_constant=lambda name: pytest.fail(f"{name} is not JSON"),
    )


def test_metadata_declares_the_xla_flags_in_libtpu_init_args(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(
        "LIBTPU_INIT_ARGS",
        "--xla_tpu_use_dynamic_smem_negotiation=true"
        " --xla_tpu_sparse_core_all_reduce_offload_min_size_in_bytes=67108864"
        " --xla_tpu_scoped_vmem_limit_kib=0.5 --xla_tpu_mode=fast"
        " --xla_tpu_enable_foo --tpu_megacore=false --xla_jf_bar=false"
        " --xla_tpu_limit=inf",
    )
    overrides = _export_metadata(tmp_path)["compile_options"]["env_option_overrides"]
    # Typed JSON values; a bare flag is true; non-XLA libtpu flags are dropped.
    assert overrides == {
        "xla_tpu_use_dynamic_smem_negotiation": True,
        "xla_tpu_sparse_core_all_reduce_offload_min_size_in_bytes": 67108864,
        "xla_tpu_scoped_vmem_limit_kib": 0.5,
        "xla_tpu_mode": "fast",
        "xla_tpu_enable_foo": True,
        "xla_jf_bar": False,
        # JSON has no infinity or NaN; the runtime would fail to parse them.
        "xla_tpu_limit": "inf",
    }


def test_metadata_has_no_xla_flags_without_libtpu_init_args(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("LIBTPU_INIT_ARGS", raising=False)
    assert "env_option_overrides" not in _export_metadata(tmp_path)["compile_options"]


def test_compute_logits_output_is_the_samplers_intermediate_input(tmp_path) -> None:
    cfg = api.TextOnlyLLMExportConfig(
        model_name="tiny-torch",
        vocab_size=32,
        hidden_size=16,
        num_layers=1,
        num_q_heads=2,
        num_kv_heads=2,
        head_dim=8,
        block_size=8,
        num_blocks=16,
        max_num_seqs=8,
        max_model_len=64,
        token_buckets=(8,),
        req_buckets=(4, 8),
        emit_byte_tokenizer=True,
    )
    out_dir = str(tmp_path / "exported")
    with mock.patch.object(
        tpu_lowering,
        "lower_torch_callable_to_mlir",
        side_effect=_mock_lower_torch_callable_to_mlir,
    ):
        exporter.export(_TinyTorchExportableLLM(cfg), out_dir)
    functions = json.loads(Path(out_dir, "metadata.json").read_text())["functions"]
    samplers = [
        "sample_tokens",
        "sample_tokens_temperature",
        "sample_tokens_stochastic",
        "sample_tokens_penalized",
    ]
    for req, spec in functions["compute_logits"].items():
        (logits,) = spec["outputs"]
        assert (logits["name"], logits["argument_type"]) == ("logits", "intermediate")
        for fn in samplers:
            inputs = {t["name"]: t for t in functions[fn][req]["inputs"]}
            assert inputs["logits"]["argument_type"] == "intermediate", (fn, req)
            assert inputs["logits"]["shape"] == logits["shape"], (fn, req)


def test_companion_files_are_copied_and_declared_in_serving_config(tmp_path) -> None:
    hf_dir = tmp_path / "hf_ckpt"
    hf_dir.mkdir()
    (hf_dir / "config.json").write_text(json.dumps({"eos_token_id": 2}))
    (hf_dir / "tokenizer.json").write_text(json.dumps({"version": "1.0"}))
    (hf_dir / "generation_config.json").write_text(
        json.dumps({"eos_token_id": [2, 0], "do_sample": False})
    )
    out_dir = tmp_path / "artifact"
    out_dir.mkdir()
    cfg = api.TextOnlyLLMExportConfig(
        model_name="tiny", vocab_size=64, hf_checkpoint_dir=str(hf_dir)
    )

    serving_config = exporter._emit_model_companion_files(cfg, str(out_dir))

    assert serving_config["tokenizer_config"] == {
        "source": {"TokenizerPath": {"path": "tokenizer.json"}}
    }
    assert serving_config["generation_config"] == {"path": "generation_config.json"}
    for name in ("tokenizer.json", "generation_config.json"):
        assert (out_dir / name).read_text() == (hf_dir / name).read_text()
    assert not (out_dir / "config.json").exists()


def test_generation_config_is_not_declared_when_the_checkpoint_has_none(
    tmp_path,
) -> None:
    hf_dir = tmp_path / "hf_ckpt"
    hf_dir.mkdir()
    (hf_dir / "config.json").write_text(json.dumps({"eos_token_id": 2}))
    (hf_dir / "tokenizer.json").write_text(json.dumps({"version": "1.0"}))
    out_dir = tmp_path / "artifact"
    out_dir.mkdir()
    cfg = api.TextOnlyLLMExportConfig(
        model_name="tiny", vocab_size=64, hf_checkpoint_dir=str(hf_dir)
    )

    serving_config = exporter._emit_model_companion_files(cfg, str(out_dir))

    assert "generation_config" not in serving_config
    assert not (out_dir / "generation_config.json").exists()


@pytest.mark.parametrize(
    "files,expected",
    [
        (
            {
                "chat_template.jinja": "jinja file",
                "tokenizer_config.json": json.dumps({"chat_template": "config"}),
            },
            "jinja file",
        ),
        ({"tokenizer_config.json": json.dumps({"chat_template": "config"})}, "config"),
        (
            {
                "tokenizer_config.json": json.dumps(
                    {
                        "chat_template": [
                            {"name": "tool_use", "template": "tools"},
                            {"name": "default", "template": "named default"},
                        ]
                    }
                )
            },
            "named default",
        ),
        ({"tokenizer_config.json": json.dumps({"bos_token": "<s>"})}, None),
        ({}, None),
    ],
)
def test_chat_template_is_written_and_declared_like_huggingface_finds_it(
    tmp_path, files: dict[str, str], expected: str | None
) -> None:
    hf_dir = tmp_path / "hf_ckpt"
    hf_dir.mkdir()
    (hf_dir / "tokenizer.json").write_text(json.dumps({"version": "1.0"}))
    for name, text in files.items():
        (hf_dir / name).write_text(text)
    out_dir = tmp_path / "artifact"
    out_dir.mkdir()
    cfg = api.TextOnlyLLMExportConfig(
        model_name="tiny", vocab_size=64, hf_checkpoint_dir=str(hf_dir)
    )

    serving_config = exporter._emit_model_companion_files(cfg, str(out_dir))

    if expected is None:
        assert "chat_template" not in serving_config
        assert not (out_dir / "chat_template.jinja").exists()
    else:
        assert serving_config["chat_template"] == {"path": "chat_template.jinja"}
        assert (out_dir / "chat_template.jinja").read_text() == expected


def _parallel_vllm_cfg(tmp_path: Path, tp: int, dp: int, pp: int) -> Any:
    return SimpleNamespace(
        model_config=SimpleNamespace(
            model=str(tmp_path), dtype="bfloat16", max_model_len=128
        ),
        cache_config=SimpleNamespace(block_size=16),
        scheduler_config=SimpleNamespace(max_num_seqs=8),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tp,
            data_parallel_size=dp,
            pipeline_parallel_size=pp,
        ),
    )


def test_main_args_read_finds_the_arguments_main_reads() -> None:
    args = ", ".join(f"%arg{i}: tensor<i64>" for i in range(11))
    mlir = "\n".join(
        [
            "module {",
            f"  func.func public @main({args}) -> tensor<i64> {{",
            "    %0 = stablehlo.add %arg2, %arg10 : tensor<i64>",
            "    return %0 : tensor<i64>",
            "  }",
            "  func.func private @helper(%arg1: tensor<i64>) -> tensor<i64> {",
            "    return %arg1 : tensor<i64>",
            "  }",
            "}",
        ]
    )
    # %arg10 does not count as %arg1, and @helper's own %arg1 is not @main's.
    assert exporter._main_args_read(mlir) == {2, 10}


def test_params_read_by_forward_drops_unread_forward_weights() -> None:
    cfg = api.TextOnlyLLMExportConfig(
        model_name="tiny", vocab_size=64, token_buckets=(8, 16), num_layers=1
    )
    params = {
        "model.embed_tokens.weight": api.ShapeDtypeStruct((64, 8), "bf16"),
        "model.embed_tokens.org_vocab_start_index": api.ShapeDtypeStruct((), "i64"),
        "model.norm.weight": api.ShapeDtypeStruct((8,), "bf16"),
        "lm_head.weight": api.ShapeDtypeStruct((64, 8), "bf16"),
    }
    model = SimpleNamespace(
        export_config=cfg,
        abstract_states=lambda: [api.ShapeDtypeStruct((4, 8, 2, 2, 4), "bf16")],
    )
    num_user = len(exporter._user_inputs(cfg, 8))

    def fake_lower(_model: Any, *, forward_names: Sequence[str], **_: Any) -> str:
        assert "lm_head.weight" not in forward_names
        read = [
            num_user + forward_names.index(n)
            for n in ("model.embed_tokens.weight", "model.norm.weight")
        ]
        args = ", ".join(f"%arg{i}: tensor<i32>" for i in range(num_user + 4))
        uses = "\n".join(
            f"    %u{i} = stablehlo.abs %arg{i} : tensor<i32>" for i in read
        )
        header = f"  func.func public @main({args}) -> () {{"
        return "\n".join(["module {", header, uses, "  }", "}", ""])

    with mock.patch.object(tpu_lowering, "lower_torch_forward_to_mlir", fake_lower):
        kept = exporter._params_read_by_forward(model, params)
    assert list(kept) == [
        "model.embed_tokens.weight",
        "model.norm.weight",
        "lm_head.weight",
    ]


def test_companion_files_require_a_tokenizer(tmp_path: Path) -> None:
    hf_dir = tmp_path / "hf_ckpt"
    hf_dir.mkdir()
    (hf_dir / "config.json").write_text(json.dumps({"eos_token_id": 2}))
    out_dir = tmp_path / "artifact"
    out_dir.mkdir()
    cfg = api.TextOnlyLLMExportConfig(
        model_name="tiny", vocab_size=64, hf_checkpoint_dir=str(hf_dir)
    )
    with pytest.raises(FileNotFoundError, match="tokenizer.json"):
        exporter._emit_model_companion_files(cfg, str(out_dir))


def _tp2_model(constant_values: Mapping[str, list[Any]]) -> Any:
    cfg = api.TextOnlyLLMExportConfig(
        model_name="tiny", vocab_size=64, num_partitions=2
    )
    return SimpleNamespace(
        export_config=cfg,
        model=None,
        constant_axis_values=lambda name: constant_values[name],
    )


def test_tensor_layouts_declare_global_shapes_of_weights_and_constants() -> None:
    values = {
        "model.embed_tokens.org_vocab_start_index": [0, 32],
        "model.embed_tokens.added_vocab_start_index": [64, 64],
    }
    model = _tp2_model(values)
    # Rank-local shapes of a tensor-parallel size 2 export.
    params = {
        "model.embed_tokens.weight": api.ShapeDtypeStruct(
            (32, 8), "bf16", sharding=["tp", None]
        ),
        "model.embed_tokens.org_vocab_start_index": api.ShapeDtypeStruct((), "i64"),
        "model.embed_tokens.added_vocab_start_index": api.ShapeDtypeStruct((), "i64"),
    }
    layouts = exporter._tensor_layouts(
        model, params, frozenset(values), exporter._axis_sizes(model.export_config)
    )

    assert layouts["model.embed_tokens.weight"] == exporter._TensorLayout(
        shape=[64, 8], sharding=["model", None], lowered_shape=(32, 8)
    )
    # A per-device value is a sharded global array; each device holds `[1]`.
    assert layouts["model.embed_tokens.org_vocab_start_index"] == (
        exporter._TensorLayout(
            shape=[2], sharding=["model"], lowered_shape=(1,), reshape=()
        )
    )
    # A value shared by every device stays a replicated scalar.
    assert layouts["model.embed_tokens.added_vocab_start_index"] == (
        exporter._TensorLayout(shape=[], sharding=[], lowered_shape=())
    )


def test_set_aliasing_outputs_pairs_each_donor_with_its_result() -> None:
    mlir = (
        "module {\n"
        "  func.func public @main(%arg0: tensor<4xi32>, "
        "%arg1: tensor<2x4xbf16> {jax.buffer_donor = true}, "
        "%arg2: tensor<2x4xbf16> {jax.buffer_donor = true}) -> "
        "(tensor<2x4xbf16>, tensor<2x4xbf16>, tensor<4xbf16>) {\n"
        "  }\n}\n"
    )
    out = tpu_lowering.set_aliasing_outputs(mlir, {1: 0, 2: 1})
    assert "jax.buffer_donor" not in out
    assert "%arg1: tensor<2x4xbf16> {tf.aliasing_output = 0 : i32}" in out
    assert "%arg2: tensor<2x4xbf16> {tf.aliasing_output = 1 : i32}" in out
    with pytest.raises(ValueError, match="no aliased result"):
        tpu_lowering.set_aliasing_outputs(mlir, {1: 0})


def test_validator_rejects_a_donor_without_aliasing_output(tmp_path: Path) -> None:
    mlir = (
        "module {\n"
        "  func.func public @main(%arg0: tensor<2xbf16> {jax.buffer_donor = true}) "
        "-> tensor<2xbf16> {\n"
        "    return %arg0 : tensor<2xbf16>\n"
        "  }\n}\n"
    )
    (tmp_path / "abstract_params.mlir").write_text(mlir)
    entry = {
        "arg_index": 0,
        "index": 0,
        "name": "w",
        "shape": [2],
        "dtype": "bf16",
        "sharding": [None],
        "argument_type": "weight",
        "buffer_donor": True,
    }
    meta = {
        "format_version": 2,
        "mlir_shapes": "shard_local",
        "mesh": {"mesh_shape": [1, 1], "axis_names": ["data", "model"]},
        "functions": {
            "abstract_params": {
                "default": {
                    "file_path": "abstract_params.mlir",
                    "inputs": [entry],
                    "outputs": [dict(entry, buffer_donor=False)],
                }
            }
        },
    }
    (tmp_path / "metadata.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="has no tf.aliasing_output"):
        validator.validate_exported_model_dir(str(tmp_path))
    (tmp_path / "abstract_params.mlir").write_text(
        mlir.replace("jax.buffer_donor = true", "tf.aliasing_output = 0 : i32")
    )
    validator.validate_exported_model_dir(str(tmp_path))
    meta["functions"]["abstract_params"]["default"]["inputs"][0]["dtype"] = "bfloat16"
    (tmp_path / "metadata.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="not a StableHLO element type"):
        validator.validate_exported_model_dir(str(tmp_path))


class _SoftCappedLM(nn.Module):
    """A vLLM-style LM whose `compute_logits` soft-caps after the projection."""

    def __init__(self, tied: bool) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(10, 4)
        if not tied:
            self.lm_head = nn.Linear(4, 10, bias=False)

    def head(self) -> torch.Tensor:
        return (getattr(self, "lm_head", None) or self.model.embed_tokens).weight

    def compute_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return 2.0 * torch.tanh(torch.matmul(hidden, self.head().T) / 2.0)


class _LlamaLikeBackbone(nn.Module):
    """Backbone whose `forward` takes `intermediate_tensors` with no default."""

    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(8, 4)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Any,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert intermediate_tensors is None
        return self.embed_tokens(input_ids)


class _KvCacheArgBackbone(_LlamaLikeBackbone):
    """Backbone that takes the KV caches and fails inside its forward."""

    def forward(  # type: ignore[override]
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: Sequence[torch.Tensor] | None = None,
        attn_metadata: Any = None,
    ) -> torch.Tensor:
        if kv_caches is not None:
            raise TypeError("bug inside the model")
        return self.embed_tokens(input_ids)


def _pallas_impl(kv_dtype: torch.dtype | None) -> Any:
    from vllm_torchtpu.layers.adapter.attention import PallasAttentionBackendImpl

    impl = object.__new__(PallasAttentionBackendImpl)
    impl.num_kv_heads, impl.head_size = 2, 128
    impl.kv_cache_quantized_dtype = kv_dtype
    return impl


def _bare_layer(cls_path: str) -> Any:
    module_name, cls_name = cls_path.rsplit(".", 1)
    module = __import__(module_name, fromlist=[cls_name])
    return object.__new__(getattr(module, cls_name))


class _GroupRecordingBackbone(nn.Module):
    """Backbone that writes each layer's KV in place, as the attention kernel
    does, and records the page table each layer reads."""

    def __init__(self, layers: Sequence[tuple[str, Any]]) -> None:
        super().__init__()
        self.layers = layers
        self.tables: dict[str, torch.Tensor] = {}

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        attn_metadata: Any = None,
    ) -> torch.Tensor:
        for i, (name, layer) in enumerate(self.layers):
            self.tables[name] = attn_metadata[name].block_tables
            layer.kv_cache.copy_(layer.kv_cache + 10**i)
        return input_ids.float()


class _FakeWorker:
    """Stands in for a worker process: writes a file named after its rank."""

    fail_ranks: tuple[int, ...] = ()

    def __init__(self, cmd: Sequence[str], env: Mapping[str, str]) -> None:
        rank = int(env["RANK"])
        assert cmd[-1] == "--export-worker"
        out_dir = Path(cmd[cmd.index("--output-dir", len(cmd) - 4) + 1])
        out_dir.mkdir(parents=True)
        (out_dir / "metadata.json").write_text(json.dumps({"rank": rank}))
        self._code = 1 if rank in self.fail_ranks else 0

    def wait(self) -> int:
        return self._code

    def poll(self) -> int:
        return self._code


class _Router:
    """A plain object holding a module's parameter, as vLLM MoE routers do."""

    def __init__(self, bias: torch.Tensor) -> None:
        self.bias = bias


class _BackboneWithRouter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate = nn.Module()
        self.gate.bias = nn.Parameter(torch.zeros(4), requires_grad=False)
        self.experts = nn.Module()
        # Mirrors vLLM's MoERunner/RoutedExperts layout: the same `gate`
        # submodule is reachable at two module paths (`gate` and
        # `experts.gate`), `routed_experts` holds the shared Parameter directly,
        # and `router` is a plain object holding the Parameter outside the
        # module tree.
        self.experts.gate = self.gate
        self.experts.routed_experts = nn.Module()
        self.experts.routed_experts.bias = self.gate.bias
        self.experts.router = _Router(self.gate.bias)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> Any:
        del positions
        assert self.gate.bias is self.experts.routed_experts.bias
        assert self.gate.bias is self.experts.router.bias
        return input_ids.float()[:, None] + self.experts.router.bias[None, :]


class _CausalLMOf(nn.Module):
    """A vLLM-style *ForCausalLM around `model`: its forward runs the decoder."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> Any:
        return self.model(input_ids, positions)


def test_uniform_per_logit_is_in_range_and_depends_only_on_the_row() -> None:
    uniform = torch.tensor([0.1, 0.5, 0.9, 0.5])
    u = sampling.uniform_per_logit(uniform, 1000)
    assert u.shape == (4, 1000) and u.dtype == torch.float32
    assert bool((u > 0).all()) and bool((u < 1).all())
    assert torch.equal(u[1], u[3])  # same seed, same values
    assert not torch.equal(u[0], u[1])
    alone = sampling.uniform_per_logit(uniform[2:3], 1000)
    assert torch.equal(alone[0], u[2])  # batch-invariant
    assert abs(float(u.mean()) - 0.5) < 0.02


def _sample_frequencies(sample, logits_row, rows=20000, **params):
    torch.manual_seed(0)
    logits = logits_row.expand(rows, -1).contiguous()
    uniform = torch.rand(rows)
    tokens = sample(logits, uniform=uniform, **params).squeeze(-1)
    return torch.bincount(tokens, minlength=logits.shape[-1]).float() / rows


def test_temperature_and_stochastic_samplers_follow_the_softmax() -> None:
    logits_row = torch.tensor([[2.0, 1.0, 0.5, 0.0, -1.0, -3.0]])
    rows = 20000
    temp = torch.full((rows,), 0.7)
    expected = torch.softmax(logits_row[0] / 0.7, dim=-1)
    freq = _sample_frequencies(
        lambda logits, uniform, **p: sampling.temperature_tokens(logits, temp, uniform),
        logits_row,
        rows,
    )
    assert torch.allclose(freq, expected, atol=0.015), (freq, expected)
    no_k = torch.zeros(rows, dtype=torch.int32)
    no_p = torch.ones(rows)
    freq = _sample_frequencies(
        lambda logits, uniform, **p: sampling.stochastic_tokens(
            logits, temp, no_k, no_p, torch.zeros(rows), uniform
        ),
        logits_row,
        rows,
    )
    assert torch.allclose(freq, expected, atol=0.015), (freq, expected)


def test_stochastic_sampler_keeps_only_the_top_k_and_top_p_tokens() -> None:
    logits_row = torch.tensor([[3.0, 2.5, 2.0, 0.0, -1.0, -2.0]])
    rows = 4000
    temp = torch.ones(rows)
    top_k = torch.full((rows,), 2, dtype=torch.int32)
    freq = _sample_frequencies(
        lambda logits, uniform, **p: sampling.stochastic_tokens(
            logits, temp, top_k, torch.ones(rows), torch.zeros(rows), uniform
        ),
        logits_row,
        rows,
    )
    assert float(freq[2:].sum()) == 0.0 and float(freq[:2].min()) > 0.2
    top_p = torch.full((rows,), 0.5)
    freq = _sample_frequencies(
        lambda logits, uniform, **p: sampling.stochastic_tokens(
            logits,
            temp,
            torch.zeros(rows, dtype=torch.int32),
            top_p,
            torch.zeros(rows),
            uniform,
        ),
        logits_row,
        rows,
    )
    # softmax([3, 2.5, 2, ...]) = [0.47, 0.29, ...]: the top-p set is tokens 0 and 1.
    assert float(freq[2:].sum()) == 0.0 and float(freq[1]) > 0.2


def test_zero_temperature_rows_take_the_argmax() -> None:
    logits = torch.tensor([[0.0, 5.0, 1.0], [4.0, 0.0, 1.0]])
    temp = torch.tensor([0.0, 0.0])
    uniform = torch.tensor([0.3, 0.8])
    assert sampling.temperature_tokens(logits, temp, uniform).squeeze(-1).tolist() == [
        1,
        0,
    ]
    tokens = sampling.stochastic_tokens(
        logits,
        temp,
        torch.tensor([1, 1], dtype=torch.int32),
        torch.ones(2),
        torch.zeros(2),
        uniform,
    )
    assert tokens.squeeze(-1).tolist() == [1, 0]


class _TinyMtpExportableLLM(_TinyTorchExportableLLM):
    """The tiny model with an MTP layer: its own `mtp.fc` among its parameters,
    the target's embedding and LM head, and one paged KV state."""

    def abstract_params(self) -> Mapping[str, api.ShapeDtypeStruct]:
        cfg = self._cfg
        return {
            **super().abstract_params(),
            "mtp.fc.weight": api.ShapeDtypeStruct(
                (2 * cfg.hidden_size, cfg.hidden_size), cfg.dtype
            ),
        }

    def mtp_abstract_states(self) -> Sequence[api.ShapeDtypeStruct]:
        cfg = self._cfg
        return [
            api.ShapeDtypeStruct(
                (cfg.num_blocks, cfg.block_size, 2 * cfg.num_kv_heads, cfg.head_dim),
                cfg.dtype,
            )
        ]

    def mtp_draft(
        self,
        weights: Mapping[str, torch.Tensor],
        kv_caches: Sequence[torch.Tensor],
        *,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_lens: torch.Tensor,
        block_tables: torch.Tensor,
        query_start_loc: torch.Tensor,
        req_dist: torch.Tensor,
        last_token_indices: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> tuple[Sequence[torch.Tensor], torch.Tensor]:
        del positions, kv_lens, block_tables, query_start_loc, req_dist
        emb = weights["model.embed_tokens.weight"][input_ids.to(torch.int64)]
        fused = torch.cat([emb, hidden_states.to(emb.dtype)], dim=-1)
        rows = torch.matmul(fused, weights["mtp.fc.weight"])[
            last_token_indices.to(torch.int64)
        ]
        logits = torch.matmul(rows, weights["lm_head.weight"])
        return list(kv_caches), logits.argmax(dim=-1).to(torch.int32).view(-1, 1)


class _ReadTracker(torch.Tensor):
    """A lowering argument that records, in `reads`, whether any op used it."""

    reads: set[int]
    arg_index: int

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        flat, _ = torch.utils._pytree.tree_flatten((args, kwargs))
        for a in flat:
            if isinstance(a, _ReadTracker):
                a.reads.add(a.arg_index)
        with torch._C.DisableTorchFunctionSubclass():
            return func(*args, **kwargs)


def _mock_lower_reading_used_args(
    fn: Any,
    arg_specs: Sequence[api.ShapeDtypeStruct],
    *,
    donate_argnums: Sequence[int] = (),
    module_name: str | None = None,
) -> str:
    """Like `_mock_lower_torch_callable_to_mlir`, but `@main` reads only the
    arguments that `fn` used, as the traced graph of a real lowering does."""
    reads: set[int] = set()

    def tracked(*args: torch.Tensor) -> Any:
        wrapped = []
        for i, a in enumerate(args):
            t = a.as_subclass(_ReadTracker)
            t.reads, t.arg_index = reads, i
            wrapped.append(t)
        return fn(*wrapped)

    mlir = _mock_lower_torch_callable_to_mlir(
        tracked, arg_specs, donate_argnums=donate_argnums, module_name=module_name
    )
    return "\n".join(
        line
        for line in mlir.split("\n")
        if not (
            "optimization_barrier" in line
            and int(line.split("%arg")[1].split(")")[0]) not in reads
        )
    )


def _mtp_config() -> api.TextOnlyLLMExportConfig:
    return api.TextOnlyLLMExportConfig(
        model_name="tiny-torch",
        vocab_size=32,
        hidden_size=16,
        num_layers=1,
        num_q_heads=2,
        num_kv_heads=2,
        head_dim=8,
        block_size=8,
        num_blocks=16,
        max_num_seqs=8,
        max_model_len=64,
        token_buckets=(8, 16),
        req_buckets=(4, 8, 16),
        emit_byte_tokenizer=True,
        speculative_config={"method": "mtp", "num_speculative_tokens": 1},
    )


def _export_mtp(tmp_path: Path, model_cls: type = _TinyMtpExportableLLM) -> str:
    cfg = _mtp_config()
    out_dir = str(tmp_path / "exported")
    with mock.patch.object(
        tpu_lowering,
        "lower_torch_callable_to_mlir",
        side_effect=_mock_lower_reading_used_args,
    ):
        exporter.export(model_cls(cfg), out_dir)
    return out_dir


def test_mtp_draft_follows_the_contract(tmp_path) -> None:
    out_dir = _export_mtp(tmp_path)
    meta = json.loads(Path(out_dir, "metadata.json").read_text())
    functions = meta["functions"]
    assert set(functions["draft"]) == set(functions["forward"]) == {"8", "16"}
    assert meta["serving_config"]["speculative_config"]["method"] == "mtp"

    variant = functions["draft"]["16"]
    assert variant["file_path"].endswith("draft_bucket16.mlir"), variant["file_path"]
    by_name = {i["name"]: i for i in variant["inputs"]}
    assert Path(out_dir, variant["file_path"]).is_file()
    # The forward's controls, shifted input_ids, then the last accepted rows.
    user = [i["name"] for i in variant["inputs"] if i["argument_type"] == "user_io"]
    assert user == [
        "input_ids",
        "positions",
        "kv_lens",
        "block_tables",
        "query_start_loc",
        "req_dist",
        "last_token_indices",
    ]
    assert by_name["last_token_indices"]["shape"] == [8]
    assert by_name["hidden_states"]["argument_type"] == "intermediate"
    assert by_name["hidden_states"]["shape"] == [16, 16]

    # Shared weights bind the target's slots; the MTP layer's own get new ones.
    forward_slots = {
        i["name"]: i["index"]
        for i in functions["forward"]["16"]["inputs"]
        if i["argument_type"] == "weight"
    }
    logits_slots = {
        i["name"]: i["index"]
        for i in functions["compute_logits"]["4"]["inputs"]
        if i["argument_type"] == "weight"
    }
    assert (
        by_name["model.embed_tokens.weight"]["index"]
        == forward_slots["model.embed_tokens.weight"]
    )
    assert by_name["lm_head.weight"]["index"] == logits_slots["lm_head.weight"]
    assert "mtp.fc.weight" not in forward_slots
    # mtp_draft takes the weights it reads: the shared ones, then its own.
    weights = [i["name"] for i in variant["inputs"] if i["argument_type"] == "weight"]
    assert weights == ["model.embed_tokens.weight", "lm_head.weight", "mtp.fc.weight"]
    assert by_name["mtp.fc.weight"]["index"] not in forward_slots.values()

    # The MTP KV is a donated state numbered after the target's.
    target_states = [
        i for i in functions["forward"]["16"]["inputs"] if i["argument_type"] == "state"
    ]
    mtp_state = by_name["mtp_kv_cache_layer_0"]
    assert mtp_state["buffer_donor"] is True
    assert mtp_state["index"] == len(target_states)
    draft = [o for o in variant["outputs"] if o["name"] == "draft_tokens"]
    assert draft[0]["shape"] == [8, 1] and draft[0]["dtype"] == "i32"


def test_a_speculative_export_needs_request_buckets_for_every_verify_row(
    tmp_path,
) -> None:
    # 8 requests verifying 1 draft each run the LM head on 16 rows.
    model = _TinyMtpExportableLLM(
        dataclasses.replace(
            _TinyMtpExportableLLM(_mtp_config()).export_config, req_buckets=(4, 8)
        )
    )
    with pytest.raises(ValueError, match="16 LM-head rows"):
        exporter.export(model, str(tmp_path / "out"))


def test_a_speculative_export_needs_no_request_bucket_above_the_largest_step(
    tmp_path,
) -> None:
    # 8 requests verifying 1 draft each would be 16 rows, but a step takes at most
    # 8 tokens, so request buckets up to 8 suffice.
    cfg = dataclasses.replace(_mtp_config(), token_buckets=(8,), req_buckets=(4, 8))
    with mock.patch.object(
        tpu_lowering,
        "lower_torch_callable_to_mlir",
        side_effect=_mock_lower_reading_used_args,
    ):
        exporter.export(_TinyMtpExportableLLM(cfg), str(tmp_path / "out"))


def test_mtp_needs_a_model_that_drafts(tmp_path) -> None:
    with pytest.raises(ValueError, match="mtp_abstract_states"):
        _export_mtp(tmp_path, model_cls=_TinyTorchExportableLLM)


class _TinyMtpWithoutWeights(_TinyMtpExportableLLM):
    """Drafts, but its parameters have no `mtp.` layer."""

    def abstract_params(self) -> Mapping[str, api.ShapeDtypeStruct]:
        return _TinyTorchExportableLLM.abstract_params(self)


def test_mtp_needs_the_mtp_layer_among_the_parameters(tmp_path) -> None:
    with pytest.raises(ValueError, match="no 'mtp.' parameters"):
        _export_mtp(tmp_path, model_cls=_TinyMtpWithoutWeights)


@pytest.mark.parametrize("method", validator.DRAFTER_METHODS)
def test_validator_rejects_a_draft_missing_a_bucket(tmp_path, method) -> None:
    # Every drafter method holds the artifact to the same `draft` contract.
    out_dir = _export_mtp(tmp_path)
    meta_path = Path(out_dir, "metadata.json")
    meta = json.loads(meta_path.read_text())
    meta["serving_config"]["speculative_config"]["method"] = method
    meta_path.write_text(json.dumps(meta))
    validator.validate_exported_model_dir(out_dir)
    del meta["functions"]["draft"]["8"]
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="differ from forward buckets"):
        validator.validate_exported_model_dir(out_dir)
    del meta["functions"]["draft"]
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="exports no draft"):
        validator.validate_exported_model_dir(out_dir)
