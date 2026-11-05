"""Tests of the SGLang adapter of the exporter (CPU): how the artifact's inputs become
the RPA backend's metadata, the KV state specs, the request buckets and the logits."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu.experimental.export import api, sglang_adapter

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

pytest.importorskip("sglang_tpu.attention_pallas_rpa_base")


def _exportable(**overrides) -> sglang_adapter.SglangExportableLLM:
    """An adapter around fakes: only the export config and the pool are real."""
    cfg = api.TextOnlyLLMExportConfig(
        model_name="tiny",
        vocab_size=10,
        hidden_size=4,
        num_layers=2,
        block_size=16,
        num_blocks=9,
        max_num_seqs=4,
        max_model_len=40,
        token_buckets=(16,),
        req_buckets=(4,),
    )
    model = sglang_adapter.SglangExportableLLM.__new__(
        sglang_adapter.SglangExportableLLM
    )
    model._export_config = dataclasses.replace(cfg, **overrides)
    kv = [torch.zeros(9, 16, 2, 2, 128, dtype=torch.bfloat16) for _ in range(2)]
    model._pool = SimpleNamespace(get_state=lambda: kv)
    model._runner = SimpleNamespace(server_args=SimpleNamespace(tp_size=1, ep_size=1))
    model._hybrid = False
    model._layout = [("kv", 0), ("kv", 1)]
    return model


def test_metadata_maps_the_artifact_inputs_onto_rpa_metadata():
    model = _exportable()
    kv_lens = torch.tensor([5, 17, 1, 0], dtype=torch.int32)
    block_tables = torch.arange(4 * 3, dtype=torch.int32)
    query_start_loc = torch.tensor([0, 1, 2, 7, 7], dtype=torch.int32)
    req_dist = torch.tensor([2, 2, 3], dtype=torch.int32)

    meta = model._metadata(
        kv_lens=kv_lens,
        block_tables=block_tables,
        query_start_loc=query_start_loc,
        req_dist=req_dist,
    )

    assert meta.kv_lens is kv_lens
    assert meta.page_indices is block_tables
    assert meta.cu_q_lens is query_start_loc
    assert meta.distribution is req_dist
    # ceil(40 / 16) = 3 pages per sequence: the rectangular page-table stride.
    assert meta.cu_kv_lens.tolist() == [0, 48, 96, 144, 192]
    # The rows of the sequences the distribution covers: query_start_loc[mixed_end].
    assert int(meta.active_num_tokens) == 7
    assert meta.decode_query_size == 1


def test_active_tokens_follow_the_distribution_not_the_bucket():
    model = _exportable()
    meta = model._metadata(
        kv_lens=torch.zeros(4, dtype=torch.int32),
        block_tables=torch.zeros(12, dtype=torch.int32),
        query_start_loc=torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32),
        req_dist=torch.tensor([1, 1, 1], dtype=torch.int32),
    )
    assert int(meta.active_num_tokens) == 1


def test_abstract_states_declare_the_fused_kv_buffers_at_num_blocks():
    model = _exportable(num_blocks=33)
    states = model.abstract_states()
    assert len(states) == 2
    assert states[0].shape == (33, 16, 2, 2, 128)
    assert states[0].dtype == "bf16"
    assert states[0].sharding == (None,) * 5  # replicated at TP 1


def test_fit_num_blocks_keeps_the_padding_page():
    model = _exportable()
    model.fit_num_blocks(64)
    assert model.export_config.num_blocks == 64
    with pytest.raises(ValueError, match="page 0"):
        model.fit_num_blocks(1)


@pytest.mark.parametrize(
    "max_num_seqs, buckets",
    [(4, (4,)), (8, (8,)), (16, (8, 16)), (48, (8, 16, 32, 48)), (64, (8, 16, 32, 64))],
)
def test_default_req_buckets(max_num_seqs, buckets):
    assert sglang_adapter.default_req_buckets(max_num_seqs) == buckets


class _Processor(nn.Module):
    """The `LogitsProcessor` surface `compute_logits` uses."""

    def __init__(self, logit_scale=None, softcap=None):
        super().__init__()
        self.logit_scale = logit_scale
        self.final_logit_softcapping = softcap

    def _compute_lm_head_rows(self, hidden_states, lm_head):
        return torch.matmul(hidden_states, lm_head.weight.t())


def _with_head(model, processor):
    head = nn.Module()
    head.weight = nn.Parameter(torch.zeros(12, 4))
    model._model = SimpleNamespace(logits_processor=processor, lm_head=head)
    return head


def test_compute_logits_binds_the_head_and_trims_the_vocab_padding():
    model = _exportable()
    head = _with_head(model, _Processor())
    hidden = torch.randn(3, 4)
    weight = torch.randn(12, 4)  # vocab 10 padded to 12

    logits = model.compute_logits(hidden, weight)

    assert logits.shape == (3, 10)
    torch.testing.assert_close(logits, (hidden @ weight.t())[:, :10])
    assert torch.count_nonzero(head.weight) == 0  # the bound weight is not kept


def test_compute_logits_applies_scale_then_softcap():
    model = _exportable()
    _with_head(model, _Processor(logit_scale=0.5, softcap=2.0))
    hidden = torch.randn(2, 4)
    weight = torch.randn(10, 4)

    logits = model.compute_logits(hidden, weight)

    expected = 2.0 * torch.tanh((hidden @ weight.t()) * 0.5 / 2.0)
    torch.testing.assert_close(logits, expected)


def test_forward_rejects_extra_controls():
    model = _exportable()
    with pytest.raises(NotImplementedError, match="extra controls"):
        model.forward(
            {},
            [],
            input_ids=torch.zeros(16, dtype=torch.int32),
            positions=torch.zeros(16, dtype=torch.int32),
            kv_lens=torch.zeros(4, dtype=torch.int32),
            query_start_loc=torch.zeros(5, dtype=torch.int32),
            req_dist=torch.zeros(3, dtype=torch.int32),
            block_tables=torch.zeros(12, dtype=torch.int32),
            mamba_slot_read_offsets=torch.zeros(5, dtype=torch.int32),
        )


def _layer(**sizes):
    layer = nn.Module()
    for k, v in sizes.items():
        setattr(layer, k, v)
    layer.weight = nn.Parameter(torch.zeros(4, 4))
    layer.weight.output_dim = 0
    layer.weight.input_dim = 1
    return layer


def test_attr_sharding_reads_the_split_dim_from_the_layer():
    model = nn.Module()
    model.col = _layer(
        input_size=8,
        input_size_per_partition=8,
        output_size=8,
        output_size_per_partition=4,
    )
    model.row = _layer(
        input_size=8,
        input_size_per_partition=4,
        output_size=8,
        output_size_per_partition=8,
    )
    model.emb = _layer(num_embeddings_per_partition=4)
    shard = sglang_adapter._attr_sharding
    assert shard(model, "col.weight", model.col.weight, 2, 2) == ["tp", None]
    assert shard(model, "row.weight", model.row.weight, 2, 2) == [None, "tp"]
    assert shard(model, "emb.weight", model.emb.weight, 2, 2) == ["tp", None]
    assert shard(model, "col.weight", model.col.weight, 2, 1) == [None, None]
    assert shard(model, "x", torch.zeros(3), 1, 2) is None  # no shard attribute


def _experts(**attrs):
    model = nn.Module()
    model.experts = nn.Module()
    for k, v in attrs.items():
        setattr(model.experts, k, v)
    return model


def test_moe_sharding_splits_experts_under_ep_and_intermediate_under_tp():
    ep = _experts(moe_ep_size=2, moe_tp_size=1)
    assert sglang_adapter._moe_sharding(ep, "experts.w13_weight", (4, 6, 8), 2) == [
        "ep",
        None,
        None,
    ]
    tp = _experts(moe_ep_size=1, moe_tp_size=2, intermediate_size_per_partition=3)
    tp.experts.w13_weight = nn.Parameter(torch.zeros(4, 6, 8))
    tp.experts.w2_weight = nn.Parameter(torch.zeros(4, 8, 3))
    assert sglang_adapter._moe_sharding(tp, "experts.w13_weight", (4, 6, 8), 1) == [
        None,
        "tp",
        None,
    ]
    assert sglang_adapter._moe_sharding(tp, "experts.w2_weight", (4, 8, 3), 1) == [
        None,
        None,
        "tp",
    ]
    # A scale is sharded like its weight.
    assert sglang_adapter._moe_sharding(
        tp, "experts.w13_weight_scale_inv", (4, 6, 1), 1
    ) == [None, "tp", None]
    assert sglang_adapter._moe_sharding(tp, "mlp.gate.weight", (4, 8), 1) is None


def _hybrid_exportable():
    """Layers 0-2 linear (GDN), layer 3 full attention, as Qwen3.5 interleaves them."""
    model = _exportable()
    kv = torch.zeros(9, 16, 2, 2, 128, dtype=torch.bfloat16)
    conv = [torch.zeros(5, 3, 1, 64) for _ in range(3)]
    rec = [torch.zeros(5, 4, 32, 32) for _ in range(3)]
    state = [kv, *conv, *rec]  # SGLang's order: full KV, conv, recurrent
    model._pool = SimpleNamespace(get_state=lambda: state)
    model._hybrid = True
    model._runner = SimpleNamespace(
        server_args=SimpleNamespace(tp_size=1, ep_size=1),
        token_to_kv_pool=SimpleNamespace(full_attention_layer_id_mapping={3: 0}),
        req_to_token_pool=SimpleNamespace(mamba_map={0: 0, 1: 1, 2: 2}),
    )
    model._layout = model._state_layout()
    return model, state


def test_hybrid_state_layout_is_per_layer_in_layer_order():
    model, _ = _hybrid_exportable()
    assert model._layout == [
        ("conv", 1),
        ("rec", 4),
        ("conv", 2),
        ("rec", 5),
        ("conv", 3),
        ("rec", 6),
        ("kv", 0),
    ]
    assert [x[0] for x in model._layout].count("kv") == 1


def test_hybrid_states_round_trip_between_orders():
    model, state = _hybrid_exportable()
    artifact_order = model._from_sglang_order(state)
    assert artifact_order[0] is state[1] and artifact_order[1] is state[4]
    assert artifact_order[-1] is state[0]
    back = model._to_sglang_order(artifact_order)
    assert all(a is b for a, b in zip(back, state))


def test_hybrid_abstract_states_size_recurrent_pools_by_state_slots():
    model, _ = _hybrid_exportable()
    specs = model.abstract_states()
    num_state_slots = model.export_config.num_state_slots  # max_num_seqs + 1
    assert specs[0].shape == (num_state_slots, 3, 1, 64)  # L0 conv
    assert specs[1].shape == (num_state_slots, 4, 32, 32)  # L0 recurrent
    assert specs[-1].shape == (9, 16, 2, 2, 128)  # L3 KV
    assert model.extra_user_inputs() == [("mamba_state_indices", (4,))]
