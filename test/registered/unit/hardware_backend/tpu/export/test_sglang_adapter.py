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
    assert states[0].sharding == (None,) * 5


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
            mamba_state_indices=torch.zeros(4, dtype=torch.int32),
        )
