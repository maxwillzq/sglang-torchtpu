"""Exporter trace tests (CPU), ported from vllm-torchtpu's tests/experimental/export.

Tests that need vLLM were dropped; the SGLang adapter has its own tests.
"""

from __future__ import annotations

import re

import pytest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu.experimental.export.trace import trace

register_cpu_ci(est_time=60, suite="base-a-test-cpu")

_EVENT = re.compile(
    r"^trace-(start|end)-event: name=(\S+) ts=(\d+\.\d+) pid=(\S+) tid=(\S+)$"
)


def _events(stderr: str) -> list[tuple[str, ...]]:
    lines = stderr.strip().splitlines()
    matches = [_EVENT.match(line) for line in lines]
    assert all(matches), lines
    return [m.groups() for m in matches if m]


def test_context_manager_logs_a_start_and_end_event(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RANK", "3")
    with trace("export.lower.forward_bucket16"):
        pass
    (start, end) = _events(capsys.readouterr().err)
    assert start[:2] == ("start", "export.lower.forward_bucket16")
    assert end[:2] == ("end", "export.lower.forward_bucket16")
    assert float(start[2]) <= float(end[2])
    assert start[3] == end[3] and "/export-rank3/" in start[3]
    assert start[4] == end[4]


def test_decorator_names_the_span_after_the_function(
    capsys: pytest.CaptureFixture[str],
) -> None:
    @trace()
    def load_model() -> int:
        return 7

    assert load_model() == 7
    names = [e[1] for e in _events(capsys.readouterr().err)]
    assert names == [f"export.{load_model.__qualname__}"] * 2


def test_one_unnamed_decorator_names_each_function_it_wraps(
    capsys: pytest.CaptureFixture[str],
) -> None:
    deco = trace()

    @deco
    def first() -> None:
        pass

    @deco
    def second() -> None:
        pass

    first()
    second()
    names = [e[1] for e in _events(capsys.readouterr().err)]
    assert (
        names
        == [f"export.{first.__qualname__}"] * 2 + [f"export.{second.__qualname__}"] * 2
    )


def test_end_event_is_logged_when_the_block_raises(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(RuntimeError), trace("export.export"):
        raise RuntimeError("boom")
    assert [e[0] for e in _events(capsys.readouterr().err)] == ["start", "end"]


def test_names_with_whitespace_are_rejected() -> None:
    with pytest.raises(ValueError, match="whitespace"):
        trace("export load")


def test_saving_the_checkpoint_is_a_span(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from sglang_tpu.experimental.export import checkpoint

    saved = []
    monkeypatch.setattr(checkpoint, "abstract_params_inputs", lambda _: [])
    monkeypatch.setattr(checkpoint, "mesh_axis_sizes", lambda _: {})
    monkeypatch.setattr(checkpoint, "iter_checkpoint_shards", lambda *_: iter(()))
    monkeypatch.setattr(checkpoint, "save_checkpoint", lambda *a: saved.append(a))
    checkpoint.save_model_checkpoint({}, "artifact", "artifact/data")
    assert len(saved) == 1
    names = [e[1] for e in _events(capsys.readouterr().err)]
    assert names == ["export.save_checkpoint"] * 2
