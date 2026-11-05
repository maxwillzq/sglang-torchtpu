"""Command line of `python -m sglang_tpu.experimental.export`.

Takes SGLang's server arguments (`python -m sglang.launch_server`'s, model
parameters 1:1) plus the export options below, loads the model on this TPU and
writes the StableHLO serving artifact to `--output-dir`:

    python -m sglang_tpu.experimental.export --model-path Qwen/Qwen3-0.6B \\
        --output-dir /tmp/exported --context-length 2048 \\
        --max-running-requests 64 --page-size 64 --token-buckets 16,32,64

The artifact's static sizes come from the server arguments: `max_num_seqs` is
`--max-running-requests`, `max_model_len` is `--context-length`, `block_size` is
`--page-size`, and `num_blocks` is the size of the KV pool SGLang allocates (set
it with `--max-total-tokens`, or override it with `--num-blocks`).

With `--tp-size N > 1` the command starts one worker process per rank (as SGLang
serving starts its schedulers): each loads its shard, traces its shard-local
programs and, in `all` mode, saves its weight shards into the shared Orbax
checkpoint; rank 0's artifact is kept.
"""

from __future__ import annotations

import argparse
import os
import shutil
import socket
import sys
import tempfile
from collections.abc import Sequence
from typing import Any

from sglang_tpu.experimental.export import api, checkpoint, exporter, sglang_adapter

EXPORT_MODES = ("model", "all")


def _int_list(text: str | None) -> tuple[int, ...] | None:
    if not text:
        return None
    values = tuple(int(v) for v in text.split(",") if v.strip())
    if not values or any(v <= 0 for v in values):
        raise argparse.ArgumentTypeError(f"expected positive integers, got {text!r}")
    return values


def build_parser() -> argparse.ArgumentParser:
    from sglang.srt.server_args import ServerArgs

    parser = argparse.ArgumentParser(
        prog="python -m sglang_tpu.experimental.export",
        description="Export an SGLang model to a StableHLO serving artifact.",
    )
    ServerArgs.add_cli_args(parser)
    group = parser.add_argument_group("Export Options")
    group.add_argument(
        "--output-dir", required=True, help="Artifact directory to write."
    )
    group.add_argument(
        "--export-mode",
        choices=EXPORT_MODES,
        default="model",
        help="`model`: graphs only, from dummy weights. `all`: graphs and the real "
        "weights in serving_checkpoint/, from one load.",
    )
    group.add_argument(
        "--token-buckets",
        default="16,32,64,128,256",
        help="Comma-separated token buckets; one forward graph each.",
    )
    group.add_argument(
        "--req-buckets",
        default=None,
        help="Comma-separated request buckets for compute_logits and the samplers "
        "(default: powers of two from 8 up to --max-running-requests).",
    )
    group.add_argument(
        "--num-blocks",
        type=int,
        default=None,
        help="KV blocks per layer the artifact declares (default: SGLang's pool).",
    )
    group.add_argument(
        "--target-tpu", default=None, help="TPU generation (default: detected)."
    )
    parser.add_argument(
        checkpoint.JAX_COORDINATOR_FLAG, default=None, help=argparse.SUPPRESS
    )
    return parser


def parse_args(argv: Sequence[str]) -> tuple[argparse.Namespace, Any]:
    from sglang.srt.server_args import ServerArgs

    args = build_parser().parse_args(list(argv))
    for name in ("token_buckets", "req_buckets"):
        setattr(args, name, _int_list(getattr(args, name)))
    if not args.token_buckets:
        raise SystemExit("--token-buckets must name at least one bucket")
    if args.export_mode == "model":
        args.load_format = "dummy"
    server_args = ServerArgs.from_cli_args(args)
    return args, server_args


def run_export(
    args: argparse.Namespace, server_args: Any, port_args: Any = None, rank: int = 0
) -> api.TextOnlyLLMExportConfig:
    """Loads rank `rank` and writes its artifact to `args.output_dir`."""
    world = int(server_args.tp_size)
    if args.export_mode == "all":
        # Before any JAX use: the Pallas kernels the model loads with use JAX.
        checkpoint.init_jax(rank, world, getattr(args, "jax_coordinator", None))
    runner = sglang_adapter.load_model_runner(
        server_args, tp_rank=rank, port_args=port_args
    )
    exportable = sglang_adapter.SglangExportableLLM(
        runner,
        sglang_adapter.ExportOptions(
            token_buckets=args.token_buckets,
            req_buckets=args.req_buckets,
            target_tpu=args.target_tpu,
        ),
    )
    if args.num_blocks is not None:
        exportable.fit_num_blocks(args.num_blocks)
    weights = (
        checkpoint.host_tensors(exportable.model) if args.export_mode == "all" else None
    )
    exporter.export(exportable, args.output_dir)
    if weights is not None:
        checkpoint.save_model_checkpoint(
            weights,
            args.output_dir,
            getattr(args, "checkpoint_data_dir", None)
            or os.path.join(args.output_dir, checkpoint.CHECKPOINT_DATA_DIR),
        )
    return exportable.export_config


def _rank_main(
    args: argparse.Namespace, server_args: Any, port_args: Any, rank: int
) -> None:
    """Entry of one rank's process (multiprocessing spawn, as SGLang's schedulers)."""
    args = argparse.Namespace(**vars(args))
    args.output_dir = os.path.join(args.rank_root, f"rank{rank}")
    os.makedirs(args.output_dir, exist_ok=True)
    run_export(args, server_args, port_args, rank)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def launch_workers(args: argparse.Namespace, server_args: Any) -> None:
    """Runs one export process per tensor-parallel rank and keeps rank 0's artifact.

    Started as SGLang's engine starts its schedulers: the server arguments are
    resolved and the ports picked once, here, and every rank gets the same
    objects through a spawned `multiprocessing.Process`. The ranks also join one
    JAX distributed runtime for the checkpoint; each writes its own artifact
    directory, of which rank 0's is copied to `args.output_dir`.
    """
    import multiprocessing as mp

    tp_size = int(server_args.tp_size)
    port_args = sglang_adapter.prepare_server_args(server_args)
    args.jax_coordinator = f"127.0.0.1:{_free_port()}"
    output_dir = os.path.abspath(args.output_dir)
    if args.export_mode == "all":
        args.checkpoint_data_dir = os.path.join(
            output_dir, checkpoint.CHECKPOINT_DATA_DIR
        )
        shutil.rmtree(args.checkpoint_data_dir, ignore_errors=True)
    parent = os.path.dirname(output_dir)
    os.makedirs(parent, exist_ok=True)
    args.rank_root = tempfile.mkdtemp(prefix=".export_ranks_", dir=parent)
    ctx = mp.get_context("spawn")
    procs = []
    try:
        for rank in range(tp_size):
            proc = ctx.Process(
                target=_rank_main, args=(args, server_args, port_args, rank)
            )
            proc.start()
            procs.append(proc)
        for proc in procs:
            proc.join()
        failed = [r for r, proc in enumerate(procs) if proc.exitcode != 0]
        if failed:
            raise RuntimeError(f"export processes for ranks {failed} failed")
        shutil.copytree(
            os.path.join(args.rank_root, "rank0"), output_dir, dirs_exist_ok=True
        )
    finally:
        for proc in procs:
            if proc.is_alive():
                proc.kill()
        shutil.rmtree(args.rank_root, ignore_errors=True)


def main(argv: Sequence[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    args, server_args = parse_args(argv)
    tp = int(server_args.tp_size)
    if tp > 1:
        launch_workers(args, server_args)
        print(f"exported {tp} ranks -> {args.output_dir}")
        return
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.makedirs(args.output_dir, exist_ok=True)
    cfg = run_export(args, server_args)
    print(
        f"exported {cfg.model_name}: {len(cfg.token_buckets)} forward buckets, "
        f"{cfg.num_blocks} KV blocks of {cfg.block_size} tokens -> {args.output_dir}"
    )
