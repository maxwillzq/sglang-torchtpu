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
"""

from __future__ import annotations

import argparse
import os
import sys
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
    group.add_argument("--output-dir", required=True, help="Artifact directory to write.")
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
    group.add_argument("--target-tpu", default=None, help="TPU generation (default: detected).")
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


def run_export(args: argparse.Namespace, server_args: Any) -> api.TextOnlyLLMExportConfig:
    if args.export_mode == "all":
        # Before any JAX use: the Pallas kernels the model loads with use JAX.
        checkpoint.init_jax(0, 1, None)
    runner = sglang_adapter.load_model_runner(server_args)
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
    weights = checkpoint.host_tensors(exportable.model) if args.export_mode == "all" else None
    exporter.export(exportable, args.output_dir)
    if weights is not None:
        checkpoint.save_model_checkpoint(
            weights,
            args.output_dir,
            os.path.join(args.output_dir, checkpoint.CHECKPOINT_DATA_DIR),
        )
    return exportable.export_config


def main(argv: Sequence[str] | None = None) -> None:
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    args, server_args = parse_args(sys.argv[1:] if argv is None else argv)
    if server_args.tp_size != 1:
        raise SystemExit(f"export supports --tp-size 1, got {server_args.tp_size}")
    os.makedirs(args.output_dir, exist_ok=True)
    cfg = run_export(args, server_args)
    print(
        f"exported {cfg.model_name}: {len(cfg.token_buckets)} forward buckets, "
        f"{cfg.num_blocks} KV blocks of {cfg.block_size} tokens -> {args.output_dir}"
    )
