# Experimental AOT export (SGLang on TPU)

`python -m sglang_tpu.experimental.export` exports a model SGLang serves on TPU to
a StableHLO serving directory that [rvllm](https://github.com/maxwillzq/rvllm-oss)
runs: `metadata.json`, per-bucket MLIR modules in `subgraphs/` and, with
`--export-mode all`, the weights in `serving_checkpoint/`. It takes
`python -m sglang.launch_server`'s arguments plus a few export options:

```bash
python -m sglang_tpu.experimental.export --model-path Qwen/Qwen3-0.6B \
  --output-dir /tmp/exported --export-mode all \
  --context-length 2048 --max-running-requests 16 --page-size 64 \
  --max-total-tokens 32768 --attention-backend pallas_tpu_rpa_v3 \
  --token-buckets 16,32,64,128,256
```

The artifact follows the same contract as vllm-torchtpu's exporter
(`vllm_torchtpu.experimental.export`), whose lowering, metadata, validator,
samplers and checkpoint writer this package shares. Only `sglang_adapter.py` and
`cli.py` are SGLang's.

## Static sizes come from the server arguments

| artifact (`serving_config`) | SGLang argument |
|---|---|
| `max_num_seqs` | `--max-running-requests` |
| `max_model_len` | `--context-length` |
| `block_size` | `--page-size` |
| `num_blocks` | the KV pool SGLang allocates (`--max-total-tokens`, `--mem-fraction-static`); override with `--num-blocks` |
| KV dtype | `--kv-cache-dtype` (SGLang's TPU default is fp8) |

Export options: `--token-buckets` (one `forward` graph each), `--req-buckets`
(`compute_logits` and samplers; default powers of two from 8 up to
`--max-running-requests`), `--num-blocks`, `--target-tpu`, `--debug-taps`.

## Export modes

| `--export-mode` | Weights | Writes |
|---|---|---|
| `model` (default) | dummy (`--load-format dummy`) | artifact without weights; rvllm runs it with `--debug-dummy-weight` |
| `all` | real, loaded once on TPU | artifact + Orbax checkpoint (Zarr v2, one `array_{i}` per `abstract_params` input) |

The weights are the ones SGLang computes with after loading, e.g. FP8 weights in
the TPU kernels' layout with their scales.

## How the forward maps onto SGLang

The artifact's `forward` is SGLang's own decoder (`model.model`) over the Pallas
ragged paged attention backend (RPA v3, `--attention-backend pallas_tpu_rpa_v3`).
Its control inputs become the backend's `PallasForwardMetadata`:
`kv_lens` → `kv_lens`, `block_tables` → `page_indices`, `query_start_loc` →
`cu_q_lens`, `req_dist` → `distribution`; `cu_kv_lens` is the rectangular page
stride and `active_num_tokens` is `query_start_loc[req_dist[2]]`. Each layer's KV
cache is the fused `[num_blocks, page_size, 2 * kv_heads / packing, packing,
head_dim]` buffer of the token-to-KV pool. `compute_logits` runs the
`LogitsProcessor`'s LM head (logit scale, soft-capping included). RoPE is
computed in-graph from `positions` instead of SGLang's `cos_sin_cache` buffer.

## Tensor and expert parallelism

With `--tp-size N > 1` (and `--ep-size` for MoE) the exporter starts one process
per rank the way SGLang's engine starts its schedulers: the server arguments are
resolved once and every rank gets them. Each rank traces its shard-local program
(collectives in the graph), rank 0's artifact is kept, and every rank writes its
own weight shards into the shared checkpoint. Shardings in `metadata.json`:

- linear weights split the dim their layer splits (`input_size_per_partition <
  input_size`: row parallel; otherwise column parallel), embeddings and the LM
  head the vocabulary;
- fused-MoE experts split dim 0 under expert parallelism, else the intermediate
  dim;
- KV caches split the fused heads axis.

Values that differ per rank and that SGLang keeps as Python ints (a vocab
shard's bounds) become per-rank buffers, exported as a constant input sharded
over the mesh, so rank 0's program reads each device's own value. The host's TPU
environment (`TPU_VISIBLE_CHIPS`, ...) must cover all ranks; do not pin a single
chip.

## Hybrid models (Gated DeltaNet)

Models on `HybridLinearAttnBackend` (Qwen3.5, Qwen3-Next) export each linear
layer's conv state `[num_state_slots, K-1, 1, conv_dim]` and recurrent state
`[num_state_slots, n_v, d_k, d_v]` (fp32, as SGLang keeps them), null slot 0 and
`num_state_slots = max_num_seqs + 1`. States are listed per layer in layer order
(`[L0.conv, L0.rec, ..., L3.kv, ...]`, as vllm-torchtpu's export lists them), and
the forward takes `mamba_state_indices[max_num_seqs]`, each request's slot.
Every request currently runs the GDN kernel's per-sequence segment (correct,
slower than the batched decode segment for decode steps).

## Speculative decoding (MTP)

With `--speculative-algorithm NEXTN` (and `--speculative-num-steps`, which sets the number of
draft tokens), the artifact also holds `draft_bucket{B}` modules for the model's MTP layer. The
target forward takes verify windows of recurrent state. The draft's own weights are exported as
`mtp.*`, and the tensors it shares with the target, such as the embedding and a tied LM head,
keep the target's names. The layout is the rvllm/vLLM one: the adapter rebuilds the GDN metadata
for it rather than using SGLang's own spec-decoding windows. Verified on Qwen3.5-0.8B.

## Results

[RESULTS.md](RESULTS.md) has parity and throughput against `sglang.launch_server` for the
models measured so far.

## Not supported yet

- `--export-mode checkpoint` (CPU-only weight export);
- attention backends other than RPA v3; MLA models.

## Debugging: `--debug-taps`

As in vllm-torchtpu: comma-separated glob patterns of module names (as in the
model's `named_modules()`, e.g. `model.layers.3.self_attn`); for the smallest
token bucket, `assets.extra/debug/forward_debug_bucket{B}.mlir` also returns
those modules' outputs, described in `assets.extra/debug/taps.json`.

## Tests

CPU: `python -m pytest test/registered/unit/hardware_backend/tpu/export`
(exporter pipeline with mocked lowering, checkpoint, RoPE patch, SGLang adapter).
End to end on TPU: the `rvllm-sglang-e2e` recipe exports a model, serves it with
rvllm and compares next-token logprobs and throughput with
`sglang.launch_server`.
