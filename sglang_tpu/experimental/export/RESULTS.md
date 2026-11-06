# Results: SGLang-exported models served by rvllm

These are measurements of artifacts exported with `sglang_tpu.experimental.export` and served
by [rvllm](https://github.com/maxwillzq/rvllm-oss). Each artifact is compared with
`python -m sglang.launch_server` serving the same checkpoint on the same TPU host. The numbers
are from 2026-11-05 on TPU v7x (`tpu7x`, one 2x2x1 host). They are early results: two models
have parity and throughput numbers, and the rest have only been checked for correct output.

## How each model is measured

Each model goes through one end-to-end job:

1. Export with `--export-mode all` (weights included), using the server arguments below.
2. rvllm one-shot: the prompt `The capital of France is` must produce `Paris`.
3. Parity. 16 prompts are sent to both servers at temperature 0, and for each prompt the
   next-token top-5 logprobs are compared at two prompt lengths. The metrics are:
   - next-token top-1 agreement;
   - mean and p95 of |Δ logprob| of the top-1 token;
   - the share of 32-token greedy completions that are identical. Greedy text diverges after
     the first near-tie, so this number is lower than the top-1 agreement.
4. Throughput with `python -m sglang.bench_serving --backend vllm --dataset-name random-ids`:
   - 256 input tokens and 128 output tokens per request, range ratio 1.0;
   - 4 × concurrency prompts at each of concurrency 1, 8 and 16.

Both servers run with the same static sizes:

```text
--dtype bfloat16 --attention-backend pallas_tpu_rpa_v3 --page-size 64
--context-length 2048 --max-running-requests 16
--kv-cache-dtype bfloat16 --disable-radix-cache --grammar-backend none
```

Exports use token buckets 16, 32, 64, 128, 256 and 512. The baseline SGLang server runs with
`--enable-torch-compile`, the setting its own TPU benchmark recipes use. Models at TP 1 run on
one chip (`TPU_VISIBLE_CHIPS=0`).

## Parity and throughput

### Qwen3-0.6B (TP 1)

Parity:
- next-token top-1 agreement: 97%;
- mean |Δ logprob|: 0.042 (p95 0.125, max 0.351);
- identical greedy text: 69%.

| concurrency | SGLang tok/s | rvllm tok/s | rvllm / SGLang | SGLang TTFT ms | rvllm TTFT ms | SGLang TPOT ms | rvllm TPOT ms |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 292.5 | 785.4 | 2.68x | 9.3 | 4.6 | 3.4 | 1.2 |
| 8 | 2221.9 | 4274.0 | 1.92x | 26.4 | 9.9 | 3.4 | 1.8 |
| 16 | 4209.8 | 6304.4 | 1.50x | 37.8 | 12.8 | 3.5 | 2.4 |

### Qwen3.5-0.8B with MTP (TP 1, hybrid Gated DeltaNet)

The artifact is exported with
`--speculative-algorithm NEXTN --speculative-num-steps 2 --speculative-eagle-topk 1
--speculative-num-draft-tokens 3`, so rvllm drafts 2 tokens per step with the model's MTP
layer.

The baseline SGLang server runs **without** speculative decoding. Its NEXTN path fails on this
model: the draft extend asserts on the multimodal embeddings, in
`chain_spec_worker._extend_draft`. So the throughput rows below compare rvllm with MTP against
SGLang without it, and do not compare the two runtimes like for like. Random token-id prompts
also make continuations repetitive, so draft acceptance is likely higher than on real text.
Greedy MTP does not change the output, so the parity numbers are valid.

Parity:
- next-token top-1 agreement: 100%;
- mean |Δ logprob|: 0.028 (p95 0.101, max 0.122);
- identical greedy text: 81%.

| concurrency | SGLang tok/s | rvllm tok/s | rvllm / SGLang | SGLang TTFT ms | rvllm TTFT ms | SGLang TPOT ms | rvllm TPOT ms |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 253.5 | 1317.7 | 5.20x | 15.6 | 5.9 | 3.8 | 0.7 |
| 8 | 1887.4 | 8032.5 | 4.26x | 50.5 | 14.5 | 3.9 | 0.8 |
| 16 | 3515.1 | 12638.2 | 3.60x | 58.8 | 19.5 | 4.1 | 1.0 |

On three greedy prompts on a dev pod, outside the benchmark, rvllm accepted 109 of the 133 drafted
tokens (82%): 88% at draft position 0 and 76% at position 1.

## Exported and served, not benchmarked yet

These models were exported on a dev pod, and rvllm's greedy output on short prompts was checked
by hand. None of them has been through the parity and throughput job above yet.

| model | parallelism | notes |
|---|---|---|
| Qwen3-0.6B | TP 2 | rvllm output matches TP 1 |
| Qwen2.5-7B-Instruct | TP 1 | |
| Qwen3-30B-A3B-FP8 | TP 2 | MoE, FP8 weights |
| Qwen3.5-0.8B | TP 1 | hybrid Gated DeltaNet, no MTP |
| Qwen3.5-35B-A3B-FP8 | TP 2 | hybrid MoE, FP8 weights |

The daily run also includes Qwen3-4B, Qwen3-32B, Qwen3-Coder-480B-A35B-Instruct-FP8 (TP 8) and
Qwen3.5-397B-A17B-FP8 (TP 8). None of them has completed a run yet.

## Known gaps

- Hybrid models run the Gated DeltaNet kernel's per-sequence segment for every request. This is
  correct but slower than the batched decode segment, so Qwen3.5 throughput has room to grow.
- The MTP comparison needs SGLang's NEXTN serving to work on Qwen3.5 before it is like for like.
- Throughput is measured on random token ids only, not on a real dataset.
