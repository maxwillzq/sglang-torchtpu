# Results: SGLang-exported models served by rvllm

These are measurements of artifacts exported with `sglang_tpu.experimental.export` and served
by [rvllm](https://github.com/maxwillzq/rvllm-oss). Each artifact is compared with
`python -m sglang.launch_server` serving the same checkpoint on the same TPU host. The numbers
come from the daily run of 2026-11-06 on TPU v7x (`tpu7x`, one 2x2x1 host).

## Summary

| model | parallelism | top-1 agreement | rvllm / SGLang at concurrency 1 / 8 / 16 |
|---|---|---:|---|
| Qwen3-0.6B | TP 1 | 97% | 2.61x / 1.94x / 1.52x |
| Qwen3-4B | TP 1 | 94% | 1.01x / 1.02x / 1.09x |
| Qwen2.5-7B-Instruct | TP 1 | 100% | 1.01x / 1.04x / 1.13x |
| Qwen3-32B | TP 1 | 94% | 1.01x / 1.04x / 1.29x |
| Qwen3.5-0.8B (hybrid GDN) | TP 1 | 100% | 3.63x / 3.18x / 2.88x |
| Qwen3.5-0.8B + MTP (rvllm only, see below) | TP 1 | 100% | 5.04x / 4.31x / 3.37x |

rvllm is ahead where host-side scheduling dominates: on small models, and at higher concurrency.
From 4B up, a step at concurrency 1 is device-bound and the two are even. TTFT is lower with
rvllm on every model.

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

Parity: next-token top-1 agreement 97%; mean |Δ logprob| 0.042 (p95 0.125, max 0.351); identical greedy text 69%.

| concurrency | SGLang tok/s | rvllm tok/s | rvllm / SGLang | SGLang TTFT ms | rvllm TTFT ms | SGLang TPOT ms | rvllm TPOT ms |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 301.1 | 784.7 | 2.61x | 8.8 | 4.7 | 3.3 | 1.2 |
| 8 | 2182.8 | 4225.2 | 1.94x | 27.4 | 10.6 | 3.5 | 1.8 |
| 16 | 4134.3 | 6279.0 | 1.52x | 38.2 | 13.5 | 3.6 | 2.4 |

### Qwen3-4B (TP 1)

Parity: next-token top-1 agreement 94%; mean |Δ logprob| 0.047 (p95 0.124, max 0.245); identical greedy text 94%.

| concurrency | SGLang tok/s | rvllm tok/s | rvllm / SGLang | SGLang TTFT ms | rvllm TTFT ms | SGLang TPOT ms | rvllm TPOT ms |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 227.7 | 230.4 | 1.01x | 18.6 | 11.1 | 4.3 | 4.3 |
| 8 | 1487.3 | 1522.9 | 1.02x | 83.9 | 48.8 | 4.8 | 4.9 |
| 16 | 2408.2 | 2616.6 | 1.09x | 111.2 | 43.9 | 5.8 | 5.8 |

### Qwen2.5-7B-Instruct (TP 1)

Parity: next-token top-1 agreement 100%; mean |Δ logprob| 0.065 (p95 0.211, max 0.375); identical greedy text 100%.

| concurrency | SGLang tok/s | rvllm tok/s | rvllm / SGLang | SGLang TTFT ms | rvllm TTFT ms | SGLang TPOT ms | rvllm TPOT ms |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 163.2 | 165.4 | 1.01x | 22.9 | 14.8 | 6.0 | 6.0 |
| 8 | 1154.9 | 1206.7 | 1.04x | 100.6 | 62.3 | 6.2 | 6.2 |
| 16 | 2011.8 | 2264.1 | 1.13x | 148.5 | 54.1 | 6.8 | 6.6 |

### Qwen3-32B (TP 1)

Runs with `--max-total-tokens 65536 --mem-fraction-static 0.9` to fit one chip.

Parity: next-token top-1 agreement 94%; mean |Δ logprob| 0.046 (p95 0.121, max 0.246); identical greedy text 88%.

| concurrency | SGLang tok/s | rvllm tok/s | rvllm / SGLang | SGLang TTFT ms | rvllm TTFT ms | SGLang TPOT ms | rvllm TPOT ms |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 37.9 | 38.2 | 1.01x | 84.4 | 55.0 | 25.9 | 26.0 |
| 8 | 267.3 | 276.8 | 1.04x | 428.8 | 394.3 | 26.8 | 25.9 |
| 16 | 404.5 | 521.8 | 1.29x | 589.5 | 416.3 | 29.0 | 27.4 |

### Qwen3.5-0.8B (TP 1, hybrid Gated DeltaNet)

Parity: next-token top-1 agreement 100%; mean |Δ logprob| 0.028 (p95 0.101, max 0.122); identical greedy text 81%.

| concurrency | SGLang tok/s | rvllm tok/s | rvllm / SGLang | SGLang TTFT ms | rvllm TTFT ms | SGLang TPOT ms | rvllm TPOT ms |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 251.3 | 912.2 | 3.63x | 16.1 | 5.3 | 3.9 | 1.1 |
| 8 | 1916.2 | 6095.2 | 3.18x | 50.9 | 16.3 | 3.8 | 1.2 |
| 16 | 3518.6 | 10119.3 | 2.88x | 56.4 | 21.1 | 4.1 | 1.4 |

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
Greedy MTP does not change the output, so the parity numbers are valid; they equal the run without MTP above. Against rvllm without MTP, MTP adds 1.45x / 1.31x / 1.22x.

Parity:
- next-token top-1 agreement: 100%;
- mean |Δ logprob|: 0.028 (p95 0.101, max 0.122);
- identical greedy text: 81%.

| concurrency | SGLang tok/s | rvllm tok/s | rvllm / SGLang | SGLang TTFT ms | rvllm TTFT ms | SGLang TPOT ms | rvllm TPOT ms |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 262.0 | 1320.2 | 5.04x | 14.4 | 5.8 | 3.7 | 0.7 |
| 8 | 1850.9 | 7981.0 | 4.31x | 52.4 | 14.7 | 3.9 | 0.8 |
| 16 | 3669.7 | 12357.5 | 3.37x | 56.6 | 19.5 | 3.9 | 1.0 |

On three greedy prompts on a dev pod, outside the benchmark, rvllm accepted 109 of the 133 drafted
tokens (82%): 88% at draft position 0 and 76% at position 1.

## Exported and served, not benchmarked yet

These models were exported on a dev pod, and rvllm's greedy output on short prompts was checked
by hand. None of them has been through the parity and throughput job above yet.

| model | parallelism | notes |
|---|---|---|
| Qwen3-0.6B | TP 2 | rvllm output matches TP 1 |
| Qwen3-30B-A3B-FP8 | TP 2 | MoE, FP8 weights |
| Qwen3.5-35B-A3B-FP8 | TP 2 | hybrid MoE, FP8 weights |

The daily run also includes Qwen3-Coder-480B-A35B-Instruct-FP8 and Qwen3.5-397B-A17B-FP8 at TP 8.
Their first runs ran out of disk while downloading the checkpoint, and they are being rerun
with a larger volume.

## Known gaps

- Hybrid models run the Gated DeltaNet kernel's per-sequence segment for every request. This is
  correct but slower than the batched decode segment, so Qwen3.5 throughput has room to grow.
- The MTP comparison needs SGLang's NEXTN serving to work on Qwen3.5 before it is like for like.
- Throughput is measured on random token ids only, not on a real dataset.
