# deepseek-v4-flash-sm120

Runtime patch for serving **DeepSeek-V4-Flash FP8** on NVIDIA SM120 / RTX PRO 6000 Blackwell workstation GPUs with SGLang.

The stock `lmsysorg/sglang:deepseek-v4-blackwell` image contains FlashMLA sparse-decode kernels for SM90/SM100, but not SM120. On RTX PRO 6000 it can fail with:

```text
RuntimeError: Unsupported architecture for sparse decode fwd
```

This repo builds a small SM120 CUDA extension and injects it at runtime with:

```bash
-v ./build-docker:/dsv4:ro
-e PYTHONPATH=/dsv4
```

No SGLang image rebuild and no install inside the container are required.

## Correct current launch recipe

Important corrections:

- Do **not** pass `--chat-template` for DeepSeek-V4. SGLang has built-in DeepSeek-V4 OpenAI chat encoding via `encoding_dsv4.py` when `--tool-call-parser deepseekv4` or the DeepseekV4 architecture is detected.
- Do **not** disable CUDA graphs. Use `--cuda-graph-max-bs 32`.
- EAGLE speculative decoding works in the current recipe: one step, topk 1, two draft tokens.
- Use the SM120 patch mount and `PYTHONPATH=/dsv4`; otherwise FlashMLA sparse decode falls back to upstream and fails on SM120.

```bash
MODEL_DIR=/mnt/llm_models/DeepSeek-V4-Flash-FP8 PORT=8000 \
  scripts/launch_dsv4_flash_sm120.sh
```

Equivalent core SGLang flags:

```bash
python3 -m sglang.launch_server \
  --model-path /workspace/model \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name deepseek-v4-flash \
  --trust-remote-code \
  --tensor-parallel-size 4 \
  --context-length 393216 \
  --mem-fraction-static 0.85 \
  --max-running-requests 8 \
  --kv-cache-dtype fp8_e4m3 \
  --tool-call-parser deepseekv4 \
  --reasoning-parser deepseek-v4 \
  --attention-backend compressed \
  --fp8-gemm-backend triton \
  --moe-runner-backend triton \
  --chunked-prefill-size 8192 \
  --watchdog-timeout 3600 \
  --page-size 256 \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 1 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 2 \
  --speculative-attention-mode decode \
  --cuda-graph-max-bs 32 \
  --enable-return-routed-experts
```

## Build

```bash
git clone https://github.com/0xSero/deepseek-v4-flash-sm120.git
cd deepseek-v4-flash-sm120
git submodule update --init --recursive

docker pull lmsysorg/sglang:deepseek-v4-blackwell
scripts/build_in_sglang_docker.sh
```

The build writes the runtime package to `build-docker/`:

```text
build-docker/sitecustomize.py
build-docker/deepseek_v4_kernel/cuda.cpython-312-x86_64-linux-gnu.so
build-docker/deepseek_v4_kernel/_patch.py
```

## Model

Use the FP8 checkpoint:

```bash
export MODEL_DIR=/mnt/llm_models/DeepSeek-V4-Flash-FP8
huggingface-cli download sgl-project/DeepSeek-V4-Flash-FP8 \
  --local-dir "$MODEL_DIR" \
  --local-dir-use-symlinks False
```

DeepSeek's official V4 repos do **not** ship a Jinja chat template. They ship `encoding/encoding_dsv4.py`; current SGLang includes an adapted encoder at `sglang.srt.entrypoints.openai.encoding_dsv4` and uses it automatically for this model path/parser.

## Smoke test

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4-flash","temperature":0,"max_tokens":32,
       "messages":[{"role":"user","content":"Say OK only."}]}'
```

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `Unsupported architecture for sparse decode fwd` | Patch did not load. Check `-v ./build-docker:/dsv4:ro`, `PYTHONPATH=/dsv4`, and `build-docker/sitecustomize.py`. |
| No `deepseek_v4_kernel.patch_flash_mla installed` log | `sitecustomize.py` was not imported; inspect `PYTHONPATH`. |
| Tool calls or thinking formatting looks wrong | Do not use a custom Jinja template; keep `--tool-call-parser deepseekv4` and `--reasoning-parser deepseek-v4`. |
| OOM near full context | Lower `--mem-fraction-static` or `--max-running-requests`. |

## License

Kernel, scripts, and docs are Apache-2.0. CUTLASS under `csrc/cutlass/` keeps its NVIDIA BSD-3 license. Model weights and SGLang image keep their upstream licenses.

## SM120 HMMA Tensor Core Optimization (feat/hmma-tensor-core-sparse-decode)

The `feat/hmma-tensor-core-sparse-decode` branch replaces the original scalar CUDA-core sparse decode kernel with an HMMA tensor core implementation, achieving **2.2–2.5× speedup** on both TTFT and decode throughput.

### Latest: HMMA vs upstream Triton sparse decode (2026-06-08)

SGLang has since merged an **in-tree Triton SM120 sparse-decode kernel** (`flash_mla_sm120_triton.py`, PR #24692) as the first-class SM120 path. Measured end-to-end against our HMMA `.so`, on the **same** server config — DeepSeek-V4-Flash, 4× RTX PRO 6000, TP=4, native MXFP4 fused-MoE experts (FlashInfer CuTe-DSL `MmaMXF4Op`), CUDA graphs on — swapping **only** the sparse-decode kernel:

| Concurrency | Triton sparse decode | **HMMA sparse decode** | HMMA speedup |
|---:|---:|---:|---:|
| 1  | 13 tok/s | **80 tok/s**  | 6.2× |
| 4  | 37 tok/s | **265 tok/s** | 7.2× |
| 8  | 49 tok/s | **479 tok/s** | 9.8× |
| 16 | 54 tok/s | **757 tok/s** | **14×** |

Decode throughput (256-token output, `ignore_eos`, greedy). Triton plateaus near ~54 tok/s and does **not** scale with concurrency; HMMA scales to 757 tok/s at 16-wide. Both produce correct output; the only variable is the attention sparse-decode kernel. The kernel is selected at runtime via `SGLANG_SM120_SPARSE_DECODE=hmma|triton` (default `hmma`).

> This kernel is a drop-in for FlashMLA's `sparse_decode_fwd` and is orthogonal to the MoE path: it pairs with either the native MXFP4 fused MoE or any other expert backend. The MoE experts above run the native MXFP4×MXFP4 FlashInfer kernels; the decode numbers isolate the attention kernel only.

### What changed

| Optimization | Impact |
|---|---|
| HMMA QK^T (`mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`) | Replaced scalar BF16 dot-products with tensor core matmul for attention scores |
| HMMA P@V | Same tensor core path for probability × value accumulation |
| Register-resident FP32 O accumulator | Moved output from 32 KB SMEM to 32 registers/thread — eliminates SMEM traffic |
| Split-KV adaptive parallelism | Spreads one CTA's work across up to 128 CTAs for better SM utilization during decode |
| KV_CHUNK=64 / 8 warps | Halves chunk iterations and barriers; natural 8 QK^T N-tiles |
| UE8M0 bitcast dequant | `__uint_as_float(b << 23)` replaces `__powf` for FP8 scale conversion |

### Direct comparison vs upstream scalar kernel (same hardware, same prompts)

| Context | Upstream Scalar Decode | HMMA Optimized Decode | Decode Speedup |
|---:|---:|---:|---:|
| 8K | 37.59 tok/s | **47.29 tok/s** | **1.26×** |
| 16K | 37.99 tok/s | **44.23 tok/s** | **1.16×** |
| 32K | 35.50 tok/s | **40.48 tok/s** | **1.14×** |
| 64K | 30.33 tok/s | **34.24 tok/s** | **1.13×** |

> Note: Upstream used EAGLE speculative decoding (broken on SM120 — produces garbled output). Our decode speedup is from HMMA tensor cores + split-KV + register-sO + KV64/8w, without speculative decoding.

### TTFT (prefill latency):

| Context | Original scalar | HMMA optimized | Speedup |
|---:|---:|---:|---:|
| 8K | 4.1s | **1.9s** | **2.2×** |
| 16K | 8.3s | **3.8s** | **2.2×** |
| 32K | 18.2s | **8.3s** | **2.2×** |
| 64K | 39.1s | **15.6s** | **2.5×** |

**Decode throughput (single stream):**

| Context | Original scalar | HMMA optimized | Speedup |
|---:|---:|---:|---:|
| 256 | 35.3 tok/s | **57.6 tok/s** | **1.6×** |
| 4K | 21.3 tok/s | **48.1 tok/s** | **2.3×** |
| 16K | 20.8 tok/s | **45.4 tok/s** | **2.2×** |
| 32K | 18.0 tok/s | **39.9 tok/s** | **2.2×** |
| 64K | 14.5 tok/s | **33.9 tok/s** | **2.3×** |

**Decode ITL (steady-state inter-token latency, 128 output tokens):**

| Context | Median ITL | Steady tok/s |
|---:|---:|---:|
| 256 | 16.4ms | 57.2 |
| 1K | 17.8ms | 52.3 |
| 4K | 20.0ms | 47.2 |
| 8K | 20.3ms | 46.2 |
| 16K | 21.4ms | 44.0 |

### SM120 hardware constraints discovered

- 100 KB shared memory per SM (not 228 KB like datacenter Blackwell)
- Block-scaled MMA (`.kind::mxf8f6f4`, `.block_scale`) — **not supported** on SM120
- Scaled packed FP8→BF16 conversion (`cvt.rn.satfinite.scaled::n2::ue8m0`) — **not supported** on SM120
- FP8 `mma.sync.aligned.m16n8k32` works but is slower than BF16 HMMA for this kernel due to per-tile scale overhead without block-scale MMA
- `torch.compile` / piecewise CUDA graphs — crash on SM120
- EAGLE speculative decoding — produces garbled output on SM120 (draft token verification bug)
- 4 warp schedulers per SM; h_q=16 per TP shard → base_ctas=1 for single-request decode

### NCCL tuning for PCIe Max-Q

These environment variables are critical for RTX PRO 6000 workstations (PCIe, no NVLink):

```bash
NCCL_PROTO=LL
NCCL_ALGO=Ring
NCCL_MIN_NCHANNELS=8
NCCL_NTHREADS=512
```

Also required: `--disable-custom-all-reduce` (prevents NCCL deadlock on PCIe Max-Q topology).

## Original scalar kernel benchmark (upstream, pre-HMMA)

> These numbers are from the **original scalar kernel** before the HMMA optimization branch. See the "SM120 HMMA Tensor Core Optimization" section above for current performance.

Configuration: no Jinja template, SGLang built-in `encoding_dsv4`, `SGLANG_ENABLE_THINKING=1`, `SGLANG_REASONING_EFFORT=max`, EAGLE draft tokens 2, CUDA graphs enabled.

| Context | TTFT | Prefill tok/s | Decode tok/s | Needle accuracy |
|---:|---:|---:|---:|:---:|
| 8K | 11.337s | 701.6 | 37.59 | yes |
| 16K | 12.770s | 1250.0 | 37.99 | yes |
| 32K | 25.101s | 1273.1 | 35.50 | yes |
| 64K | 55.850s | 1145.1 | 30.33 | yes |
| 128K | 135.271s | 946.0 | 24.08 | yes |
| 196K | 246.863s | 793.8 | 19.33 | yes |
| 300K | 464.072s | 646.4 | 15.09 | yes |

> **TODO:** Re-run this full benchmark suite with the HMMA kernel at all context lengths including 128K–300K with needle accuracy verification.
