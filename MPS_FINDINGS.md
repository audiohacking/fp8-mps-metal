# MPS (Metal Performance Shaders) — Complete Findings

> Validated benchmarks and failure modes from testing 40+ repositories on M4 Pro (48GB), macOS 26.2, PyTorch 2.10.0

This document contains raw benchmark data and technical details backing the claims in the README.

## Table of Contents
- [Operations That Fail](#operations-that-fail)
- [FP8 Deep Dive](#fp8-deep-dive)
- [MPS vs CPU Crossover Points](#mps-vs-cpu-crossover-points)
- [LoRA Performance (Corrected)](#lora-performance-corrected)
- [Elementwise Fusion](#elementwise-fusion)
- [Dtype Conversion](#dtype-conversion)
- [INT8 Dequantization](#int8-dequantization)
- [MPSGraph Fusion](#mpsgraph-fusion)
- [Quantization Comparison](#quantization-comparison)
- [Weight Operation Costs](#weight-operation-costs)
- [Memory Allocation](#memory-allocation)
- [Transformer Building Blocks](#transformer-building-blocks)

---

## Operations That Fail

Tested with `PYTORCH_ENABLE_MPS_FALLBACK=0` to expose real failures (not silent CPU fallbacks).

### Hard Failures (RuntimeError)
| Operation | Error Message | Affected Models |
|-----------|--------------|-----------------|
| `tensor.to(torch.float8_e4m3fn)` | "does not have support for that dtype" | FLUX, SD3.5, any FP8 quantized model |
| `tensor.to(torch.float8_e5m2)` | "does not have support for that dtype" | Training with FP8 gradients |
| `torch._scaled_mm(fp8, fp8)` | "not implemented for MPS" | FP8 inference pipeline |
| `torch.quantize_per_tensor(...)` | "QuantizedMPS backend not implemented" | PyTorch native quantization |
| `torch.quantize_per_channel(...)` | "QuantizedMPS backend not implemented" | Per-channel quantization |
| `torch.linalg.eigh(...)` | "not implemented for MPS" | PCA, spectral methods |
| `torch.linalg.qr(...)` | "not implemented for MPS" | QR decomposition |

### Silent CPU Fallbacks (Warning only)
| Operation | Behavior |
|-----------|----------|
| `torch.linalg.svd(...)` | Prints warning, runs on CPU, returns to MPS |
| Any op with `PYTORCH_ENABLE_MPS_FALLBACK=1` | Silently runs on CPU |

### Operations That Work Fine
All standard diffusion/transformer operations pass:
- `F.scaled_dot_product_attention` (all mask types, GQA with 32Q/8KV heads)
- `conv2d`, `conv3d`, `conv_transpose2d`
- `F.group_norm`, `F.layer_norm`, `F.instance_norm`
- `F.interpolate` (nearest, bilinear, bicubic)
- `scatter_add`, `gather`, `index_select`, `embedding`
- `multinomial`, `topk(50000, 50)`, `searchsorted`, `gumbel_softmax`
- `torch.fft.rfft2`, `torch.fft.irfft2`, `torch.fft.fft`
- `F.cross_entropy`, `einsum`

---

## FP8 Deep Dive

### What FP8 e4m3fn Is
```
Bit layout: [sign:1][exponent:4][mantissa:3]
Bias: 7
Range: ±448.0
Precision: 3 mantissa bits = 8 distinct mantissa values per exponent
Special: 0x7F and 0xFF = NaN (no infinity representation)

Comparison:
  FP8 e4m3fn: 1 + 4 + 3 = 8 bits, range ±448, ~3 decimal digits
  FP16:       1 + 5 + 10 = 16 bits, range ±65504, ~3.3 decimal digits
  FP32:       1 + 8 + 23 = 32 bits, range ±3.4e38, ~7.2 decimal digits
```

### Why Models Use FP8
FLUX (12B params): FP32 = 48GB, FP16 = 24GB, **FP8 = 12GB**. FP8 halves memory vs FP16 while maintaining acceptable accuracy for inference. NVIDIA added hardware FP8 tensor cores in Ada Lovelace (RTX 4090, 2022). The ecosystem moved fast.

### The MPS FP8 Gap (Detailed)

```python
import torch

# This works (empty storage, no compute):
t = torch.empty(4, dtype=torch.float8_e4m3fn, device="cpu")
t_mps = t.to("mps")  # Raw bytes transfer — works

# This fails (actual compute):
x = torch.randn(4, device="mps")
x.to(torch.float8_e4m3fn)  # RuntimeError: MPS doesn't support dtype

# This fails (the real blocker):
a = torch.empty(4, 4, dtype=torch.float8_e4m3fn, device="mps")
b = torch.empty(4, 4, dtype=torch.float8_e4m3fn, device="mps")
torch._scaled_mm(a, b, scale_a=..., scale_b=...)  # NotImplementedError
```

### Our Solution: Metal Kernel FP8 Decode
Since Metal has no native FP8 type, we store as `uint8_t` and decode per-element:

```metal
inline float fp8_e4m3fn_to_float(uint8_t bits) {
    if ((bits & 0x7F) == 0x7F) return 0.0f;  // NaN → 0

    uint sign = (bits >> 7) & 1;
    uint exp_bits = (bits >> 3) & 0xF;
    uint mant_bits = bits & 0x7;

    float value;
    if (exp_bits == 0) {
        // Subnormal: 2^(1-bias) * (0.mantissa) = 2^(-6) * mant/8
        value = float(mant_bits) / 8.0f * (1.0f / 64.0f);
    } else {
        // Normal: 2^(exp-bias) * (1.mantissa)
        value = (1.0f + float(mant_bits) / 8.0f) * exp2(float(int(exp_bits) - 7));
    }
    return sign ? -value : value;
}
```

Accuracy: **0.0 error** across all 256 possible uint8 values when compared to Python reference decoder.

### FP8 Workaround Benchmarks (4096×4096 matmul, M4 Pro)

| Path | Latency | vs FP16 Native | Notes |
|------|---------|---------------|-------|
| FP16 native matmul | 0.91 ms | 1.0x | Ideal baseline |
| **Our fused Metal FP8 kernel** | **0.66 ms** | **0.7x** | Yes, faster at M=1 due to vecmat SIMD |
| INT8 GPU dequant + matmul | 1.41 ms | 1.54x | Best existing workaround |
| INT8 per-channel dequant | 2.28 ms | 2.49x | Better accuracy, slower |
| FP8 CPU dequant + transfer + matmul | 3.42 ms | 3.74x | What ComfyUI does today |
| **Our Metal FP8 kernel (K=14336)** | **2.38 ms** | — | vs 62.8ms CPU = **26x speedup** |

---

## MPS vs CPU Crossover Points

MPS has higher kernel dispatch overhead than CPU. Small operations are faster on CPU (using Apple's AMX matrix accelerator).

| Matrix Size | CPU Time | MPS Time | MPS/CPU Ratio | Winner |
|-------------|----------|----------|---------------|--------|
| 32×32 | 0.025 ms | 2.854 ms | 114x slower | CPU |
| 128×128 | 0.007 ms | 0.090 ms | 13x slower | CPU |
| 256×256 | 0.010 ms | 0.050 ms | 5x slower | CPU |
| 512×512 | 0.200 ms | 0.110 ms | **1.8x faster** | **MPS** |
| 1024×1024 | 0.800 ms | 0.400 ms | 2.0x faster | MPS |
| 2048×2048 | 5.676 ms | 3.155 ms | 1.8x faster | MPS |
| 4096×4096 | 40+ ms | 6+ ms | 6x faster | MPS |

**Rule of thumb:** MPS wins at 512×512 and above. Below that, CPU is faster.

---

## LoRA Performance (Corrected)

**Important correction:** Earlier community findings that "separate LoRA is faster on MPS" were wrong. The error was caused by accidentally re-merging weights every forward pass in the benchmark.

### Actual Results: Pre-Merged Always Wins

| Configuration | MPS Separate | MPS Merged | MPS Ratio | CPU Separate | CPU Merged | CPU Ratio |
|--------------|-------------|-----------|-----------|-------------|-----------|-----------|
| 4096→4096, r=16 | 1.60 ms | 0.95 ms | **1.67x faster merged** | 2.84 ms | 2.21 ms | 1.28x |
| 4096→4096, r=64 | 1.62 ms | 0.97 ms | **1.68x faster merged** | 2.94 ms | 2.21 ms | 1.33x |
| 4096→11008, r=16 | 2.29 ms | 2.12 ms | 1.08x faster merged | 7.07 ms | 5.79 ms | 1.22x |

**Why MPS penalty is worse:** MPS dispatch overhead is ~10% higher than CPU per kernel. Three small matmuls (base + A + B) pay 3x the dispatch tax.

### LoRA Hot-Swap Speed (7 target modules, Qwen3-7B-like)

| LoRA Rank | Adapter Size | Re-merge Time |
|-----------|-------------|--------------|
| r=8 | 13 MB | 39 ms |
| r=16 | 26 MB | 26 ms |
| r=64 | 104 MB | 37 ms |

Re-merging is cheap enough to do on every adapter switch.

---

## Elementwise Fusion

MPS genuinely fuses elementwise operations after matmul at zero cost. This is a real Metal advantage over CPU.

| Operation Added After Matmul | MPS Overhead | CPU Overhead |
|------------------------------|-------------|-------------|
| + bias | -0.4% (free) | -1.7% |
| + residual add | -4.0% (free) | **+11.4%** |
| + bias + SiLU activation | -0.3% (free) | **+8.9%** |

**Negative overhead = within noise = free.** CPU pays 9-11% for the same chain. MPSGraph automatically fuses consecutive elementwise ops into a single kernel.

---

## Dtype Conversion

On NVIDIA, FP32→FP16 (downcast) is faster than FP16→FP32 (upcast). On MPS, **both directions are identical speed.**

| Direction | MPS Bandwidth | CPU Bandwidth | Ratio (up/down) |
|-----------|-------------|--------------|-----------------|
| FP32 → FP16 | 90 GB/s | 54 GB/s | |
| FP16 → FP32 | 93 GB/s | 33 GB/s | |
| **Asymmetry** | **0.97x (symmetric)** | **1.60x (asymmetric)** | |

This is because MPS uses unified memory (no host↔device transfer), while CPU's AMX accelerator has asymmetric cast instructions.

---

## INT8 Dequantization

INT8 dequantization genuinely runs on the Metal GPU, not falling back to CPU.

| Operation | MPS | CPU | GPU Speedup |
|-----------|-----|-----|------------|
| INT8 dequant (4096×4096) | 1.89 ms | 5.08 ms | **2.69x** |
| INT8 dequant + matmul (4096×4096) | 2.47 ms | 7.00 ms | **2.84x** |
| INT8 dequant + matmul (4096×11008) | 6.41 ms | 16.66 ms | **2.60x** |

Consistent ~2.6x GPU acceleration for INT8 operations.

---

## MPSGraph Fusion

MPSGraph (PyTorch's MPS backend) automatically fuses some consecutive operations. The benefit is modest but real.

| Approach | Time | Speedup |
|----------|------|---------|
| RMSNorm fused (no sync barriers) | 1.46 ms | baseline |
| RMSNorm step-by-step (sync between each op) | 1.79 ms | |
| **Fusion benefit** | | **1.22x** |

For maximum performance, avoid `torch.mps.synchronize()` between operations in the forward pass. Let MPSGraph batch them.

---

## Quantization Comparison

| Method | MPS Time (1×4096 @ 4096×4096) | vs FP16 | Notes |
|--------|------------------------------|---------|-------|
| FP16 native | 0.91 ms | 1.0x | Baseline |
| INT8 symmetric | 2.21 ms | 2.4x | Best speed/memory tradeoff |
| INT4 grouped (g=128) | 4.13 ms | 4.5x | Bit unpacking overhead |
| **INT4/INT8 ratio** | **1.87x** | | **INT4 almost 2x slower than INT8** |

**On NVIDIA**, INT4 and INT8 are similar speed (hardware tensor cores handle both). **On MPS**, INT4 is ~2x slower because Metal has no native 4-bit type — every INT4 value needs bit shifting and masking to unpack.

---

## Weight Operation Costs

### View Operations (Free)
| Op | Time | Notes |
|----|------|-------|
| reshape | 0.00 ms | Same memory, different shape |
| chunk | 0.00 ms | Views into existing tensor |
| split | 0.00 ms | Views into existing tensor |

### Memory Operations
| Op | Time (4096×4096) | Notes |
|----|-----------------|-------|
| transpose | 1.06 ms | Real memory shuffle |
| cat (2 tensors) | 1.39 ms | New allocation + copy |
| clone | 0.75 ms | Full copy |

### Weight Loading (disk → CPU → MPS)
| Layer | Size | Time | Throughput |
|-------|------|------|-----------|
| q_proj (4096×4096) | 64 MB | 9.8 ms | 6.4 GB/s |
| gate_proj (11008×4096) | 172 MB | 22.6 ms | 7.4 GB/s |
| Full transformer layer | ~676 MB | ~92 ms | ~7.2 GB/s |

---

## Memory Allocation

| Allocation Size | Default Watermark | Watermark=0.0 |
|-----------------|------------------|---------------|
| 8 GB | OK | OK |
| 16 GB | OK | OK |
| 24 GB | OK (max) | OK |
| 28 GB | FAIL | OK (max) |
| 32 GB | FAIL | FAIL (Metal hard cap) |
| 48 GB (6×8GB) | OK | OK |

**Multi-buffer access confirmed:** 6 × 8GB buffers = 48GB total, using full unified memory. The 32GB per-buffer cap is a Metal framework limit, not a hardware limit.

---

## Transformer Building Blocks

All benchmarks: M4 Pro, batch=1, hidden=4096 unless noted.

| Block | Config | Time |
|-------|--------|------|
| RMSNorm | seq=2048, hidden=4096 | 2.2 ms |
| SwiGLU FFN | hidden=4096, intermediate=10924 | 21.5 ms |
| RoPE | seq=2048, 32 heads, d=128 | 2.8 ms |
| SDPA Attention | seq=2048, 32 heads, d=128 | 31 ms |
| KV-Cache (cold start) | First step | ~11 ms |
| KV-Cache (steady state) | cache=256, per step | 0.25 ms |
| KV-Cache (256 steps total) | | 65 ms |

---

## Methodology

- All MPS benchmarks use `torch.mps.synchronize()` after each operation
- Timing: `time.perf_counter()`, warmup=2-5 runs, measurement=5-20 runs, median reported
- Environment: `PYTORCH_ENABLE_MPS_FALLBACK=0` for failure testing, `=1` for performance testing
- CPU benchmarks use the same PyTorch operations without `.to("mps")`
- Memory measured via `torch.mps.current_allocated_memory()` and `torch.mps.driver_allocated_memory()`
