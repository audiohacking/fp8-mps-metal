# fp8-mps-metal

**Custom FP8 compute kernels for Apple Silicon (MPS) — fixing what PyTorch doesn't support yet.**

If you've tried to run FLUX, SD3.5, or any FP8-quantized model on your Mac and hit cryptic errors like `"does not have support for that dtype"` or `"_scaled_mm not implemented for MPS"` — this is the fix. This repo contains a Metal compute shader that performs FP8 (e4m3fn) dequantization and matrix multiplication directly on the GPU, exposed to PyTorch as a drop-in monkey-patch.

> **Tested on:** M4 Pro (48GB), macOS 26.2, PyTorch 2.10.0, Python 3.12

## Quick Start for ComfyUI Users

**This is the easiest way to use FP8 models in ComfyUI on Apple Silicon:**

### Automatic

Browse `ComfyUI > Manage Extensions` and search for `fp8-mps-metal`

<img width="600" alt="ComfyUI Extension" src="https://github.com/user-attachments/assets/11024d0e-5b2a-413e-8d78-25ee59e3eb27" />


### Manual

```bash
cd /Users/USERNAME/Documents/ComfyUI/custom_nodes
git clone https://github.com/audiohacking/fp8-mps-metal.git
```

That's it! The patch will automatically load when ComfyUI starts. You'll see a message confirming it's installed.

## Quick Start for Other Users

```bash
git clone https://github.com/audiohacking/fp8-mps-metal.git
cd fp8-mps-metal

# Option A: Pure Python (no compilation needed — recommended)
python -c "
import torch, sys; sys.path.insert(0, '.')
import fp8_mps_patch
fp8_mps_patch.install()
print('FP8 MPS patch installed — FLUX/SD3.5 models should now work')
"

# Option B: Build C++ extension (slightly different perf characteristics)
pip install -e .
```

No Xcode required. The Metal shader compiles at runtime via `torch.mps.compile_shader()`.

## What This Solves

PyTorch's MPS backend (Apple's GPU acceleration) has a hard gap: **FP8 tensors exist but can't compute**. This blocks an entire class of modern AI models.

| What works | What doesn't (before this fix) | Impact |
|-----------|-------------|--------|
| Create FP8 tensors | Cast float32 → FP8 on MPS | Can't quantize on-device |
| Transfer FP8 CPU → MPS | `torch._scaled_mm` on MPS | **FLUX/SD3.5 won't run** |
| FP16/BF16/INT8 compute | Any FP8 arithmetic on MPS | ComfyUI crashes |

**Now with this patch:** ✅ All FP8 operations work on MPS, including `.to(torch.float8_e4m3fn)` conversions!

This repo provides 4 Metal GPU kernels:
1. **`fp8_scaled_matmul_kernel`** — General M×N FP8 matmul with 4-element unrolling
2. **`fp8_scaled_vecmat_kernel`** — SIMD-optimized single-token inference (M=1)
3. **`fp8_to_half_kernel`** — FP8 → FP16 dequantization
4. **`float_to_fp8_kernel`** — Float → FP8 quantization

And a monkey-patch that transparently intercepts:
- `torch._scaled_mm` for FP8 matrix multiplication
- `Tensor.to(torch.float8_e4m3fn)` for FP8 dtype conversions

So ComfyUI/diffusers/transformers work without code changes.

## Performance

Tested on M4 Pro (48GB, 20 GPU cores, Metal 4):

| Path | M=1, K=4096 | M=1, K=14336 | M=4, K=4096 |
|------|------------|-------------|-------------|
| **FP16 native** (baseline) | 0.20 ms | 1.64 ms | 0.13 ms |
| **Our fused kernel** | 0.66 ms | 2.38 ms | 1.03 ms |
| **CPU fallback** (what you have now) | 4.47 ms | 62.79 ms | 3.99 ms |
| **Speedup vs CPU** | **6.4x** | **26.3x** | **3.9x** |

The fused kernel is 4–26x faster than the CPU fallback path. At K=14336 (typical for large diffusion models), the improvement is dramatic.

**Accuracy:** 4% relative RMSE vs FP32 reference (expected for 8-bit quantization). Perfect decode across all 256 FP8 bit patterns.

---

# Why Things Don't Work on Mac (NVIDIA → Apple Silicon Guide)

If you're coming from NVIDIA where things "just work," here's what's actually going on and what to do about it.

## The Big Picture

NVIDIA has CUDA — a mature, 15-year ecosystem with native FP8/INT8/INT4 tensor cores, cuBLAS, cuDNN, and libraries for every conceivable operation. Apple has MPS (Metal Performance Shaders) — a much younger ML acceleration layer built on Metal, Apple's graphics API. MPS is fast for what it supports, but has real gaps.

**The honest truth:** Most things DO work on MPS. The gaps are specific and documented below. Knowing exactly what fails (and why) is more useful than vague "Mac doesn't support ML" claims.

## What Actually Works (and Works Well)

| Operation | MPS Status | Notes |
|-----------|-----------|-------|
| Float32/Float16 matmul | Full speed | 6.45 TFLOPS on M4 Pro |
| BFloat16 | Fully supported | Since PyTorch 2.10 |
| SDPA attention | Works | Including causal masks, GQA |
| Conv2D/3D | Works | All configurations |
| Group/Layer/Instance norm | Works | |
| Interpolate (all modes) | Works | nearest, bilinear, bicubic |
| Scatter/Gather/Embedding | Works | |
| FFT operations | Works | rfft2, irfft2, fft |
| TopK, multinomial | Works | |
| INT8 dequant + matmul | GPU-accelerated | 2.6x faster than CPU |

## What Fails (and Why)

### 1. FP8 (float8_e4m3fn / float8_e5m2) — SOLVED BY THIS REPO

**Symptom:** `RuntimeError: "mps" does not have support for that dtype`

**Why:** Metal Shading Language has no native 8-bit float type. PyTorch's MPS backend never implemented the cast or compute kernels for FP8. NVIDIA added FP8 tensor cores in Ada Lovelace (RTX 4000+) with hardware support; Apple has no equivalent.

**Who it affects:** Anyone running FLUX, SD3.5, or models quantized with FP8 (increasingly common since 2024).

**Fix:** This repo. Install the monkey-patch and FP8 models work on MPS.

```python
import fp8_mps_patch
fp8_mps_patch.install()
# Now load your FLUX/SD3.5 model normally
```

### 2. `torch._scaled_mm` not implemented — SOLVED BY THIS REPO

**Symptom:** `NotImplementedError: scaled_mm not implemented for MPS`

**Why:** `_scaled_mm` is PyTorch's internal API for quantized matmul with separate scale tensors. It's the foundation for FP8 inference in transformers and diffusion models. MPS simply doesn't have this kernel.

**Fix:** Same monkey-patch as above.

### 3. `quantize_per_tensor` / Native Quantized Tensors

**Symptom:** `RuntimeError: "QuantizedMPS" backend not implemented`

**Why:** PyTorch's legacy quantization API (QAT, PTQ with `torch.quantize_per_tensor`) uses a separate "Quantized" backend that was never ported to MPS. This is different from modern quantization approaches.

**Workaround:** Use "fake quantization" instead — store weights as INT8, dequantize to FP16 at matmul time. This actually works on MPS and is GPU-accelerated (2.6x faster than CPU).

```python
# Instead of: q = torch.quantize_per_tensor(x, scale, zero_point, torch.qint8)
# Do this:
scale = x.abs().max() / 127.0
x_int8 = (x / scale).round().clamp(-128, 127).to(torch.int8)
# At matmul time:
x_fp16 = x_int8.half() * scale.half()
result = x_fp16 @ weights
```

### 4. `torch.linalg.eigh` / `torch.linalg.qr`

**Symptom:** `RuntimeError: not implemented for MPS`

**Why:** Eigenvalue decomposition and QR factorization require iterative algorithms (Householder, Jacobi) that are hard to parallelize efficiently on GPU. NVIDIA has cuSOLVER; Apple hasn't implemented Metal equivalents in PyTorch.

**Workaround:** Move to CPU for these operations (they're usually not in the hot path):
```python
eigenvalues, eigenvectors = torch.linalg.eigh(matrix.cpu())
eigenvalues = eigenvalues.to("mps")
eigenvectors = eigenvectors.to("mps")
```

### 5. `torch.linalg.svd` — Silent CPU Fallback

**Symptom:** Works but prints a warning and runs on CPU.

**Impact:** Slow but not broken. If SVD is in your training loop, it's a bottleneck.

### 6. Small Matrix Operations Are Slower Than CPU

**Symptom:** Model with many small layers (< 512x512) is slower on MPS than CPU.

**Why:** Every MPS kernel dispatch has overhead (~10% more than CPU). For a 128x128 matmul, CPU (using AMX accelerator) is **13x faster** than MPS. The crossover is around 512x512.

**Fix:** For models with small layers, use `device="cpu"`. For mixed architectures, keep large matmuls on MPS and small ops on CPU. Or batch small ops together.

| Matrix Size | CPU | MPS | Winner |
|-------------|-----|-----|--------|
| 32×32 | 0.025ms | 2.85ms | CPU (114x) |
| 128×128 | 0.007ms | 0.09ms | CPU (13x) |
| 512×512 | 0.45ms | 0.25ms | MPS (1.8x) |
| 2048×2048 | 5.68ms | 3.16ms | MPS (1.8x) |
| 4096×4096 | — | — | MPS (dominant) |

### 7. FP16 Barely Faster Than FP32

**Symptom:** Switching to FP16 doesn't give the 2x speedup you expect from NVIDIA.

**Why:** Apple's M-series chips use ALUs that handle both FP32 and FP16 similarly (unlike NVIDIA's separate FP16 tensor cores). On M4 Pro, FP16 matmul is only ~1.1x faster than FP32.

**What to do:** Use FP16 for **memory savings** (half the VRAM), not for compute speed. This is the opposite of NVIDIA where FP16 gives both.

### 8. Batching / Multi-Stream Doesn't Help

**Symptom:** Using `torch.bmm` or multiple streams doesn't speed things up.

**Why:** NVIDIA GPUs have hardware schedulers that run multiple kernels concurrently. MPS serializes everything through a single command queue. `bmm` on MPS is only ~1.04x faster than sequential matmuls.

**What to do:** Focus on making individual operations larger rather than parallelizing small ones. Pre-merge LoRA weights. Fuse operations where possible.

## Memory: What You Actually Get

| Config | Usable GPU Memory |
|--------|------------------|
| Default PyTorch | 24 GB (of 48 GB unified) |
| `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` | 28 GB |
| Per-buffer hard cap | 32 GB (Metal limit, always fails above this) |
| Total across multiple buffers | **Full 48 GB** (via sharding) |

**Key insight:** You can access all 48GB by using multiple buffers under 32GB each. Model parallelism / layer-by-layer loading works.

```bash
# Put this in your shell profile
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

| Model | FP32 | FP16 | INT8 | INT4 | Fits in 28GB? |
|-------|------|------|------|------|--------------|
| SD 1.5 (860M) | 3.4 GB | 1.7 GB | 860 MB | 430 MB | All |
| SDXL (3.5B) | 14 GB | 7 GB | 3.5 GB | 1.75 GB | All |
| FLUX (12B) | 48 GB | 24 GB | 12 GB | 6 GB | FP16 tight, INT8/4 yes |
| Llama-3 70B | 280 GB | 140 GB | 70 GB | 35 GB | INT4 only (with offload) |

## PyTorch MPS Best Practices

```python
import os
import torch

# Environment (set before importing torch)
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"  # More GPU memory
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"         # CPU fallback for unsupported ops

# Device selection
device = "mps" if torch.backends.mps.is_available() else "cpu"

# For training: use float32 (more stable on MPS)
model = model.to(device=device, dtype=torch.float32)

# For inference: float16 saves memory (speed similar to float32)
model = model.to(device=device, dtype=torch.float16)

# Attention: use SDPA (MPS-compatible)
# In HuggingFace: attn_implementation="sdpa"

# Synchronize before timing
torch.mps.synchronize()
start = time.perf_counter()
output = model(input)
torch.mps.synchronize()
elapsed = time.perf_counter() - start
```

## LoRA: Pre-Merge Your Weights

On NVIDIA, keeping LoRA adapters separate (base + A @ B) is fine because kernel launch is cheap. On MPS, **pre-merging is 1.67x faster** because dispatch overhead is higher.

```python
# WRONG on MPS (3 kernel launches):
out = F.linear(x, base_weight) + F.linear(F.linear(x, lora_A), lora_B) * scale

# RIGHT on MPS (1 kernel launch):
merged_weight = base_weight + (lora_B @ lora_A) * scale
out = F.linear(x, merged_weight)
```

LoRA hot-swap (re-merging on adapter change): <40ms for a 7B model. Negligible.

## INT8 vs INT4 Quantization

On NVIDIA, INT4 (GPTQ, AWQ) is standard for memory savings with good speed. On MPS, **INT4 is 1.9x slower than INT8** due to bit-unpacking overhead (no native 4-bit support in Metal).

| Quantization | MPS Speed (1x4096 @ 4096x4096) | Recommendation |
|-------------|-------------------------------|----------------|
| FP16 | 0.91ms (baseline) | Best speed |
| INT8 | 2.21ms (2.4x overhead) | Best speed/memory tradeoff |
| INT4 | 4.13ms (4.5x overhead) | Only if you need the memory |

**Recommendation:** Use INT8 on MPS unless you absolutely need INT4 for memory. This is the opposite of NVIDIA where INT4 is preferred.

## ComfyUI on Mac

```bash
# Install ComfyUI
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI
pip install -r requirements.txt

# Required environment variables
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export PYTORCH_ENABLE_MPS_FALLBACK=1

# For FP8 models (FLUX, SD3.5):
# Install fp8-mps-metal and add to ComfyUI's startup
pip install -e /path/to/fp8-mps-metal
# Add to ComfyUI/main.py or a custom node:
import fp8_mps_patch; fp8_mps_patch.install()

# Run with memory optimizations
python main.py --force-fp16 --use-split-cross-attention
```

---

## Architecture

```
fp8_matmul.metal      Metal GPU kernels (FP8 bit-unpacking + matmul)
fp8_mps_native.py     Zero-copy API using torch.mps.compile_shader()  ← RECOMMENDED
fp8_mps_patch.py      Monkey-patch for torch._scaled_mm
fp8_bridge.cpp        C++ extension (alternative, uses metal-cpp + pybind11)
setup.py              Build system for C++ extension
test_fp8_metal.py     Full test suite (accuracy + performance + integration)
```

### How FP8 Decoding Works

Metal has no native FP8 type. We store FP8 as `uint8_t` and decode in-register:

```
e4m3fn format: [sign:1][exponent:4][mantissa:3]
Bias = 7, max value = 448.0, NaN = 0x7F/0xFF

Decode:
  sign = (bits >> 7) & 1
  exp  = (bits >> 3) & 0xF
  mant = bits & 0x7

  Normal:    value = (-1)^sign * 2^(exp-7) * (1 + mant/8)
  Subnormal: value = (-1)^sign * 2^(-6) * (mant/8)
  NaN:       value = 0  (we flush NaN to zero)
```

The vecmat kernel (M=1 inference path) uses Metal's hardware SIMD reduction (`simd_sum`) across 32 lanes for the K-dimension dot product, matching the pattern from [metalQwen3](https://github.com/Architect2040/metalQwen3).

## Running Tests

```bash
python test_fp8_metal.py
```

Expected output:
```
Test 1: Exhaustive FP8 decode (256 patterns)    PASS
Test 2: Matmul accuracy — C++ extension          PASS  (4.0% RMSE)
Test 3: Matmul accuracy — Native (fused + fast)  PASS  (4.0% RMSE)
Test 4: Quantize/dequantize roundtrip            PASS
Test 5: Vecmat (M=1)                             PASS
Test 6: Performance benchmarks                   REPORTED
Test 7: Monkey-patch install/uninstall           PASS
```

## Hardware Benchmarks (M4 Pro, 48GB)

These numbers are from our validated test suite. Your results will vary by chip.

| Metric | Value |
|--------|-------|
| Peak GEMM (FP32, 4096x4096) | 6.45 TFLOPS |
| Memory bandwidth (1GB copy) | 197 GB/s |
| SDPA attention (seq=2048, 32h, d=128) | 31 ms |
| SwiGLU FFN (hidden=4096) | 21.5 ms |
| RMSNorm (batch=1, seq=2048) | 2.2 ms |
| KV-cache per step (cache=256) | 0.25 ms |
| Safetensors loading throughput | ~7 GB/s |

## System Requirements

- macOS 15+ (Sequoia) or macOS 26+ (tested)
- Apple Silicon (M1/M2/M3/M4, any variant)
- PyTorch 2.4+ (for `torch._scaled_mm`), 2.10+ recommended (for `torch.mps.compile_shader`)
- Python 3.10+
- **No Xcode required** (runtime shader compilation)

## Related Resources

- [metalQwen3](https://github.com/Architect2040/metalQwen3) — Custom Metal shaders for Qwen3 transformer inference
- [metal-ml-shaders-experiments](https://github.com/philipturner/metal-ml-shaders-experiments) — Outperforming MPS with custom Metal
- [VectorAccelerate](https://github.com/nickthorpe71/VectorAccelerate) — Metal 4 SDK tiled matmul
- [mpsparse](https://github.com/roberto-nai/mpsparse) — Sparse matrix ops on Metal
- [jax-mps](https://github.com/nicholasgasior/jax-mps) — JAX on Apple Silicon via StableHLO→MPSGraph
- [ComfyUI Mac Install Guide](https://github.com/comfyanonymous/ComfyUI/wiki/Installing-ComfyUI-on-macOS)

## License

MIT

## Testing

### Running Tests Locally

**On macOS with Apple Silicon (M1/M2/M3/M4):**
```bash
# Full test suite with MPS Metal kernels
python test_fp8_metal.py

# MPS vs CPU validation (recommended for verifying correctness)
python test_mps_vs_cpu.py

# FP8 spec correctness validation
python test_fp8_correctness.py

# Cross-implementation validation (Metal vs C++)
python test_cross_validation.py
```

**Test the patch without MPS hardware:**
```bash
# Quick inline test
python -c "
import torch
import fp8_mps_patch

# Test patch mechanism
fp8_mps_patch.install()
assert fp8_mps_patch.is_installed()
print('✓ Patch installed')

# Test that normal operations still work
x = torch.randn(4, 4)
assert x.to(torch.float16).dtype == torch.float16
print('✓ Normal operations unaffected')

fp8_mps_patch.uninstall()
print('✓ All tests passed')
"
```

**Recommended testing workflow for validating conversions:**
```bash
# 1. Verify spec compliance (can run on any system)
python test_fp8_correctness.py

# 2. Validate MPS vs CPU consistency (requires Apple Silicon)
python test_mps_vs_cpu.py

# 3. Run full integration tests (requires Apple Silicon)
python test_fp8_metal.py
```

### Example Usage

See `example_fp8_conversion.py` for practical examples of using Float8_e4m3fn conversions on MPS.

### Continuous Integration

This repository includes GitHub Actions workflows that automatically test:

- **macOS**: Patch mechanism, FP8 conversion support
- **Linux**: CPU-only validation to ensure no regressions
- **Linting**: Syntax checking and code quality

See `.github/workflows/test-mps.yml` for details.

**Note:** GitHub Actions hosted runners do not have Apple Silicon, so full MPS Metal kernel tests must be run locally on compatible hardware.

