"""
MPS Pattern Validation Suite for M4 Pro (48GB)
Tests key patterns discovered across 40 MPS repositories
"""
import torch
import time
import os
import gc

def header(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def timed(fn, warmup=2, runs=5):
    """Run function with warmup, return median time in ms"""
    for _ in range(warmup):
        fn()
        torch.mps.synchronize()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        torch.mps.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return times[len(times)//2]

# ============================================================
header("1. BASIC MPS DEVICE INFO")
# ============================================================
print(f"PyTorch version: {torch.__version__}")
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")

# Check current memory
x = torch.rand(1, device="mps")
print(f"MPS device: {x.device}")
del x

# ============================================================
header("2. DTYPE SUPPORT ON MPS")
# ============================================================
dtypes_to_test = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "bool": torch.bool,
    "complex64": torch.complex64,
}

for name, dtype in dtypes_to_test.items():
    try:
        t = torch.zeros(4, 4, dtype=dtype, device="mps")
        # Try a basic operation
        if dtype in (torch.float32, torch.float16, torch.bfloat16):
            r = t + 1.0
            r = torch.matmul(torch.ones(4,4,dtype=dtype,device="mps"),
                           torch.ones(4,4,dtype=dtype,device="mps"))
            print(f"  {name:12s}: OK (creation + matmul)")
        elif dtype == torch.complex64:
            r = t + 1.0
            print(f"  {name:12s}: OK (creation + add)")
        else:
            r = t + 1
            print(f"  {name:12s}: OK (creation + add)")
        del t, r
    except Exception as e:
        print(f"  {name:12s}: FAIL - {e}")

# ============================================================
header("3. MATMUL PERFORMANCE: MPS vs CPU")
# ============================================================
sizes = [(512, 512), (1024, 1024), (2048, 2048), (4096, 4096)]

for M, N in sizes:
    K = M
    a_cpu = torch.randn(M, K)
    b_cpu = torch.randn(K, N)
    a_mps = a_cpu.to("mps")
    b_mps = b_cpu.to("mps")

    cpu_ms = timed(lambda: torch.matmul(a_cpu, b_cpu), warmup=1, runs=3)
    mps_ms = timed(lambda: torch.matmul(a_mps, b_mps), warmup=2, runs=5)
    speedup = cpu_ms / mps_ms if mps_ms > 0 else 0

    # Estimate TFLOPS
    flops = 2 * M * N * K
    mps_tflops = flops / (mps_ms / 1000) / 1e12

    print(f"  {M}x{N}: CPU={cpu_ms:.1f}ms  MPS={mps_ms:.2f}ms  "
          f"Speedup={speedup:.1f}x  MPS={mps_tflops:.2f} TFLOPS")

    del a_cpu, b_cpu, a_mps, b_mps

# ============================================================
header("4. FLOAT16 vs FLOAT32 MATMUL")
# ============================================================
M, N, K = 2048, 2048, 2048
a32 = torch.randn(M, K, device="mps", dtype=torch.float32)
b32 = torch.randn(K, N, device="mps", dtype=torch.float32)
a16 = a32.half()
b16 = b32.half()

try:
    abf16 = a32.bfloat16()
    bbf16 = b32.bfloat16()
    has_bf16 = True
except:
    has_bf16 = False

ms_f32 = timed(lambda: torch.matmul(a32, b32))
ms_f16 = timed(lambda: torch.matmul(a16, b16))
print(f"  float32: {ms_f32:.2f}ms")
print(f"  float16: {ms_f16:.2f}ms  ({ms_f32/ms_f16:.1f}x faster)")

if has_bf16:
    try:
        ms_bf16 = timed(lambda: torch.matmul(abf16, bbf16))
        print(f"  bfloat16: {ms_bf16:.2f}ms  ({ms_f32/ms_bf16:.1f}x faster)")
    except Exception as e:
        print(f"  bfloat16 matmul: FAIL - {e}")

del a32, b32, a16, b16

# ============================================================
header("5. SDPA ATTENTION (Scaled Dot-Product)")
# ============================================================
# Test F.scaled_dot_product_attention on MPS
import torch.nn.functional as F

batch, heads, seq_len, d_head = 1, 32, 512, 128

for seq in [128, 512, 1024, 2048]:
    q = torch.randn(batch, heads, seq, d_head, device="mps", dtype=torch.float32)
    k = torch.randn(batch, heads, seq, d_head, device="mps", dtype=torch.float32)
    v = torch.randn(batch, heads, seq, d_head, device="mps", dtype=torch.float32)

    try:
        ms = timed(lambda: F.scaled_dot_product_attention(q, k, v))
        print(f"  seq={seq:5d}, heads={heads}, d={d_head}: {ms:.2f}ms  OK")
    except Exception as e:
        print(f"  seq={seq:5d}: FAIL - {e}")

    del q, k, v

# Also test with float16
print("\n  Float16 SDPA:")
for seq in [512, 2048]:
    q = torch.randn(batch, heads, seq, d_head, device="mps", dtype=torch.float16)
    k = torch.randn(batch, heads, seq, d_head, device="mps", dtype=torch.float16)
    v = torch.randn(batch, heads, seq, d_head, device="mps", dtype=torch.float16)
    try:
        ms = timed(lambda: F.scaled_dot_product_attention(q, k, v))
        print(f"  seq={seq:5d}, float16: {ms:.2f}ms  OK")
    except Exception as e:
        print(f"  seq={seq:5d}, float16: FAIL - {e}")
    del q, k, v

# ============================================================
header("6. MEMORY ALLOCATION LIMITS")
# ============================================================
gc.collect()
torch.mps.empty_cache()

# Test how much we can allocate
alloc_sizes_gb = [1, 2, 4, 8, 16, 24, 32, 40]
for gb in alloc_sizes_gb:
    try:
        n = int(gb * 1024**3 / 4)  # float32 = 4 bytes
        t = torch.zeros(n, device="mps", dtype=torch.float32)
        # Touch it
        t[0] = 1.0
        torch.mps.synchronize()
        actual_gb = t.element_size() * t.nelement() / 1024**3
        print(f"  {gb:2d}GB allocation: OK ({actual_gb:.1f}GB actual)")
        del t
        torch.mps.empty_cache()
    except Exception as e:
        print(f"  {gb:2d}GB allocation: FAIL - {str(e)[:60]}")
        break

# ============================================================
header("7. WATERMARK RATIO EFFECT")
# ============================================================
print("  Current PYTORCH_MPS_HIGH_WATERMARK_RATIO:",
      os.environ.get("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "default (not set)"))
print("  Current PYTORCH_ENABLE_MPS_FALLBACK:",
      os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "default (not set)"))

# ============================================================
header("8. TRANSFORMER BUILDING BLOCKS")
# ============================================================

# Test common transformer ops
print("\n  RMSNorm:")
hidden = 4096
for bs_seq in [(1, 128), (1, 512), (1, 2048), (4, 512)]:
    bs, seq = bs_seq
    x = torch.randn(bs, seq, hidden, device="mps")
    weight = torch.ones(hidden, device="mps")

    def rmsnorm(x, w, eps=1e-6):
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + eps) * w

    ms = timed(lambda: rmsnorm(x, weight))
    print(f"    batch={bs}, seq={seq}, hidden={hidden}: {ms:.2f}ms")
    del x, weight

print("\n  SiLU + Gate (SwiGLU FFN pattern):")
for hidden in [2048, 4096]:
    ffn_dim = int(hidden * 8/3)  # Qwen3-style
    x = torch.randn(1, 512, hidden, device="mps")
    w_gate = torch.randn(hidden, ffn_dim, device="mps")
    w_up = torch.randn(hidden, ffn_dim, device="mps")
    w_down = torch.randn(ffn_dim, hidden, device="mps")

    def swiglu_ffn(x, wg, wu, wd):
        gate = torch.nn.functional.silu(x @ wg)
        up = x @ wu
        return (gate * up) @ wd

    ms = timed(lambda: swiglu_ffn(x, w_gate, w_up, w_down))
    print(f"    hidden={hidden}, ffn={ffn_dim}: {ms:.2f}ms")
    del x, w_gate, w_up, w_down

print("\n  RoPE (Rotary Position Embeddings):")
for seq in [128, 512, 2048]:
    d_head = 128
    n_heads = 32
    x = torch.randn(1, n_heads, seq, d_head, device="mps")

    # Build rotation matrix
    freqs = 1.0 / (10000.0 ** (torch.arange(0, d_head, 2, device="mps").float() / d_head))
    t = torch.arange(seq, device="mps").float()
    freqs = torch.outer(t, freqs)
    cos_cached = torch.cos(freqs).unsqueeze(0).unsqueeze(0)
    sin_cached = torch.sin(freqs).unsqueeze(0).unsqueeze(0)

    def apply_rope(x, cos, sin):
        # Standard complex-number RoPE
        x_reshape = x.float().reshape(*x.shape[:-1], -1, 2)
        x1 = x_reshape[..., 0]
        x2 = x_reshape[..., 1]
        out1 = x1 * cos - x2 * sin
        out2 = x1 * sin + x2 * cos
        return torch.stack([out1, out2], dim=-1).flatten(-2).to(x.dtype)

    ms = timed(lambda: apply_rope(x, cos_cached, sin_cached))
    print(f"    seq={seq}, heads={n_heads}, d={d_head}: {ms:.2f}ms")
    del x, cos_cached, sin_cached

# ============================================================
header("9. QUANTIZATION SUPPORT")
# ============================================================

# Test int8 matmul patterns
print("\n  INT8 simulation (dequant → float16 matmul):")
M, K, N = 4096, 4096, 4096
w_int8 = torch.randint(-128, 127, (K, N), dtype=torch.int8, device="mps")
scale = torch.randn(N, device="mps", dtype=torch.float16) * 0.01
x_f16 = torch.randn(1, M, K, device="mps", dtype=torch.float16)

def dequant_matmul(x, w, s):
    w_f16 = w.to(torch.float16)
    w_scaled = w_f16 * s.unsqueeze(0)
    return x @ w_scaled

try:
    ms = timed(lambda: dequant_matmul(x_f16, w_int8, scale), warmup=2, runs=3)
    print(f"    4096x4096 INT8→FP16 matmul: {ms:.2f}ms")
except Exception as e:
    print(f"    INT8→FP16 matmul: FAIL - {e}")

del w_int8, scale, x_f16

# Test per-group quantization (from metalQwen3 patterns)
print("\n  Per-group INT8 quantization (group_size=128):")
group_size = 128
M, K, N = 1, 4096, 4096
n_groups = K // group_size
w_int8 = torch.randint(-128, 127, (K, N), dtype=torch.int8, device="mps")
scales = torch.randn(n_groups, N, device="mps", dtype=torch.float16) * 0.01
x_f16 = torch.randn(M, K, device="mps", dtype=torch.float16)

def grouped_dequant_matmul(x, w, s, gs):
    w_f16 = w.to(torch.float16)
    # Apply per-group scaling
    scales_expanded = s.repeat_interleave(gs, dim=0)
    w_scaled = w_f16 * scales_expanded
    return x @ w_scaled

try:
    ms = timed(lambda: grouped_dequant_matmul(x_f16, w_int8, scales, group_size), warmup=2, runs=3)
    print(f"    4096x4096 grouped INT8→FP16: {ms:.2f}ms")
except Exception as e:
    print(f"    Grouped INT8→FP16: FAIL - {e}")

del w_int8, scales, x_f16

# ============================================================
header("10. MPS-SPECIFIC QUIRKS & FALLBACKS")
# ============================================================

# Test operations known to have MPS issues
tests = {
    "torch.linalg.svd": lambda: torch.linalg.svd(torch.randn(64, 64, device="mps")),
    "torch.linalg.eigh": lambda: torch.linalg.eigh(torch.randn(64, 64, device="mps").T @ torch.randn(64, 64, device="mps")),
    "torch.searchsorted": lambda: torch.searchsorted(torch.sort(torch.rand(100, device="mps"))[0], torch.rand(10, device="mps")),
    "torch.multinomial": lambda: torch.multinomial(torch.softmax(torch.randn(1, 100, device="mps"), dim=-1), 5),
    "torch.topk": lambda: torch.topk(torch.randn(1, 50000, device="mps"), 50),
    "torch.scatter_add": lambda: torch.zeros(10, device="mps").scatter_add_(0, torch.randint(0, 10, (100,), device="mps"), torch.randn(100, device="mps")),
    "F.cross_entropy": lambda: F.cross_entropy(torch.randn(4, 100, device="mps"), torch.randint(0, 100, (4,), device="mps")),
    "F.layer_norm": lambda: F.layer_norm(torch.randn(2, 128, 768, device="mps"), [768]),
    "F.group_norm": lambda: F.group_norm(torch.randn(2, 32, 64, 64, device="mps"), 8),
    "conv2d": lambda: F.conv2d(torch.randn(1, 3, 256, 256, device="mps"), torch.randn(64, 3, 3, 3, device="mps")),
    "F.interpolate_bilinear": lambda: F.interpolate(torch.randn(1, 3, 64, 64, device="mps"), scale_factor=4, mode='bilinear'),
    "einsum_attention": lambda: torch.einsum('bhsd,bhsd->bhs', torch.randn(1,32,512,128,device="mps"), torch.randn(1,32,512,128,device="mps")),
}

for name, fn in tests.items():
    try:
        result = fn()
        torch.mps.synchronize()
        print(f"  {name:30s}: OK")
        del result
    except Exception as e:
        print(f"  {name:30s}: FAIL - {str(e)[:50]}")

# ============================================================
header("11. CONCURRENT MPS STREAM BEHAVIOR")
# ============================================================
# Test if MPS benefits from concurrent operations
M = 2048

a = torch.randn(M, M, device="mps")
b = torch.randn(M, M, device="mps")
c = torch.randn(M, M, device="mps")
d = torch.randn(M, M, device="mps")

# Sequential
def sequential():
    r1 = torch.matmul(a, b)
    r2 = torch.matmul(c, d)
    return r1, r2

seq_ms = timed(sequential)

# "Batched" - single larger matmul
ab = torch.stack([a, c])
cd = torch.stack([b, d])
def batched():
    return torch.bmm(ab, cd)

batch_ms = timed(batched)
print(f"  2x {M}x{M} matmul sequential: {seq_ms:.2f}ms")
print(f"  2x {M}x{M} matmul batched:    {batch_ms:.2f}ms")
print(f"  Batching benefit: {seq_ms/batch_ms:.2f}x")

del a, b, c, d, ab, cd

# ============================================================
header("12. MEMORY BANDWIDTH TEST")
# ============================================================
# Measure effective memory bandwidth
for size_mb in [64, 256, 1024]:
    n = size_mb * 1024 * 1024 // 4  # float32
    src = torch.randn(n, device="mps")

    def copy_test():
        return src.clone()

    ms = timed(copy_test, warmup=3, runs=5)
    bandwidth_gbps = (size_mb * 2 / 1024) / (ms / 1000)  # read + write
    print(f"  {size_mb:5d}MB copy: {ms:.2f}ms  ({bandwidth_gbps:.1f} GB/s effective)")
    del src

print(f"\n{'='*60}")
print(f"  ALL TESTS COMPLETE")
print(f"{'='*60}")
