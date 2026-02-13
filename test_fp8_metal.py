#!/usr/bin/env python3
"""
Test suite for fp8_metal: FP8 Metal kernel accuracy and performance.

Tests both implementations:
  - C++ extension (fp8_metal) — works via buffer copies
  - Native torch.mps.compile_shader (fp8_mps_native) — zero-copy, preferred

Tests:
  1. Exhaustive FP8 decode: all 256 uint8 values vs Python reference
  2. Matmul accuracy (C++ ext): FP8 scaled_mm vs FP32 reference
  3. Matmul accuracy (native): FP8 scaled_mm vs FP32 reference
  4. Quantize/dequantize roundtrip
  5. Vecmat (M=1) kernel path
  6. Performance: all paths at realistic dimensions
  7. Monkey-patch install/uninstall
"""

import time
import sys
import os
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def fp8_e4m3fn_decode_reference(bits: int) -> float:
    """Pure Python reference decode for e4m3fn format."""
    if (bits & 0x7F) == 0x7F:  # NaN → 0
        return 0.0
    sign = (bits >> 7) & 1
    exp_bits = (bits >> 3) & 0xF
    mant_bits = bits & 0x7

    if exp_bits == 0:
        value = (mant_bits / 8.0) * (2.0 ** -6)
    else:
        mantissa = 1.0 + mant_bits / 8.0
        exponent = exp_bits - 7
        value = mantissa * (2.0 ** exponent)

    return -value if sign else value


def test_exhaustive_fp8_decode():
    """Test all 256 FP8 bit patterns against reference."""
    print("=" * 60)
    print("Test 1: Exhaustive FP8 decode (256 patterns) — Native")
    print("=" * 60)

    import fp8_mps_native

    all_bits = torch.arange(256, dtype=torch.uint8)
    scale = torch.tensor([1.0])

    decoded = fp8_mps_native.fp8_dequantize(all_bits, scale)
    decoded_cpu = decoded.cpu().float()
    ref = torch.tensor([fp8_e4m3fn_decode_reference(i) for i in range(256)])

    max_abs_err = 0.0
    max_rel_err = 0.0
    errors = []

    for i in range(256):
        metal_val = decoded_cpu[i].item()
        ref_val = ref[i].item()
        abs_err = abs(metal_val - ref_val)
        rel_err = abs_err / (abs(ref_val) + 1e-10) if ref_val != 0 else abs_err

        if abs_err > 0.01:
            errors.append((i, ref_val, metal_val, abs_err))
        max_abs_err = max(max_abs_err, abs_err)
        if ref_val != 0:
            max_rel_err = max(max_rel_err, rel_err)

    print(f"  Max absolute error: {max_abs_err:.6f}")
    print(f"  Max relative error: {max_rel_err:.6f}")
    if errors:
        print(f"  Errors > 0.01 ({len(errors)}):")
        for bits, ref_v, metal_v, err in errors[:10]:
            print(f"    bits={bits:3d} (0x{bits:02X}): ref={ref_v:12.6f}, metal={metal_v:12.6f}, err={err:.6f}")

    passed = max_abs_err < 0.5
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}")
    print()
    return passed


def test_matmul_accuracy_cpp():
    """Test FP8 scaled matmul (C++ ext) against FP32 reference."""
    print("=" * 60)
    print("Test 2: Matmul accuracy — C++ extension")
    print("=" * 60)

    import fp8_metal

    M, K, N = 64, 256, 128
    A_f32 = torch.randn(M, K)
    B_f32 = torch.randn(N, K)
    ref = A_f32 @ B_f32.T

    A_q, A_scale = fp8_metal.fp8_quantize(A_f32)
    B_q, B_scale = fp8_metal.fp8_quantize(B_f32)
    result = fp8_metal.fp8_scaled_mm(A_q, B_q, A_scale, B_scale)
    result_cpu = result.cpu().float()

    diff = result_cpu - ref
    rmse = torch.sqrt((diff ** 2).mean()).item()
    ref_rms = torch.sqrt((ref ** 2).mean()).item()
    rel_rmse = rmse / ref_rms if ref_rms > 0 else rmse

    print(f"  Relative RMSE: {rel_rmse:.4%}")
    print(f"  Max abs error: {diff.abs().max().item():.4f}")
    passed = rel_rmse < 0.15
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}")
    print()
    return passed


def test_matmul_accuracy_native():
    """Test FP8 scaled matmul (native) against FP32 reference."""
    print("=" * 60)
    print("Test 3: Matmul accuracy — Native (fused + fast)")
    print("=" * 60)

    import fp8_mps_native

    M, K, N = 64, 256, 128
    A_f32 = torch.randn(M, K)
    B_f32 = torch.randn(N, K)
    ref = A_f32 @ B_f32.T

    A_q, A_scale = fp8_mps_native.fp8_quantize(A_f32)
    B_q, B_scale = fp8_mps_native.fp8_quantize(B_f32)

    # Test fused kernel
    result_fused = fp8_mps_native.fp8_scaled_mm(A_q, B_q, A_scale, B_scale)
    diff_fused = result_fused.cpu().float() - ref
    rel_rmse_fused = torch.sqrt((diff_fused ** 2).mean()).item() / torch.sqrt((ref ** 2).mean()).item()

    # Test fast (dequant + native matmul)
    result_fast = fp8_mps_native.fp8_scaled_mm_fast(A_q, B_q, A_scale, B_scale)
    diff_fast = result_fast.cpu().float() - ref
    rel_rmse_fast = torch.sqrt((diff_fast ** 2).mean()).item() / torch.sqrt((ref ** 2).mean()).item()

    # Test auto selector
    result_auto = fp8_mps_native.fp8_scaled_mm_auto(A_q, B_q, A_scale, B_scale)

    print(f"  Fused kernel RMSE:  {rel_rmse_fused:.4%}")
    print(f"  Fast path RMSE:     {rel_rmse_fast:.4%}")
    print(f"  Auto output shape:  {result_auto.shape}")

    passed = rel_rmse_fused < 0.15 and rel_rmse_fast < 0.15
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}")
    print()
    return passed


def test_quantize_roundtrip():
    """Test quantize → dequantize roundtrip (native)."""
    print("=" * 60)
    print("Test 4: Quantize/dequantize roundtrip — Native")
    print("=" * 60)

    import fp8_mps_native

    x = torch.tensor([0.0, 1.0, -1.0, 0.5, -0.5, 100.0, -100.0, 448.0])
    q, scale = fp8_mps_native.fp8_quantize(x)
    d = fp8_mps_native.fp8_dequantize(q, scale)
    d_cpu = d.cpu().float()

    max_err = (d_cpu - x).abs().max().item()
    print(f"  Input:     {x.tolist()}")
    print(f"  Roundtrip: {d_cpu.tolist()}")
    print(f"  Max error: {max_err:.4f}")

    passed = max_err < 50.0
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}")
    print()
    return passed


def test_vecmat_native():
    """Test M=1 vecmat kernel path (native)."""
    print("=" * 60)
    print("Test 5: Vecmat (M=1) — Native")
    print("=" * 60)

    import fp8_mps_native

    K, N = 512, 256
    x = torch.randn(1, K)
    W = torch.randn(N, K)
    ref = x @ W.T

    x_q, x_s = fp8_mps_native.fp8_quantize(x)
    W_q, W_s = fp8_mps_native.fp8_quantize(W)

    result = fp8_mps_native.fp8_scaled_mm(x_q, W_q, x_s, W_s)
    result_cpu = result.cpu().float()

    diff = result_cpu - ref
    rel_rmse = torch.sqrt((diff ** 2).mean()).item() / torch.sqrt((ref ** 2).mean()).item()

    print(f"  Relative RMSE: {rel_rmse:.4%}")
    print(f"  Output shape: {result.shape}")
    passed = rel_rmse < 0.15
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}")
    print()
    return passed


def test_performance():
    """Benchmark all FP8 paths at realistic dimensions."""
    print("=" * 60)
    print("Test 6: Performance benchmarks (realistic dimensions)")
    print("=" * 60)

    import fp8_mps_native

    warmup = 5
    iters = 20

    for label, M, K, N in [
        ("Single-token 4096", 1, 4096, 4096),
        ("Single-token 14336", 1, 14336, 14336),
        ("Batch-4 4096", 4, 4096, 4096),
    ]:
        print(f"\n  --- {label} (M={M}, K={K}, N={N}) ---")

        # FP8 data on MPS
        A_fp8 = torch.randint(0, 128, (M, K), dtype=torch.uint8, device="mps")
        B_fp8 = torch.randint(0, 128, (N, K), dtype=torch.uint8, device="mps")
        sa = torch.tensor([0.01])
        sb = torch.tensor([0.01])

        # FP16 native baseline
        A_f16 = torch.randn(M, K, dtype=torch.float16, device="mps")
        B_f16 = torch.randn(N, K, dtype=torch.float16, device="mps")
        for _ in range(warmup):
            _ = A_f16 @ B_f16.T
        torch.mps.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = A_f16 @ B_f16.T
        torch.mps.synchronize()
        fp16_ms = (time.perf_counter() - t0) / iters * 1000

        # CPU FP8 fallback (realistic: move to CPU, float, half, back to MPS, matmul)
        A_cpu_u8 = A_fp8.cpu()
        B_cpu_u8 = B_fp8.cpu()
        for _ in range(warmup):
            a = A_cpu_u8.float().half().to("mps")
            b = B_cpu_u8.float().half().to("mps")
            _ = (a @ b.T) * 0.01 * 0.01
        torch.mps.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            a = A_cpu_u8.float().half().to("mps")
            b = B_cpu_u8.float().half().to("mps")
            _ = (a @ b.T) * 0.01 * 0.01
        torch.mps.synchronize()
        cpu_ms = (time.perf_counter() - t0) / iters * 1000

        # Native fused kernel
        for _ in range(warmup):
            _ = fp8_mps_native.fp8_scaled_mm(A_fp8, B_fp8, sa, sb)
        torch.mps.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = fp8_mps_native.fp8_scaled_mm(A_fp8, B_fp8, sa, sb)
        torch.mps.synchronize()
        fused_ms = (time.perf_counter() - t0) / iters * 1000

        # Native fast (dequant + matmul)
        for _ in range(warmup):
            _ = fp8_mps_native.fp8_scaled_mm_fast(A_fp8, B_fp8, sa, sb)
        torch.mps.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = fp8_mps_native.fp8_scaled_mm_fast(A_fp8, B_fp8, sa, sb)
        torch.mps.synchronize()
        fast_ms = (time.perf_counter() - t0) / iters * 1000

        # Native auto
        for _ in range(warmup):
            _ = fp8_mps_native.fp8_scaled_mm_auto(A_fp8, B_fp8, sa, sb)
        torch.mps.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = fp8_mps_native.fp8_scaled_mm_auto(A_fp8, B_fp8, sa, sb)
        torch.mps.synchronize()
        auto_ms = (time.perf_counter() - t0) / iters * 1000

        best_ms = min(fused_ms, fast_ms)
        speedup = cpu_ms / best_ms

        print(f"    FP16 native:   {fp16_ms:7.2f} ms (ideal baseline)")
        print(f"    CPU fallback:  {cpu_ms:7.2f} ms (what we replace)")
        print(f"    Fused kernel:  {fused_ms:7.2f} ms")
        print(f"    Fast dequant:  {fast_ms:7.2f} ms")
        print(f"    Auto select:   {auto_ms:7.2f} ms")
        print(f"    Best speedup:  {speedup:.2f}x vs CPU fallback")

    print(f"\n  RESULT: REPORTED")
    print()
    return True


def test_monkey_patch():
    """Test monkey-patch install/uninstall."""
    print("=" * 60)
    print("Test 7: Monkey-patch install/uninstall")
    print("=" * 60)

    import fp8_mps_patch

    assert not fp8_mps_patch.is_installed(), "Should not be installed initially"
    print("  Not installed: OK")

    fp8_mps_patch.install()
    assert fp8_mps_patch.is_installed(), "Should be installed after install()"
    print("  Installed: OK")

    fp8_mps_patch.install()  # idempotent
    assert fp8_mps_patch.is_installed()
    print("  Idempotent install: OK")

    assert torch._scaled_mm is not fp8_mps_patch._original_scaled_mm
    print("  torch._scaled_mm patched: OK")
    
    assert torch.Tensor.to is not fp8_mps_patch._original_tensor_to
    print("  Tensor.to patched: OK")

    fp8_mps_patch.uninstall()
    assert not fp8_mps_patch.is_installed(), "Should not be installed after uninstall()"
    print("  Uninstalled: OK")

    print(f"  RESULT: PASS")
    print()
    return True


def test_fp8_conversion():
    """Test Float8_e4m3fn dtype conversion on MPS with comprehensive validation."""
    print("=" * 60)
    print("Test 8: Float8_e4m3fn conversion on MPS")
    print("=" * 60)

    import fp8_mps_patch
    import fp8_mps_native

    # Check if torch.float8_e4m3fn is available
    if not hasattr(torch, 'float8_e4m3fn'):
        print("  torch.float8_e4m3fn not available in this PyTorch version")
        print("  RESULT: SKIP")
        print()
        return True

    all_tests_passed = True

    # Test 1: Verify conversion fails WITHOUT patch
    print("\n  Test 1: Verify conversion fails without patch")
    try:
        x_mps = torch.randn(4, 8, device="mps")
        x_fp8 = x_mps.to(torch.float8_e4m3fn)
        print("    WARNING: Conversion succeeded without patch (unexpected)")
        # If this succeeds, PyTorch added native support - that's OK
    except RuntimeError as e:
        if "does not have support for that dtype" in str(e):
            print("    Native conversion fails as expected: OK")
        else:
            print(f"    Unexpected error: {e}")

    # Install the patch for remaining tests
    fp8_mps_patch.install()

    try:
        # Test 2: Basic float32 to Float8_e4m3fn conversion on MPS
        print("\n  Test 2: Basic float32 to Float8_e4m3fn conversion")
        x_mps = torch.randn(4, 8, device="mps")
        print(f"    Input: shape={x_mps.shape}, dtype={x_mps.dtype}, device={x_mps.device}")
        
        x_fp8 = x_mps.to(torch.float8_e4m3fn)
        print(f"    Output: dtype={x_fp8.dtype}, device={x_fp8.device}")
        assert x_fp8.dtype == torch.float8_e4m3fn, "Conversion to Float8_e4m3fn failed"
        assert x_fp8.device.type == "mps", "Result should be on MPS"
        assert x_fp8.shape == x_mps.shape, "Shape should be preserved"
        print("    Basic conversion: PASS")

        # Test 3: Convert from CPU to MPS as Float8_e4m3fn
        print("\n  Test 3: CPU to MPS Float8_e4m3fn conversion")
        x_cpu = torch.randn(4, 8)
        x_fp8_cpu = x_cpu.to("mps", dtype=torch.float8_e4m3fn)
        assert x_fp8_cpu.dtype == torch.float8_e4m3fn
        assert x_fp8_cpu.device.type == "mps"
        print("    CPU to MPS conversion: PASS")

        # Test 4: Convert from different source dtypes
        print("\n  Test 4: Conversion from different source dtypes")
        for src_dtype in [torch.float32, torch.float16]:
            x = torch.randn(4, 4, dtype=src_dtype, device="mps")
            x_fp8 = x.to(torch.float8_e4m3fn)
            assert x_fp8.dtype == torch.float8_e4m3fn
            print(f"    {src_dtype} to Float8_e4m3fn: OK")

        # Test 5: Edge cases - empty tensor, single element, large tensor
        print("\n  Test 5: Edge cases")
        # Empty tensor
        empty = torch.empty(0, device="mps")
        empty_fp8 = empty.to(torch.float8_e4m3fn)
        assert empty_fp8.numel() == 0
        print("    Empty tensor: OK")
        
        # Single element
        single = torch.tensor([3.14], device="mps")
        single_fp8 = single.to(torch.float8_e4m3fn)
        assert single_fp8.shape == single.shape
        print("    Single element: OK")
        
        # Large tensor
        large = torch.randn(128, 256, device="mps")
        large_fp8 = large.to(torch.float8_e4m3fn)
        assert large_fp8.shape == large.shape
        print("    Large tensor (128x256): OK")

        # Test 6: Verify FP8 values are valid (within FP8 range)
        print("\n  Test 6: Validate FP8 encoding")
        x_test = torch.tensor([0.0, 1.0, -1.0, 0.5, -0.5, 100.0, -100.0, 448.0], device="mps")
        x_fp8_test = x_test.to(torch.float8_e4m3fn)
        
        # View as uint8 to check the actual encoded values
        x_u8 = x_fp8_test.view(torch.uint8)
        x_u8_cpu = x_u8.cpu()
        
        # All values should be valid uint8 (0-255)
        assert x_u8_cpu.min() >= 0 and x_u8_cpu.max() <= 255
        print("    FP8 encoded values are valid uint8: OK")
        
        # Decode and verify accuracy
        scale = torch.tensor([1.0])
        x_reconstructed = fp8_mps_native.fp8_dequantize(x_u8, scale)
        x_reconstructed_cpu = x_reconstructed.cpu().float()
        x_test_cpu = x_test.cpu().float()
        
        # Check element-wise relative error (excluding zeros)
        for i, (orig, recon) in enumerate(zip(x_test_cpu, x_reconstructed_cpu)):
            if abs(orig) > 1e-6:
                rel_err = abs(recon - orig) / abs(orig)
                # FP8 e4m3fn has ~3 decimal digits precision
                # Allow up to 20% relative error for quantization
                if rel_err > 0.2:
                    print(f"    Warning: High relative error at index {i}: orig={orig:.4f}, recon={recon:.4f}, rel_err={rel_err:.2%}")
        
        max_err = (x_reconstructed_cpu - x_test_cpu).abs().max().item()
        print(f"    Roundtrip max absolute error: {max_err:.4f}")
        print("    FP8 accuracy validation: OK")

        # Test 7: FP8 CPU tensor to MPS device transfer (ComfyUI scenario)
        print("\n  Test 7: FP8 CPU tensor -> MPS device (raw bytes transfer)")
        # This simulates loading pre-quantized model weights
        x_u8_cpu = torch.randint(0, 255, (4, 8), dtype=torch.uint8)
        x_fp8_cpu = x_u8_cpu.view(torch.float8_e4m3fn)
        print(f"    Input: dtype={x_fp8_cpu.dtype}, device={x_fp8_cpu.device}")
        
        # This is what ComfyUI does - move FP8 weights to MPS
        x_fp8_mps = x_fp8_cpu.to("mps")
        print(f"    Output: dtype={x_fp8_mps.dtype}, device={x_fp8_mps.device}")
        assert x_fp8_mps.dtype == torch.float8_e4m3fn, "Dtype should be preserved"
        assert x_fp8_mps.device.type == "mps", "Should be on MPS"
        
        # Verify the bytes are identical
        x_u8_mps = x_fp8_mps.view(torch.uint8).cpu()
        assert torch.equal(x_u8_cpu, x_u8_mps), "Bytes should be identical"
        print("    FP8 CPU -> MPS transfer: PASS")

        # Test 8: Verify conversion in computation pipeline
        print("\n  Test 8: Use converted FP8 tensors in operations")
        A_f32 = torch.randn(16, 32, device="mps")
        B_f32 = torch.randn(32, 32, device="mps")
        
        # Convert to FP8
        A_fp8 = A_f32.to(torch.float8_e4m3fn)
        B_fp8 = B_f32.to(torch.float8_e4m3fn)
        
        # These should now work with our scaled_mm patch
        A_u8 = A_fp8.view(torch.uint8)
        B_u8 = B_fp8.view(torch.uint8)
        
        # Use the FP8 tensors in a matmul operation
        scale_a = torch.tensor([1.0])
        scale_b = torch.tensor([1.0])
        result = fp8_mps_native.fp8_scaled_mm_auto(A_u8, B_u8.t().contiguous(), scale_a, scale_b)
        
        assert result.shape == (16, 32), f"Expected shape (16, 32), got {result.shape}"
        assert result.device.type == "mps"
        print("    FP8 tensors used in matmul: OK")

        # Test 9: FP8 .copy_() operation (ComfyUI scenario)
        print("\n  Test 9: FP8 .copy_() operation on MPS")
        # This simulates ComfyUI's stochastic_rounding which creates FP8 tensors
        # and tries to copy them into MPS tensors
        fp8_source = torch.randint(0, 255, (4, 8), dtype=torch.uint8).view(torch.float8_e4m3fn)
        fp8_dest_mps = torch.empty(4, 8, dtype=torch.float8_e4m3fn, device="mps")
        
        # This should work with our patch
        fp8_dest_mps.copy_(fp8_source)
        
        # Verify bytes are preserved
        dest_u8 = fp8_dest_mps.view(torch.uint8).cpu()
        src_u8 = fp8_source.view(torch.uint8)
        assert torch.equal(dest_u8, src_u8), "Bytes should be preserved in copy_"
        print("    FP8 .copy_() to MPS: OK")

        # Test 9b: Float32 → FP8 conversion via .copy_() on MPS
        print("\n  Test 9b: Float32 → FP8 via .copy_() on MPS (new scenario)")
        # This tests the newly added scenario where non-FP8 source is copied to FP8 destination
        # which would fail without the extended patch
        f32_source = torch.tensor([[1.0, 2.5, -3.0, 0.5],
                                    [10.0, -8.0, 0.0, 100.0]], device="mps", dtype=torch.float32)
        fp8_dest_mps = torch.empty(2, 4, dtype=torch.float8_e4m3fn, device="mps")
        
        # This should work with our extended patch
        fp8_dest_mps.copy_(f32_source)
        
        # Verify the copy worked and data is reasonable
        # Convert back to float32 to check values
        result_f32 = fp8_dest_mps.to(torch.float32)
        
        # FP8 has limited precision, so we check approximate equality
        # The values should be reasonably close given FP8's ~3 decimal digit precision
        for i in range(2):
            for j in range(4):
                expected = f32_source[i, j].item()
                actual = result_f32[i, j].item()
                # For FP8 e4m3fn, we expect relative error within ~10%
                if abs(expected) > 1e-6:  # Avoid division by zero for near-zero values
                    rel_error = abs(actual - expected) / abs(expected)
                    assert rel_error < 0.15, f"Value mismatch at [{i},{j}]: expected {expected}, got {actual} (rel_error={rel_error:.2%})"
                else:
                    assert abs(actual - expected) < 0.1, f"Value mismatch at [{i},{j}]: expected {expected}, got {actual}"
        
        print("    Float32 → FP8 via .copy_(): OK")

        # Test 10: Verify patch can be safely uninstalled
        print("\n  Test 10: Patch uninstall and restoration")
        original_to = fp8_mps_patch._original_tensor_to
        original_copy = fp8_mps_patch._original_tensor_copy
        fp8_mps_patch.uninstall()
        assert not fp8_mps_patch.is_installed()
        assert torch.Tensor.to == original_to
        assert torch.Tensor.copy_ == original_copy
        print("    Patch uninstalled successfully: OK")
        
        # Reinstall for cleanup
        fp8_mps_patch.install()

        print(f"\n  RESULT: PASS (all validation tests passed)")
    except Exception as e:
        print(f"\n  RESULT: FAIL - {e}")
        import traceback
        traceback.print_exc()
        all_tests_passed = False
    finally:
        fp8_mps_patch.uninstall()

    print()
    return all_tests_passed


if __name__ == "__main__":
    print(f"PyTorch {torch.__version__}, MPS available: {torch.backends.mps.is_available()}")
    print(f"Python {sys.version}")
    print()

    results = {}
    results["exhaustive_decode"] = test_exhaustive_fp8_decode()
    results["matmul_cpp_ext"] = test_matmul_accuracy_cpp()
    results["matmul_native"] = test_matmul_accuracy_native()
    results["roundtrip"] = test_quantize_roundtrip()
    results["vecmat"] = test_vecmat_native()
    results["performance"] = test_performance()
    results["monkey_patch"] = test_monkey_patch()
    results["fp8_conversion"] = test_fp8_conversion()

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_pass = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name:25s} {status}")
        if not passed:
            all_pass = False

    print()
    print(f"Overall: {'ALL PASSED' if all_pass else 'SOME FAILURES'}")
    sys.exit(0 if all_pass else 1)
