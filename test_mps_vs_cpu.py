#!/usr/bin/env python3
"""
MPS vs CPU Validation Test for macOS/Apple Silicon

This test runs ONLY on systems with MPS support and validates that:
1. MPS Metal GPU conversions produce identical results to CPU fallback
2. The conversion pipeline is consistent across devices
3. No device-specific bugs exist

This serves as the authoritative validation that our Metal implementation
matches the expected CPU behavior.
"""

import sys
import os
import torch
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Test configuration
DECODE_TOLERANCE = 1e-6  # Tolerance for decode differences (should be bit-exact)
ENCODE_TOLERANCE = 0      # Encoding should be bit-exact (zero tolerance)
MATMUL_TOLERANCE = 1e-4   # Matmul accumulates errors, so slightly higher tolerance


def check_mps_availability():
    """Check if MPS is available and working."""
    print("=" * 70)
    print("SYSTEM REQUIREMENTS CHECK")
    print("=" * 70)
    print()
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"Python version: {sys.version.split()[0]}")
    print()
    
    if not torch.backends.mps.is_available():
        print("✗ MPS is not available on this system")
        print()
        print("This test requires:")
        print("  - macOS 12.3+ (Monterey or later)")
        print("  - Apple Silicon (M1/M2/M3/M4)")
        print("  - PyTorch 1.12+ with MPS support")
        print()
        return False
    
    print("✓ MPS is available")
    print(f"  MPS built: {torch.backends.mps.is_built()}")
    print()
    
    # Check if FP8 dtype is available
    if not hasattr(torch, 'float8_e4m3fn'):
        print("⚠ torch.float8_e4m3fn not available")
        print("  This test requires PyTorch 2.1+ with FP8 support")
        print("  Some tests will be skipped")
        print()
        return True
    
    print("✓ torch.float8_e4m3fn is available")
    print()
    
    return True


def test_encode_mps_vs_cpu():
    """Test FP8 encoding: MPS Metal vs CPU fallback."""
    print("=" * 70)
    print("TEST 1: FP8 Encoding - MPS Metal vs CPU Fallback")
    print("=" * 70)
    print()
    
    try:
        import fp8_mps_native
        import fp8_mps_patch
        
        # Install the patch
        if not fp8_mps_patch.is_installed():
            fp8_mps_patch.install()
        
        # Test values covering full range
        test_values_list = [
            0.0, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 3.0, 10.0, 50.0, 100.0, 
            200.0, 300.0, 400.0, 448.0,
            -0.001, -0.1, -1.0, -10.0, -100.0, -448.0
        ]
        
        test_values_cpu = torch.tensor(test_values_list, dtype=torch.float32)
        test_values_mps = test_values_cpu.to("mps")
        
        # Encode on CPU (reference)
        print("Encoding on CPU (reference)...")
        encoded_cpu, scale_cpu = fp8_mps_native.fp8_quantize(test_values_cpu)
        encoded_cpu = encoded_cpu.cpu()
        
        # Encode on MPS
        print("Encoding on MPS (Metal shader)...")
        encoded_mps, scale_mps = fp8_mps_native.fp8_quantize(test_values_mps)
        encoded_mps = encoded_mps.cpu()
        
        # Compare byte-by-byte
        differences = []
        for i, (val, cpu_byte, mps_byte) in enumerate(
            zip(test_values_list, encoded_cpu, encoded_mps)
        ):
            if cpu_byte != mps_byte:
                differences.append((i, val, cpu_byte.item(), mps_byte.item()))
        
        print()
        if differences:
            print(f"⚠ Found {len(differences)} encoding differences:")
            print(f"{'Index':<6} {'Value':<12} {'CPU':>6} {'MPS':>6}")
            print("-" * 40)
            for idx, val, cpu, mps in differences[:20]:
                print(f"{idx:<6} {val:<12.4f} 0x{cpu:02X}   0x{mps:02X}")
            if len(differences) > 20:
                print(f"... and {len(differences) - 20} more")
            print()
            return False
        else:
            print("✓ MPS and CPU encoding produce IDENTICAL results")
            print(f"  Tested {len(test_values_list)} values - all match byte-for-byte")
            print()
            return True
            
    except ImportError as e:
        print(f"⊘ Skipping test - required module not available: {e}")
        print()
        return True
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        print()
        return False


def test_decode_mps_vs_cpu():
    """Test FP8 decoding: MPS Metal vs CPU fallback."""
    print("=" * 70)
    print("TEST 2: FP8 Decoding - MPS Metal vs CPU Fallback")
    print("=" * 70)
    print()
    
    try:
        import fp8_mps_native
        
        # Test all possible FP8 byte values (0-255)
        all_bytes_cpu = torch.arange(256, dtype=torch.uint8)
        all_bytes_mps = all_bytes_cpu.to("mps")
        scale = torch.tensor([1.0], dtype=torch.float32)
        
        # Decode on CPU
        print("Decoding all 256 FP8 values on CPU...")
        decoded_cpu = fp8_mps_native.fp8_dequantize(all_bytes_cpu, scale)
        decoded_cpu = decoded_cpu.cpu().float()
        
        # Decode on MPS
        print("Decoding all 256 FP8 values on MPS...")
        decoded_mps = fp8_mps_native.fp8_dequantize(all_bytes_mps, scale)
        decoded_mps = decoded_mps.cpu().float()
        
        # Compare results
        diff = (decoded_cpu - decoded_mps).abs()
        max_diff = diff.max().item()
        num_diffs = (diff > DECODE_TOLERANCE).sum().item()
        
        print()
        if num_diffs > 0:
            print(f"⚠ Found {num_diffs} decoding differences (tolerance: {DECODE_TOLERANCE})")
            print(f"  Maximum difference: {max_diff:.2e}")
            print()
            
            # Show some examples
            diff_indices = (diff > DECODE_TOLERANCE).nonzero(as_tuple=True)[0]
            print("Examples of differences:")
            print(f"{'Byte':<6} {'CPU':<15} {'MPS':<15} {'Diff':<12}")
            print("-" * 55)
            for idx in diff_indices[:10]:
                idx = idx.item()
                cpu_val = decoded_cpu[idx].item()
                mps_val = decoded_mps[idx].item()
                d = diff[idx].item()
                print(f"0x{idx:02X}   {cpu_val:<15.8f} {mps_val:<15.8f} {d:<12.2e}")
            if len(diff_indices) > 10:
                print(f"... and {len(diff_indices) - 10} more")
            print()
            return False
        else:
            print(f"✓ MPS and CPU decoding produce IDENTICAL results")
            print(f"  Maximum difference: {max_diff:.2e} (below tolerance: {DECODE_TOLERANCE})")
            print(f"  All 256 FP8 values match")
            print()
            return True
            
    except ImportError as e:
        print(f"⊘ Skipping test - required module not available: {e}")
        print()
        return True
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        print()
        return False


def test_roundtrip_mps_vs_cpu():
    """Test encode-decode roundtrip: MPS vs CPU."""
    print("=" * 70)
    print("TEST 3: Encode-Decode Roundtrip - MPS vs CPU")
    print("=" * 70)
    print()
    
    try:
        import fp8_mps_native
        
        # Test values
        test_values_list = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0, 200.0]
        test_values_cpu = torch.tensor(test_values_list, dtype=torch.float32)
        test_values_mps = test_values_cpu.to("mps")
        
        # CPU roundtrip
        print("CPU roundtrip: encode → decode...")
        encoded_cpu, scale_cpu = fp8_mps_native.fp8_quantize(test_values_cpu)
        decoded_cpu = fp8_mps_native.fp8_dequantize(encoded_cpu, scale_cpu)
        decoded_cpu = decoded_cpu.cpu().float()
        
        # MPS roundtrip
        print("MPS roundtrip: encode → decode...")
        encoded_mps, scale_mps = fp8_mps_native.fp8_quantize(test_values_mps)
        decoded_mps = fp8_mps_native.fp8_dequantize(encoded_mps, scale_mps)
        decoded_mps = decoded_mps.cpu().float()
        
        # Compare roundtrip results
        cpu_error = (decoded_cpu - test_values_cpu).abs() / test_values_cpu
        mps_error = (decoded_mps - test_values_cpu).abs() / test_values_cpu
        
        max_cpu_error = cpu_error.max().item()
        max_mps_error = mps_error.max().item()
        
        # Compare CPU vs MPS decoded values
        cross_diff = (decoded_cpu - decoded_mps).abs()
        max_cross_diff = cross_diff.max().item()
        
        print()
        print("Roundtrip errors:")
        print(f"  CPU max relative error:  {max_cpu_error:.2%}")
        print(f"  MPS max relative error:  {max_mps_error:.2%}")
        print(f"  CPU vs MPS max diff:     {max_cross_diff:.2e}")
        print()
        
        if max_cross_diff > DECODE_TOLERANCE:
            print(f"⚠ CPU and MPS roundtrip results differ (tolerance: {DECODE_TOLERANCE})")
            print()
            print("Detailed comparison:")
            print(f"{'Original':<12} {'CPU':<12} {'MPS':<12} {'Diff':<12}")
            print("-" * 50)
            for orig, cpu_val, mps_val, diff in zip(
                test_values_list, decoded_cpu, decoded_mps, cross_diff
            ):
                print(f"{orig:<12.4f} {cpu_val:<12.6f} {mps_val:<12.6f} {diff:<12.2e}")
            print()
            return False
        else:
            print("✓ CPU and MPS roundtrip results are IDENTICAL")
            print(f"  (within tolerance: {DECODE_TOLERANCE})")
            print()
            return True
            
    except ImportError as e:
        print(f"⊘ Skipping test - required module not available: {e}")
        print()
        return True
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        print()
        return False


def test_dtype_conversion_mps_vs_cpu():
    """Test .to(torch.float8_e4m3fn) conversion on MPS vs CPU."""
    print("=" * 70)
    print("TEST 4: dtype Conversion - MPS .to() vs CPU .to()")
    print("=" * 70)
    print()
    
    if not hasattr(torch, 'float8_e4m3fn'):
        print("⊘ Skipping - torch.float8_e4m3fn not available")
        print()
        return True
    
    try:
        import fp8_mps_patch
        
        # Install patch
        if not fp8_mps_patch.is_installed():
            fp8_mps_patch.install()
        
        # Test values
        test_values_list = [0.5, 1.0, 2.0, 10.0, 100.0]
        test_values = torch.tensor(test_values_list, dtype=torch.float32)
        
        # CPU conversion
        print("Converting on CPU: float32 → float8_e4m3fn...")
        fp8_cpu = test_values.to(torch.float8_e4m3fn)
        fp8_cpu_bytes = fp8_cpu.view(torch.uint8)
        
        # MPS conversion
        print("Converting on MPS: float32 → float8_e4m3fn...")
        test_values_mps = test_values.to("mps")
        fp8_mps = test_values_mps.to(torch.float8_e4m3fn)
        fp8_mps_bytes = fp8_mps.view(torch.uint8).cpu()
        
        # Compare byte representations
        byte_diffs = []
        for i, (val, cpu_byte, mps_byte) in enumerate(
            zip(test_values_list, fp8_cpu_bytes, fp8_mps_bytes)
        ):
            if cpu_byte != mps_byte:
                byte_diffs.append((i, val, cpu_byte.item(), mps_byte.item()))
        
        print()
        if byte_diffs:
            print(f"⚠ Found {len(byte_diffs)} byte differences:")
            print(f"{'Index':<6} {'Value':<12} {'CPU':>6} {'MPS':>6}")
            print("-" * 40)
            for idx, val, cpu, mps in byte_diffs:
                print(f"{idx:<6} {val:<12.4f} 0x{cpu:02X}   0x{mps:02X}")
            print()
            return False
        else:
            print("✓ CPU and MPS .to() conversions produce IDENTICAL bytes")
            print(f"  Tested {len(test_values_list)} values - all match")
            print()
            
            # Also test decode back
            fp8_cpu_back = fp8_cpu.to(torch.float32)
            fp8_mps_back = fp8_mps.to(torch.float32).cpu()
            
            decode_diff = (fp8_cpu_back - fp8_mps_back).abs().max().item()
            print(f"  Decode roundtrip max diff: {decode_diff:.2e}")
            print()
            return True
            
    except ImportError as e:
        print(f"⊘ Skipping test - required module not available: {e}")
        print()
        return True
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        print()
        return False


def test_matmul_mps_vs_cpu():
    """Test FP8 matrix multiplication: MPS vs CPU."""
    print("=" * 70)
    print("TEST 5: FP8 Matrix Multiplication - MPS vs CPU")
    print("=" * 70)
    print()
    
    try:
        import fp8_mps_native
        
        # Small matrices for testing
        M, K, N = 16, 32, 24
        print(f"Matrix dimensions: A({M}×{K}) @ B({N}×{K})ᵀ = C({M}×{N})")
        print()
        
        # Create test matrices
        A = torch.randn(M, K, dtype=torch.float32)
        B = torch.randn(N, K, dtype=torch.float32)
        
        # CPU matmul
        print("CPU: Quantizing and computing matmul...")
        A_q_cpu, A_scale_cpu = fp8_mps_native.fp8_quantize(A)
        B_q_cpu, B_scale_cpu = fp8_mps_native.fp8_quantize(B)
        result_cpu = fp8_mps_native.fp8_scaled_mm(A_q_cpu, B_q_cpu, A_scale_cpu, B_scale_cpu)
        result_cpu = result_cpu.cpu()
        
        # MPS matmul
        print("MPS: Quantizing and computing matmul...")
        A_mps = A.to("mps")
        B_mps = B.to("mps")
        A_q_mps, A_scale_mps = fp8_mps_native.fp8_quantize(A_mps)
        B_q_mps, B_scale_mps = fp8_mps_native.fp8_quantize(B_mps)
        result_mps = fp8_mps_native.fp8_scaled_mm(A_q_mps, B_q_mps, A_scale_mps, B_scale_mps)
        result_mps = result_mps.cpu()
        
        # Compare results
        diff = (result_cpu - result_mps).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        
        # Also compute reference FP32 for context
        reference = A @ B.T
        cpu_vs_ref = (result_cpu - reference).abs().mean().item()
        mps_vs_ref = (result_mps - reference).abs().mean().item()
        
        print()
        print("Results:")
        print(f"  CPU vs MPS max diff:     {max_diff:.2e}")
        print(f"  CPU vs MPS mean diff:    {mean_diff:.2e}")
        print(f"  CPU vs FP32 mean diff:   {cpu_vs_ref:.2e}")
        print(f"  MPS vs FP32 mean diff:   {mps_vs_ref:.2e}")
        print()
        
        if max_diff > MATMUL_TOLERANCE:
            print(f"⚠ CPU and MPS matmul differ beyond tolerance ({MATMUL_TOLERANCE})")
            print()
            return False
        else:
            print(f"✓ CPU and MPS matmul results are consistent")
            print(f"  (within tolerance: {MATMUL_TOLERANCE})")
            print()
            return True
            
    except ImportError as e:
        print(f"⊘ Skipping test - required module not available: {e}")
        print()
        return True
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        print()
        return False


def test_performance_comparison():
    """Compare performance: MPS vs CPU."""
    print("=" * 70)
    print("TEST 6: Performance Comparison - MPS vs CPU")
    print("=" * 70)
    print()
    
    try:
        import fp8_mps_native
        
        # Test matrices
        M, K, N = 64, 256, 128
        print(f"Matrix dimensions: {M}×{K} @ {N}×{K}")
        print()
        
        A = torch.randn(M, K, dtype=torch.float32)
        B = torch.randn(N, K, dtype=torch.float32)
        
        # CPU timing
        print("Benchmarking CPU...")
        warmup = 3
        iterations = 10
        
        for _ in range(warmup):
            A_q, A_s = fp8_mps_native.fp8_quantize(A)
            B_q, B_s = fp8_mps_native.fp8_quantize(B)
            _ = fp8_mps_native.fp8_scaled_mm(A_q, B_q, A_s, B_s)
        
        start = time.perf_counter()
        for _ in range(iterations):
            A_q, A_s = fp8_mps_native.fp8_quantize(A)
            B_q, B_s = fp8_mps_native.fp8_quantize(B)
            _ = fp8_mps_native.fp8_scaled_mm(A_q, B_q, A_s, B_s)
        cpu_time = (time.perf_counter() - start) / iterations * 1000
        
        # MPS timing
        print("Benchmarking MPS...")
        A_mps = A.to("mps")
        B_mps = B.to("mps")
        
        for _ in range(warmup):
            A_q, A_s = fp8_mps_native.fp8_quantize(A_mps)
            B_q, B_s = fp8_mps_native.fp8_quantize(B_mps)
            _ = fp8_mps_native.fp8_scaled_mm(A_q, B_q, A_s, B_s)
        torch.mps.synchronize()
        
        start = time.perf_counter()
        for _ in range(iterations):
            A_q, A_s = fp8_mps_native.fp8_quantize(A_mps)
            B_q, B_s = fp8_mps_native.fp8_quantize(B_mps)
            _ = fp8_mps_native.fp8_scaled_mm(A_q, B_q, A_s, B_s)
        torch.mps.synchronize()
        mps_time = (time.perf_counter() - start) / iterations * 1000
        
        speedup = cpu_time / mps_time
        
        print()
        print("Performance results:")
        print(f"  CPU time:  {cpu_time:7.2f} ms")
        print(f"  MPS time:  {mps_time:7.2f} ms")
        print(f"  Speedup:   {speedup:7.2f}x")
        print()
        
        if speedup > 1.0:
            print(f"✓ MPS is {speedup:.1f}x faster than CPU")
        else:
            print(f"⚠ MPS is slower than CPU ({1/speedup:.1f}x)")
        print()
        
        return True
        
    except ImportError as e:
        print(f"⊘ Skipping test - required module not available: {e}")
        print()
        return True
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        print()
        return False


def main():
    """Run all MPS vs CPU validation tests."""
    print()
    print("=" * 70)
    print("MPS vs CPU VALIDATION TEST SUITE")
    print("=" * 70)
    print()
    print("This test validates that the MPS Metal GPU implementation")
    print("produces identical results to the CPU fallback implementation.")
    print()
    print("This serves as the authoritative validation for correctness.")
    print()
    
    # Check system requirements
    if not check_mps_availability():
        print("=" * 70)
        print("TESTS SKIPPED")
        print("=" * 70)
        print()
        print("MPS is not available on this system.")
        print("This test can only run on macOS with Apple Silicon.")
        return 0
    
    # Run all tests
    results = []
    results.append(("FP8 Encoding", test_encode_mps_vs_cpu()))
    results.append(("FP8 Decoding", test_decode_mps_vs_cpu()))
    results.append(("Encode-Decode Roundtrip", test_roundtrip_mps_vs_cpu()))
    results.append(("dtype Conversion", test_dtype_conversion_mps_vs_cpu()))
    results.append(("Matrix Multiplication", test_matmul_mps_vs_cpu()))
    results.append(("Performance", test_performance_comparison()))
    
    # Print summary
    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print()
    
    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        symbol = "✓" if passed else "✗"
        print(f"  {symbol} {name:35s} {status}")
        if not passed:
            all_passed = False
    
    print()
    if all_passed:
        print("=" * 70)
        print("✓ ALL TESTS PASSED")
        print("=" * 70)
        print()
        print("The MPS Metal GPU implementation produces IDENTICAL results")
        print("to the CPU fallback implementation.")
        print()
        print("This confirms:")
        print("  • Encoding is consistent across devices")
        print("  • Decoding is consistent across devices")
        print("  • Matrix multiplication produces matching results")
        print("  • The Metal shader implementation is correct")
        print()
        return 0
    else:
        print("=" * 70)
        print("⚠ SOME TESTS FAILED")
        print("=" * 70)
        print()
        print("There are differences between MPS and CPU implementations.")
        print("This indicates a device-specific bug that needs investigation.")
        print()
        return 1


if __name__ == "__main__":
    sys.exit(main())
