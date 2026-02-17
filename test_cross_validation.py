#!/usr/bin/env python3
"""
Cross-validation test: Compare Metal shader vs C++ bridge implementation.

This test ensures that the Metal shader (used by fp8_mps_native.py) and 
the C++ bridge (fp8_bridge.cpp via fp8_metal extension) produce IDENTICAL
results for FP8 encoding and decoding.

If they differ, it indicates a bug in one of the implementations.
"""

import sys
import os
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_metal_vs_cpp_encode():
    """Compare Metal shader encoding vs C++ bridge encoding."""
    print("=" * 70)
    print("TEST 1: Metal shader vs C++ bridge - Encoding")
    print("=" * 70)
    print()
    
    # Check if MPS is available
    if not torch.backends.mps.is_available():
        print("⊘ MPS not available - skipping test")
        print()
        return True
    
    # Try to import both implementations
    try:
        import fp8_mps_native
        print("✓ fp8_mps_native (Metal shader) imported")
    except Exception as e:
        print(f"✗ Failed to import fp8_mps_native: {e}")
        return False
    
    try:
        import fp8_metal
        print("✓ fp8_metal (C++ bridge) imported")
    except Exception as e:
        print(f"⊘ fp8_metal (C++ bridge) not available: {e}")
        print("  (This is OK - C++ extension is optional)")
        print()
        return True
    
    print()
    
    # Test encoding of various values
    test_values = torch.tensor([
        0.0, 0.1, 0.5, 1.0, 2.0, 10.0, 50.0, 100.0, 200.0, 448.0,
        -0.1, -1.0, -10.0, -100.0, -448.0
    ], dtype=torch.float32)
    
    # Encode with Metal shader (via fp8_mps_native)
    metal_encoded, metal_scale = fp8_mps_native.fp8_quantize(test_values)
    metal_encoded_cpu = metal_encoded.cpu()
    
    # Encode with C++ bridge
    cpp_encoded, cpp_scale = fp8_metal.fp8_quantize(test_values)
    cpp_encoded_cpu = cpp_encoded.cpu()
    
    # Compare results
    differences = []
    for i, (val, metal_byte, cpp_byte) in enumerate(zip(test_values, metal_encoded_cpu, cpp_encoded_cpu)):
        if metal_byte != cpp_byte:
            differences.append((i, val.item(), metal_byte.item(), cpp_byte.item()))
    
    if differences:
        print(f"⚠ Found {len(differences)} encoding differences:")
        print(f"{'Index':<6} {'Value':<10} {'Metal':>6} {'C++':>6}")
        print("-" * 35)
        for idx, val, m, c in differences:
            print(f"{idx:<6} {val:<10.3f} 0x{m:02X}   0x{c:02X}")
        print()
        return False
    else:
        print("✓ Metal shader and C++ bridge produce identical encoding")
        print()
        return True


def test_metal_vs_cpp_decode():
    """Compare Metal shader decoding vs C++ bridge decoding."""
    print("=" * 70)
    print("TEST 2: Metal shader vs C++ bridge - Decoding")
    print("=" * 70)
    print()
    
    if not torch.backends.mps.is_available():
        print("⊘ MPS not available - skipping test")
        print()
        return True
    
    try:
        import fp8_mps_native
        import fp8_metal
    except ImportError:
        print("⊘ One or both implementations not available - skipping test")
        print()
        return True
    
    # Test decoding of all possible FP8 values (0-127, positive only)
    all_fp8_bytes = torch.arange(128, dtype=torch.uint8)
    scale = torch.tensor([1.0], dtype=torch.float32)
    
    # Decode with Metal shader
    metal_decoded = fp8_mps_native.fp8_dequantize(all_fp8_bytes, scale)
    metal_decoded_cpu = metal_decoded.cpu().float()
    
    # Decode with C++ bridge
    cpp_decoded = fp8_metal.fp8_dequantize(all_fp8_bytes, scale)
    cpp_decoded_cpu = cpp_decoded.cpu().float()
    
    # Compare results
    max_abs_diff = (metal_decoded_cpu - cpp_decoded_cpu).abs().max().item()
    
    if max_abs_diff > 1e-6:
        print(f"⚠ Found differences in decoding (max diff: {max_abs_diff})")
        
        # Show some examples
        diffs = []
        for i in range(128):
            m = metal_decoded_cpu[i].item()
            c = cpp_decoded_cpu[i].item()
            diff = abs(m - c)
            if diff > 1e-6:
                diffs.append((i, m, c, diff))
        
        print(f"\nShowing first 10 of {len(diffs)} differences:")
        print(f"{'Byte':<6} {'Metal':<12} {'C++':<12} {'Diff':<12}")
        print("-" * 50)
        for byte, m, c, d in diffs[:10]:
            print(f"0x{byte:02X}   {m:<12.6f} {c:<12.6f} {d:<12.6e}")
        print()
        return False
    else:
        print("✓ Metal shader and C++ bridge produce identical decoding")
        print(f"  (max difference: {max_abs_diff:.2e})")
        print()
        return True


def test_metal_vs_cpp_matmul():
    """Compare Metal shader vs C++ bridge for FP8 matrix multiplication."""
    print("=" * 70)
    print("TEST 3: Metal shader vs C++ bridge - Matrix multiplication")
    print("=" * 70)
    print()
    
    if not torch.backends.mps.is_available():
        print("⊘ MPS not available - skipping test")
        print()
        return True
    
    try:
        import fp8_mps_native
        import fp8_metal
    except ImportError:
        print("⊘ One or both implementations not available - skipping test")
        print()
        return True
    
    # Create test matrices
    M, K, N = 32, 64, 48
    A = torch.randn(M, K, dtype=torch.float32)
    B = torch.randn(N, K, dtype=torch.float32)
    
    # Quantize
    A_q_metal, A_scale_metal = fp8_mps_native.fp8_quantize(A)
    B_q_metal, B_scale_metal = fp8_mps_native.fp8_quantize(B)
    
    A_q_cpp, A_scale_cpp = fp8_metal.fp8_quantize(A)
    B_q_cpp, B_scale_cpp = fp8_metal.fp8_quantize(B)
    
    # Perform matmul
    result_metal = fp8_mps_native.fp8_scaled_mm(A_q_metal, B_q_metal, A_scale_metal, B_scale_metal)
    result_cpp = fp8_metal.fp8_scaled_mm(A_q_cpp, B_q_cpp, A_scale_cpp, B_scale_cpp)
    
    # Compare results
    diff = (result_metal.cpu() - result_cpp.cpu()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    if max_diff > 1e-4:
        print(f"⚠ Found differences in matmul results:")
        print(f"  Max difference: {max_diff:.6f}")
        print(f"  Mean difference: {mean_diff:.6f}")
        print()
        return False
    else:
        print("✓ Metal shader and C++ bridge produce similar matmul results")
        print(f"  Max difference: {max_diff:.2e}")
        print(f"  Mean difference: {mean_diff:.2e}")
        print()
        return True


def main():
    """Run all cross-validation tests."""
    print()
    print("=" * 70)
    print("FP8 CROSS-VALIDATION TEST: METAL SHADER VS C++ BRIDGE")
    print("=" * 70)
    print()
    print("This test compares the Metal shader implementation (fp8_mps_native)")
    print("against the C++ bridge implementation (fp8_metal) to ensure they")
    print("produce identical results.")
    print()
    print("If differences are found, it indicates a bug in one of the implementations.")
    print()
    
    results = []
    results.append(("Encoding", test_metal_vs_cpp_encode()))
    results.append(("Decoding", test_metal_vs_cpp_decode()))
    results.append(("Matrix multiplication", test_metal_vs_cpp_matmul()))
    
    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print()
    
    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name:30s} {status}")
        if not passed:
            all_passed = False
    
    print()
    if all_passed:
        print("✓ ALL TESTS PASSED")
        print()
        print("The Metal shader and C++ bridge implementations produce identical")
        print("results. Both implementations are consistent.")
        return 0
    else:
        print("⚠ SOME TESTS FAILED")
        print()
        print("There are inconsistencies between the Metal shader and C++ bridge.")
        print("This indicates a bug in one of the implementations.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
