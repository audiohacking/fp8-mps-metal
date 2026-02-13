#!/usr/bin/env python3
"""
Simple validation script to demonstrate the fix for FP8 value preservation.

This script shows that:
1. Float → FP8 conversion preserves value semantics (no unwanted scaling)
2. FP8 → Float conversion works correctly
3. The fix resolves the "completely corrupted" results issue
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    try:
        import torch
    except ImportError:
        print("PyTorch not installed. This script requires PyTorch with MPS support.")
        print("Skipping validation.")
        return 0

    if not torch.backends.mps.is_available():
        print("MPS not available. This script is for Apple Silicon Macs.")
        print("Skipping validation.")
        return 0

    if not hasattr(torch, 'float8_e4m3fn'):
        print("torch.float8_e4m3fn not available in this PyTorch version.")
        print("Requires PyTorch 2.4+ for FP8 support.")
        print("Skipping validation.")
        return 0

    print("=" * 70)
    print("FP8 Value Preservation Validation")
    print("=" * 70)
    print()

    # Install the patch
    import fp8_mps_patch
    if not fp8_mps_patch.is_installed():
        fp8_mps_patch.install()
        print("✓ FP8 MPS patch installed")
    else:
        print("✓ FP8 MPS patch already installed")
    print()

    # Test 1: Simple value preservation
    print("Test 1: Simple value preservation")
    print("-" * 70)
    input_values = [1.0, 2.0, 5.0, 10.0, 50.0, 100.0]
    x = torch.tensor(input_values, device="mps")
    print(f"Input (float32):  {x.cpu().tolist()}")
    
    # Convert to FP8
    x_fp8 = x.to(torch.float8_e4m3fn)
    print(f"Converted to FP8:  dtype={x_fp8.dtype}")
    
    # Convert back to float32
    x_recovered = x_fp8.to(torch.float32)
    print(f"Recovered (float32): {x_recovered.cpu().tolist()}")
    
    # Check errors
    errors = []
    for orig, recovered in zip(x.cpu(), x_recovered.cpu()):
        rel_error = abs(recovered - orig) / abs(orig) if abs(orig) > 1e-6 else abs(recovered - orig)
        errors.append(rel_error.item())
    
    max_error = max(errors)
    print(f"Max relative error: {max_error:.2%}")
    
    if max_error < 0.15:  # 15% tolerance for FP8 precision
        print("✓ PASS: Values preserved within expected precision")
    else:
        print(f"✗ FAIL: Errors too large! Max error: {max_error:.2%}")
        return 1
    print()

    # Test 2: Verify no unwanted scaling (the bug we fixed)
    print("Test 2: Verify no unwanted scaling (bug fix validation)")
    print("-" * 70)
    test_value = 3.0
    x = torch.tensor([test_value], device="mps")
    x_fp8 = x.to(torch.float8_e4m3fn)
    x_recovered = x_fp8.to(torch.float32)
    
    print(f"Input:     {test_value}")
    print(f"Recovered: {x_recovered.cpu().item():.4f}")
    
    # If automatic scaling were applied (the bug), we would get:
    # scale = 448.0 / 3.0 = 149.33
    # encoded: 3.0 * 149.33 = 448.0
    # decoded: 448.0 (NO SCALING BACK because scale was lost)
    # So if we get ~448.0 back, that's the bug!
    
    if abs(x_recovered.cpu().item() - test_value) < 1.0:
        print(f"✓ PASS: Value preserved (no unwanted scaling)")
        print(f"  The old bug would have returned ~448.0 (149x scaled up)")
    else:
        print(f"✗ FAIL: Value was incorrectly scaled!")
        print(f"  Expected ~{test_value}, got {x_recovered.cpu().item()}")
        return 1
    print()

    # Test 3: Copy operation
    print("Test 3: Copy operation value preservation")
    print("-" * 70)
    src = torch.tensor([1.0, 5.0, 10.0, 50.0], device="mps", dtype=torch.float32)
    dst = torch.empty(4, dtype=torch.float8_e4m3fn, device="mps")
    
    print(f"Source (float32): {src.cpu().tolist()}")
    
    # Copy with dtype conversion
    dst.copy_(src)
    
    # Convert back to float to check
    dst_float = dst.to(torch.float32)
    print(f"After copy (float32): {dst_float.cpu().tolist()}")
    
    max_error = max(abs(dst_float.cpu()[i] - src.cpu()[i]) / abs(src.cpu()[i]) 
                    for i in range(len(src)))
    print(f"Max relative error: {max_error:.2%}")
    
    if max_error < 0.15:
        print("✓ PASS: Copy operation preserves values")
    else:
        print(f"✗ FAIL: Copy operation corrupted values! Max error: {max_error:.2%}")
        return 1
    print()

    # Test 4: Edge cases
    print("Test 4: Edge cases (zeros, small values, large values)")
    print("-" * 70)
    edge_values = [0.0, 0.1, 0.5, 1.0, 10.0, 100.0, 440.0]
    x = torch.tensor(edge_values, device="mps")
    x_fp8 = x.to(torch.float8_e4m3fn)
    x_recovered = x_fp8.to(torch.float32)
    
    print(f"{'Input':>10s} {'Recovered':>10s} {'Error':>10s}")
    all_ok = True
    for orig, recovered in zip(x.cpu(), x_recovered.cpu()):
        if abs(orig) > 1e-6:
            error = abs(recovered - orig) / abs(orig)
        else:
            error = abs(recovered - orig)
        
        status = "✓" if error < 0.15 else "✗"
        print(f"{status} {orig:10.2f} {recovered.item():10.2f} {error:9.1%}")
        
        if error >= 0.15:
            all_ok = False
    
    if all_ok:
        print("✓ PASS: All edge cases handled correctly")
    else:
        print("✗ FAIL: Some edge cases failed")
        return 1
    print()

    print("=" * 70)
    print("ALL VALIDATION TESTS PASSED!")
    print("=" * 70)
    print()
    print("The fix correctly:")
    print("  • Preserves value semantics in Float → FP8 conversions")
    print("  • Avoids unwanted automatic scaling")
    print("  • Enables correct FP8 → Float conversions")
    print("  • Supports .copy_() operations with dtype conversion")
    print()
    print("This resolves the 'completely corrupted results' issue.")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
