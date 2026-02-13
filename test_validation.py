#!/usr/bin/env python3
"""
Comprehensive validation of Float8_e4m3fn conversion support.

Tests that can run on CPU:
- Patch installation/uninstall mechanism
- Argument parsing and dispatch logic
- Compatibility with existing operations

Tests that require MPS (documented but not run on CPU):
- Actual FP8 quantization using Metal kernels
- FP8 matmul operations
- Memory efficiency validation
"""

import sys
import torch

def test_patch_mechanism():
    """Test the monkey-patching mechanism."""
    print("\n" + "=" * 60)
    print("TEST 1: Monkey-Patch Mechanism")
    print("=" * 60)
    
    import fp8_mps_patch
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1.1: Initial state
    tests_total += 1
    if not fp8_mps_patch.is_installed():
        print("✓ Initial state: Not installed")
        tests_passed += 1
    else:
        print("✗ Initial state should not be installed")
    
    # Test 1.2: Install
    tests_total += 1
    fp8_mps_patch.install()
    if fp8_mps_patch.is_installed():
        print("✓ After install(): Installed")
        tests_passed += 1
    else:
        print("✗ Should be installed after install()")
    
    # Test 1.3: Verify both patches are active
    tests_total += 1
    if hasattr(torch, '_scaled_mm'):
        if torch._scaled_mm != fp8_mps_patch._original_scaled_mm:
            print("✓ torch._scaled_mm is patched")
            tests_passed += 1
        else:
            print("✗ torch._scaled_mm should be patched")
    else:
        print("⊘ torch._scaled_mm not available (OK for old PyTorch)")
        tests_passed += 1
    
    tests_total += 1
    if torch.Tensor.to != fp8_mps_patch._original_tensor_to:
        print("✓ Tensor.to is patched")
        tests_passed += 1
    else:
        print("✗ Tensor.to should be patched")
    
    # Test 1.4: Idempotent install
    tests_total += 1
    fp8_mps_patch.install()
    if fp8_mps_patch.is_installed():
        print("✓ Idempotent install works")
        tests_passed += 1
    else:
        print("✗ Idempotent install failed")
    
    # Test 1.5: Uninstall
    tests_total += 1
    original_to = fp8_mps_patch._original_tensor_to
    fp8_mps_patch.uninstall()
    if not fp8_mps_patch.is_installed() and torch.Tensor.to == original_to:
        print("✓ Uninstall restores original methods")
        tests_passed += 1
    else:
        print("✗ Uninstall should restore original")
    
    print(f"\nResult: {tests_passed}/{tests_total} tests passed")
    return tests_passed == tests_total


def test_argument_parsing():
    """Test that Tensor.to() argument parsing works correctly."""
    print("\n" + "=" * 60)
    print("TEST 2: Argument Parsing Compatibility")
    print("=" * 60)
    
    import fp8_mps_patch
    fp8_mps_patch.install()
    
    tests_passed = 0
    tests_total = 0
    
    x = torch.randn(4, 4)
    
    test_cases = [
        ("to(dtype)", lambda: x.to(torch.float16)),
        ("to(device)", lambda: x.to('cpu')),
        ("to(device, dtype)", lambda: x.to('cpu', torch.float16)),
        ("to(dtype=...)", lambda: x.to(dtype=torch.float16)),
        ("to(device=...)", lambda: x.to(device='cpu')),
        ("to(device=..., dtype=...)", lambda: x.to(device='cpu', dtype=torch.float16)),
    ]
    
    for desc, func in test_cases:
        tests_total += 1
        try:
            result = func()
            print(f"✓ {desc}")
            tests_passed += 1
        except Exception as e:
            print(f"✗ {desc}: {e}")
    
    fp8_mps_patch.uninstall()
    
    print(f"\nResult: {tests_passed}/{tests_total} tests passed")
    return tests_passed == tests_total


def test_fp8_dtype_support():
    """Test FP8 dtype detection and handling."""
    print("\n" + "=" * 60)
    print("TEST 3: FP8 Dtype Support")
    print("=" * 60)
    
    if not hasattr(torch, 'float8_e4m3fn'):
        print("⊘ torch.float8_e4m3fn not available in this PyTorch version")
        return True
    
    import fp8_mps_patch
    fp8_mps_patch.install()
    
    tests_passed = 0
    tests_total = 0
    
    # Test FP8 dtypes
    fp8_dtypes = [torch.float8_e4m3fn]
    if hasattr(torch, 'float8_e5m2'):
        fp8_dtypes.append(torch.float8_e5m2)
    
    for dtype in fp8_dtypes:
        tests_total += 1
        try:
            x = torch.randn(4, 4)
            # On CPU, this uses native PyTorch FP8 support
            x_fp8 = x.to(dtype)
            if x_fp8.dtype == dtype:
                print(f"✓ Conversion to {dtype}")
                tests_passed += 1
            else:
                print(f"✗ Conversion to {dtype} produced wrong dtype")
        except Exception as e:
            print(f"✗ Conversion to {dtype}: {e}")
    
    fp8_mps_patch.uninstall()
    
    print(f"\nResult: {tests_passed}/{tests_total} tests passed")
    return tests_passed == tests_total


def test_no_side_effects():
    """Verify the patch doesn't break normal operations."""
    print("\n" + "=" * 60)
    print("TEST 4: No Side Effects on Normal Operations")
    print("=" * 60)
    
    import fp8_mps_patch
    fp8_mps_patch.install()
    
    tests_passed = 0
    tests_total = 0
    
    test_operations = [
        ("float32 to float16", lambda: torch.randn(4, 4).to(torch.float16)),
        ("float32 to float64", lambda: torch.randn(4, 4).to(torch.float64)),
        ("int32 to int64", lambda: torch.randint(0, 10, (4, 4)).to(torch.int64)),
        ("device copy", lambda: torch.randn(4, 4).to('cpu')),
        ("matrix multiply", lambda: torch.randn(4, 4) @ torch.randn(4, 4)),
        ("addition", lambda: torch.randn(4, 4) + torch.randn(4, 4)),
        ("mean", lambda: torch.randn(4, 4).mean()),
    ]
    
    for desc, func in test_operations:
        tests_total += 1
        try:
            result = func()
            print(f"✓ {desc}")
            tests_passed += 1
        except Exception as e:
            print(f"✗ {desc}: {e}")
    
    fp8_mps_patch.uninstall()
    
    print(f"\nResult: {tests_passed}/{tests_total} tests passed")
    return tests_passed == tests_total


def print_mps_requirements():
    """Document what requires MPS hardware to test."""
    print("\n" + "=" * 60)
    print("TESTS REQUIRING MPS HARDWARE (Not Run)")
    print("=" * 60)
    print("""
These tests require Apple Silicon with MPS backend:

1. FP8 Quantization with Metal Kernel
   - Verify fp8_quantize() produces correct uint8 encoding
   - Verify scale calculation (max_fp8 / max_abs)
   - Test quantization accuracy across value ranges

2. FP8 to FP16 Dequantization
   - Verify fp8_dequantize() decodes correctly
   - Test all 256 possible FP8 bit patterns
   - Validate against reference implementation

3. FP8 Scaled Matrix Multiplication
   - Test fp8_scaled_mm() for various matrix sizes
   - Verify accuracy vs FP32 reference
   - Test vecmat kernel (M=1) for single-token inference
   - Benchmark performance vs CPU fallback

4. End-to-End FP8 Conversion on MPS
   - tensor.to(torch.float8_e4m3fn) on MPS device
   - CPU to MPS with FP8 conversion
   - FP8 tensor in torch._scaled_mm()
   - Memory efficiency validation (50% reduction vs FP32)

5. Integration with Real Models
   - Load FLUX/SD3.5 FP8-quantized weights
   - Run inference on MPS
   - Validate output quality

To run these tests:
1. Use macOS with Apple Silicon (M1/M2/M3/M4)
2. Install PyTorch with MPS support
3. Run: python test_fp8_metal.py
""")


def main():
    """Run all validation tests."""
    print("=" * 60)
    print("float8_e4m3fn Conversion Support Validation")
    print("=" * 60)
    print(f"PyTorch: {torch.__version__}")
    print(f"Float8_e4m3fn: {hasattr(torch, 'float8_e4m3fn')}")
    print(f"MPS Available: {torch.backends.mps.is_available()}")
    
    if not torch.backends.mps.is_available():
        print("\n⚠ MPS not available - running limited tests on CPU")
    
    results = []
    results.append(("Patch Mechanism", test_patch_mechanism()))
    results.append(("Argument Parsing", test_argument_parsing()))
    results.append(("FP8 Dtype Support", test_fp8_dtype_support()))
    results.append(("No Side Effects", test_no_side_effects()))
    
    print_mps_requirements()
    
    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        symbol = "✓" if passed else "✗"
        print(f"{symbol} {name:30s} {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ ALL TESTS PASSED")
        print("=" * 60)
        print("\nThe Float8_e4m3fn conversion patch is working correctly.")
        print("Metal kernel tests require MPS hardware (Apple Silicon).")
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
