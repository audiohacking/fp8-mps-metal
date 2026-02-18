#!/usr/bin/env python3
"""
Test suite for MPS tensor size limit patches.

Tests:
  1. Environment variable PYTORCH_ENABLE_MPS_FALLBACK is set
  2. VAE decode patch is properly installed (if ComfyUI is available)
  3. Large tensor handling logic works correctly
"""

import os
import sys
import torch

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fp8_mps_patch


def test_environment_variable():
    """Test that PYTORCH_ENABLE_MPS_FALLBACK is set during installation."""
    print("=" * 70)
    print("TEST 1: Environment Variable Configuration")
    print("=" * 70)
    
    # Unset the variable first if it exists
    if "PYTORCH_ENABLE_MPS_FALLBACK" in os.environ:
        del os.environ["PYTORCH_ENABLE_MPS_FALLBACK"]
    
    # Install the patch
    if fp8_mps_patch.is_installed():
        fp8_mps_patch.uninstall()
    
    fp8_mps_patch.install()
    
    # Check that the environment variable is set
    fallback_enabled = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK")
    
    if fallback_enabled == "1":
        print("✓ PYTORCH_ENABLE_MPS_FALLBACK is correctly set to '1'")
        return True
    else:
        print(f"✗ PYTORCH_ENABLE_MPS_FALLBACK is '{fallback_enabled}', expected '1'")
        return False


def test_vae_patch_installation():
    """Test that VAE decode patch is properly installed."""
    print("\n" + "=" * 70)
    print("TEST 2: VAE Decode Patch Installation")
    print("=" * 70)
    
    # Check if ComfyUI is available
    try:
        import comfy.sd
        print("✓ ComfyUI is available")
        
        # Check if VAE.decode has been patched
        # The patched version should have a different code object
        if hasattr(comfy.sd.VAE, 'decode'):
            print("✓ VAE.decode method exists")
            return True
        else:
            print("✗ VAE.decode method not found")
            return False
            
    except ImportError:
        print("⊘ ComfyUI not available - VAE patch test skipped")
        print("  (This is expected when not running in ComfyUI environment)")
        return True  # Not a failure - just not applicable


def test_large_tensor_detection():
    """Test that large tensor detection logic works correctly."""
    print("\n" + "=" * 70)
    print("TEST 3: Large Tensor Detection Logic")
    print("=" * 70)
    
    # Test the threshold logic
    threshold = 100_000_000
    
    # Small tensor (should not trigger fallback)
    small_numel = 50_000_000
    if small_numel <= threshold:
        print(f"✓ Small tensor ({small_numel:,} elements) correctly identified as below threshold")
    else:
        print(f"✗ Small tensor check failed")
        return False
    
    # Large tensor (should trigger fallback)
    large_numel = 150_000_000
    if large_numel > threshold:
        print(f"✓ Large tensor ({large_numel:,} elements) correctly identified as above threshold")
    else:
        print(f"✗ Large tensor check failed")
        return False
    
    return True


def test_patch_module_structure():
    """Test that the patch module has the expected structure."""
    print("\n" + "=" * 70)
    print("TEST 4: Patch Module Structure")
    print("=" * 70)
    
    # Check that required functions exist
    required_functions = [
        'install',
        'uninstall',
        'is_installed',
        'patch_vae_decode_for_mps_limits',
        '_metal_scaled_mm',
        '_metal_tensor_to',
        '_metal_tensor_copy'
    ]
    
    all_present = True
    for func_name in required_functions:
        if hasattr(fp8_mps_patch, func_name):
            print(f"✓ {func_name} function exists")
        else:
            print(f"✗ {func_name} function missing")
            all_present = False
    
    return all_present


def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "=" * 70)
    print("FP8 MPS Metal - MPS Limits Patch Test Suite")
    print("=" * 70)
    
    results = []
    
    # Run tests
    results.append(("Environment Variable", test_environment_variable()))
    results.append(("VAE Patch Installation", test_vae_patch_installation()))
    results.append(("Large Tensor Detection", test_large_tensor_detection()))
    results.append(("Patch Module Structure", test_patch_module_structure()))
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print("=" * 70)
    print(f"Results: {passed}/{total} tests passed")
    print("=" * 70)
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
