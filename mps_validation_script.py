#!/usr/bin/env python3
"""
Apple MPS Hardware Validation Script for fp8-mps-metal VAE Tiling

Run this script on Apple Silicon (M1/M2/M3/M4) to validate the VAE decode
tiling strategy works correctly on real MPS hardware.

Usage:
    python mps_validation_script.py

This will:
1. Check that MPS is available
2. Test the tiling logic with real MPS tensors
3. Simulate large VAE decodes to verify the patch works
4. Report performance metrics
"""

import torch
import sys
import os
import time

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def check_mps_available():
    """Check if MPS is available on this system."""
    print("="*70)
    print("MPS AVAILABILITY CHECK")
    print("="*70)
    
    if not torch.backends.mps.is_available():
        print("✗ MPS is NOT available on this system")
        print("  This script requires Apple Silicon with MPS support")
        return False
    
    print("✓ MPS is available")
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  MPS built: {torch.backends.mps.is_built()}")
    return True


def test_mps_tensor_operations():
    """Test basic MPS tensor operations."""
    print("\n" + "="*70)
    print("BASIC MPS TENSOR OPERATIONS")
    print("="*70)
    
    try:
        # Create tensor on MPS
        x = torch.randn(100, 100, device='mps')
        print(f"✓ Created tensor on MPS: {x.shape}, device: {x.device}")
        
        # Perform operation
        y = x @ x.T
        print(f"✓ Matrix multiply successful: {y.shape}")
        
        # Move to CPU and back
        x_cpu = x.to('cpu')
        x_mps_again = x_cpu.to('mps')
        print(f"✓ Device transfers working: CPU ↔ MPS")
        
        return True
    except Exception as e:
        print(f"✗ MPS tensor operation failed: {e}")
        return False


def test_tiling_with_mps():
    """Test the tiling functions with real MPS tensors."""
    print("\n" + "="*70)
    print("VAE TILING WITH MPS TENSORS")
    print("="*70)
    
    try:
        from fp8_mps_patch import (
            _tile_tensor_spatial, 
            _reconstruct_from_tiles,
            MPS_TENSOR_SIZE_THRESHOLD,
            VAE_UPSCALE_FACTOR
        )
        
        # Test 1: Small MPS tensor
        print("\nTest 1: Small tensor on MPS (no tiling)")
        small = torch.randn(1, 4, 64, 64, device='mps')
        tiles, tile_info = _tile_tensor_spatial(small, MPS_TENSOR_SIZE_THRESHOLD)
        print(f"  Input: {small.shape}, device: {small.device}")
        print(f"  Tiles: {len(tiles)}")
        assert len(tiles) == 1
        assert tiles[0].device.type == 'mps'
        print("  ✓ Pass")
        
        # Test 2: Large MPS tensor requiring tiling
        print("\nTest 2: Large tensor on MPS (requires tiling)")
        large = torch.randn(1, 4, 512, 512, device='mps')
        tiles, tile_info = _tile_tensor_spatial(large, 500_000)  # Force tiling
        print(f"  Input: {large.shape}, device: {large.device}, elements: {large.numel():,}")
        print(f"  Tiles: {len(tiles)}")
        
        for i, tile in enumerate(tiles):
            print(f"    Tile {i}: {tile.shape}, device: {tile.device}, elements: {tile.numel():,}")
            assert tile.device.type == 'mps', "Tile should stay on MPS"
        
        print("  ✓ Pass: All tiles remain on MPS")
        
        # Test 3: Reconstruction
        print("\nTest 3: Tile reconstruction")
        # Simulate upscaling (like VAE decode does)
        upscaled_tiles = []
        for tile in tiles:
            B, C, H, W = tile.shape
            # Simulate VAE decode: different channels, 8x spatial upscale
            upscaled = torch.randn(B, 3, H * 8, W * 8, device='mps')
            upscaled_tiles.append(upscaled)
        
        result = _reconstruct_from_tiles(upscaled_tiles, tile_info)
        expected_shape = (1, 3, large.shape[2] * 8, large.shape[3] * 8)
        print(f"  Reconstructed: {result.shape}, device: {result.device}")
        print(f"  Expected: {expected_shape}")
        assert result.shape == expected_shape
        assert result.device.type == 'mps'
        print("  ✓ Pass: Reconstruction preserves MPS device")
        
        return True
        
    except Exception as e:
        print(f"✗ Tiling test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_strategy_thresholds():
    """Test the strategy decision logic with realistic tensor sizes."""
    print("\n" + "="*70)
    print("STRATEGY THRESHOLD VALIDATION")
    print("="*70)
    
    try:
        from fp8_mps_patch import MPS_TENSOR_SIZE_THRESHOLD, VAE_UPSCALE_FACTOR
        
        # Real-world VAE latent sizes
        test_cases = [
            {'name': '512×512 image (SD 1.5)', 'shape': (1, 4, 64, 64), 'expected': 'pass_through'},
            {'name': '1024×1024 image (SDXL)', 'shape': (1, 4, 128, 128), 'expected': 'pass_through'},
            {'name': '2048×2048 image', 'shape': (1, 4, 256, 256), 'expected': 'tiled_mps'},
            {'name': '4096×4096 image', 'shape': (1, 4, 512, 512), 'expected': 'tiled_mps'},
            {'name': '8192×8192 image', 'shape': (1, 4, 1024, 1024), 'expected': 'cpu_fallback'},
        ]
        
        print(f"\nMPS Threshold: {MPS_TENSOR_SIZE_THRESHOLD:,} elements")
        print(f"VAE Upscale Factor: {VAE_UPSCALE_FACTOR}x\n")
        
        for tc in test_cases:
            shape = tc['shape']
            numel = shape[0] * shape[1] * shape[2] * shape[3]
            output_est = numel * VAE_UPSCALE_FACTOR
            
            # Determine strategy
            if output_est > MPS_TENSOR_SIZE_THRESHOLD * 5:
                strategy = 'cpu_fallback'
            elif output_est > MPS_TENSOR_SIZE_THRESHOLD:
                strategy = 'tiled_mps'
            else:
                strategy = 'pass_through'
            
            status = '✓' if strategy == tc['expected'] else '✗'
            print(f"{status} {tc['name']}")
            print(f"   Shape: {shape}, Input: {numel:,} elements")
            print(f"   Estimated output: {output_est:,} elements")
            print(f"   Strategy: {strategy}")
            
            assert strategy == tc['expected'], f"Strategy mismatch for {tc['name']}"
        
        print("\n✓ All strategies correct for typical use cases")
        return True
        
    except Exception as e:
        print(f"✗ Strategy test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_patch_installation():
    """Test that the patch installs correctly."""
    print("\n" + "="*70)
    print("PATCH INSTALLATION TEST")
    print("="*70)
    
    try:
        import fp8_mps_patch
        
        # Uninstall if already installed
        if fp8_mps_patch.is_installed():
            fp8_mps_patch.uninstall()
            print("  Uninstalled existing patch")
        
        # Install
        fp8_mps_patch.install()
        print("  ✓ Patch installed")
        
        # Verify environment variable
        import os
        fallback = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK")
        print(f"  ✓ PYTORCH_ENABLE_MPS_FALLBACK = {fallback}")
        assert fallback == "1"
        
        # Verify patch state
        assert fp8_mps_patch.is_installed()
        print("  ✓ Patch state: installed")
        
        return True
        
    except Exception as e:
        print(f"✗ Patch installation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def performance_benchmark():
    """Benchmark tiling overhead on MPS."""
    print("\n" + "="*70)
    print("PERFORMANCE BENCHMARK (MPS)")
    print("="*70)
    
    try:
        from fp8_mps_patch import _tile_tensor_spatial, _reconstruct_from_tiles
        
        # Test with a realistic large tensor
        print("\nBenchmarking 2048×2048 latent (typical large image)")
        tensor = torch.randn(1, 4, 256, 256, device='mps')
        
        # Warm up
        for _ in range(3):
            tiles, tile_info = _tile_tensor_spatial(tensor, 500_000)
        
        # Time tiling
        torch.mps.synchronize()
        start = time.time()
        for _ in range(100):
            tiles, tile_info = _tile_tensor_spatial(tensor, 500_000)
        torch.mps.synchronize()
        tiling_time = (time.time() - start) / 100 * 1000  # ms
        
        print(f"  Tiling time: {tiling_time:.3f} ms (average over 100 runs)")
        print(f"  Number of tiles: {len(tiles)}")
        
        # Time reconstruction (with simulated upscaling)
        upscaled_tiles = [torch.randn(1, 3, t.shape[2] * 8, t.shape[3] * 8, device='mps') 
                         for t in tiles]
        
        torch.mps.synchronize()
        start = time.time()
        for _ in range(100):
            result = _reconstruct_from_tiles(upscaled_tiles, tile_info)
        torch.mps.synchronize()
        recon_time = (time.time() - start) / 100 * 1000  # ms
        
        print(f"  Reconstruction time: {recon_time:.3f} ms (average over 100 runs)")
        print(f"  Total overhead: {(tiling_time + recon_time):.3f} ms")
        print("\n  Note: This overhead is negligible compared to actual VAE decode time (~100-1000ms)")
        
        return True
        
    except Exception as e:
        print(f"⚠ Benchmark failed (non-critical): {e}")
        return True  # Non-critical


def main():
    """Run all validation tests."""
    print("\n" + "="*70)
    print("FP8-MPS-METAL: Apple MPS Hardware Validation")
    print("="*70)
    print("\nThis script validates the VAE tiling strategy on real MPS hardware")
    print()
    
    results = []
    
    # Critical tests
    results.append(("MPS Available", check_mps_available()))
    if not results[-1][1]:
        print("\n✗ MPS not available. Cannot continue.")
        print("This script requires Apple Silicon (M1/M2/M3/M4) with MPS support.")
        return False
    
    results.append(("Basic MPS Operations", test_mps_tensor_operations()))
    results.append(("Patch Installation", test_patch_installation()))
    results.append(("Tiling with MPS", test_tiling_with_mps()))
    results.append(("Strategy Thresholds", test_strategy_thresholds()))
    
    # Non-critical benchmark
    performance_benchmark()
    
    # Summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print("="*70)
    if passed == total:
        print(f"✓ ALL TESTS PASSED ({passed}/{total})")
        print("\nThe VAE tiling strategy is working correctly on your MPS hardware!")
        print("Large VAE decodes will now:")
        print("  • Stay on GPU for medium-sized images (tiled decode)")
        print("  • Only fall back to CPU for extremely large images (> 8K)")
    else:
        print(f"✗ SOME TESTS FAILED ({passed}/{total})")
        print("\nPlease report this issue at:")
        print("https://github.com/audiohacking/fp8-mps-metal/issues")
    print("="*70)
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
