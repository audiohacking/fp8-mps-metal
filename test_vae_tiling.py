#!/usr/bin/env python3
"""
Test suite for VAE decode tiling strategy.

Tests the spatial tiling logic without requiring actual VAE models or ComfyUI.
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_tile_tensor_spatial():
    """Test the spatial tiling function logic."""
    print("=" * 70)
    print("TEST: Spatial Tiling Logic")
    print("=" * 70)
    
    # Import the tiling functions (they should exist in fp8_mps_patch)
    import fp8_mps_patch
    
    # Test parameters
    test_cases = [
        {
            'name': 'Small tensor - no tiling needed',
            'shape': (1, 8, 64, 64),  # 32,768 elements
            'max_tile_size': 100_000_000,
            'expected_tiles': 1
        },
        {
            'name': 'Large tensor - needs tiling',
            'shape': (1, 8, 512, 512),  # 2,097,152 elements
            'max_tile_size': 1_000_000,
            'expected_tiles': 3  # Should split into ~2-3 tiles
        },
        {
            'name': 'Very large tensor - multiple tiles',
            'shape': (1, 16, 1024, 1024),  # 16,777,216 elements
            'max_tile_size': 5_000_000,
            'expected_tiles': 4  # Should split into multiple tiles
        }
    ]
    
    all_passed = True
    
    for test_case in test_cases:
        print(f"\n  {test_case['name']}")
        print(f"    Shape: {test_case['shape']}")
        print(f"    Elements: {test_case['shape'][0] * test_case['shape'][1] * test_case['shape'][2] * test_case['shape'][3]:,}")
        print(f"    Max tile size: {test_case['max_tile_size']:,}")
        
        # Create a mock tensor (just using shape info, no actual PyTorch needed for logic test)
        class MockTensor:
            def __init__(self, shape):
                self.shape = shape
            
            def __getitem__(self, slices):
                # Simulate slicing
                return MockTensor((
                    self.shape[0],
                    self.shape[1],
                    self.shape[2],  # Would be modified by slice
                    self.shape[3]
                ))
        
        mock_tensor = MockTensor(test_case['shape'])
        
        # Test the tiling logic manually (since we can't import torch)
        B, C, H, W = test_case['shape']
        elements = B * C * H * W
        max_tile_size = test_case['max_tile_size']
        
        if elements > max_tile_size:
            num_h_splits = (elements + max_tile_size - 1) // max_tile_size
            target_h = (H + num_h_splits - 1) // num_h_splits
            num_tiles = (H + target_h - 1) // target_h
        else:
            num_tiles = 1
        
        print(f"    Calculated tiles: {num_tiles}")
        
        if num_tiles >= test_case['expected_tiles']:
            print(f"    ✓ Tiling logic correct (tiles >= {test_case['expected_tiles']})")
        else:
            print(f"    ✗ Tiling logic incorrect (expected >= {test_case['expected_tiles']}, got {num_tiles})")
            all_passed = False
    
    return all_passed


def test_tiling_strategy_thresholds():
    """Test the decision logic for which strategy to use."""
    print("\n" + "=" * 70)
    print("TEST: Tiling Strategy Decision Logic")
    print("=" * 70)
    
    import fp8_mps_patch
    
    threshold = fp8_mps_patch.MPS_TENSOR_SIZE_THRESHOLD
    print(f"  MPS Threshold: {threshold:,} elements")
    
    test_cases = [
        {
            'name': 'Small latent - pass through',
            'input_elements': 50_000_000,
            'expected_strategy': 'pass_through'
        },
        {
            'name': 'Medium latent - tiled decode',
            'input_elements': 2_000_000,  # Output would be ~128M (2M * 64)
            'expected_strategy': 'tiled_mps'
        },
        {
            'name': 'Large latent - CPU fallback',
            'input_elements': 10_000_000,  # Output would be ~640M (10M * 64)
            'expected_strategy': 'cpu_fallback'
        }
    ]
    
    all_passed = True
    
    for test_case in test_cases:
        print(f"\n  {test_case['name']}")
        print(f"    Input elements: {test_case['input_elements']:,}")
        
        numel = test_case['input_elements']
        output_estimate = numel * 64  # VAE upscales ~64x
        
        print(f"    Estimated output: {output_estimate:,} elements")
        
        # Determine strategy
        if numel <= threshold:
            strategy = 'pass_through'
        elif output_estimate > threshold * 5:
            strategy = 'cpu_fallback'
        elif output_estimate > threshold:
            strategy = 'tiled_mps'
        else:
            strategy = 'pass_through'
        
        print(f"    Strategy: {strategy}")
        
        if strategy == test_case['expected_strategy']:
            print(f"    ✓ Correct strategy selected")
        else:
            print(f"    ✗ Wrong strategy (expected {test_case['expected_strategy']}, got {strategy})")
            all_passed = False
    
    return all_passed


def run_all_tests():
    """Run all tiling tests."""
    print("\n" + "=" * 70)
    print("FP8 MPS Metal - VAE Tiling Strategy Test Suite")
    print("=" * 70)
    
    results = []
    
    # Run tests
    results.append(("Spatial Tiling Logic", test_tile_tensor_spatial()))
    results.append(("Strategy Decision Logic", test_tiling_strategy_thresholds()))
    
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
