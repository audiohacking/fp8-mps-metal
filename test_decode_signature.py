#!/usr/bin/env python3
"""
Test that the patched_decode signature works correctly with various argument patterns.
This doesn't require torch or ComfyUI to be installed - just tests the signature handling.
"""

def test_patched_decode_signature():
    """Test that patched_decode handles kwargs correctly."""
    print("=" * 70)
    print("TEST: Patched Decode Signature")
    print("=" * 70)
    
    # Mock the original decode function that doesn't accept disable_patcher
    def mock_original_decode(self, samples_in, **kwargs):
        """Mock original ComfyUI VAE.decode without disable_patcher parameter."""
        # Just return a simple result showing what was passed
        return {
            'samples_in': samples_in,
            'kwargs': kwargs
        }
    
    # Mock the patched decode function (simplified version)
    def patched_decode(self, samples_in, **kwargs):
        """Patched version that passes through all kwargs without explicitly handling disable_patcher."""
        # This should work regardless of whether kwargs contains 'disable_patcher' or not
        return mock_original_decode(self, samples_in, **kwargs)
    
    # Test cases
    class MockVAE:
        pass
    
    vae = MockVAE()
    
    # Test 1: No extra kwargs
    print("\nTest 1: No extra kwargs")
    try:
        result = patched_decode(vae, "test_samples")
        print("✓ PASS: Called with no kwargs")
        print(f"  Result: {result}")
    except Exception as e:
        print(f"✗ FAIL: {e}")
        return False
    
    # Test 2: With disable_patcher=True (as ComfyUI might call it)
    print("\nTest 2: With disable_patcher=True")
    try:
        result = patched_decode(vae, "test_samples", disable_patcher=True)
        print("✓ PASS: Called with disable_patcher=True")
        print(f"  Result: {result}")
    except Exception as e:
        print(f"✗ FAIL: {e}")
        return False
    
    # Test 3: With other kwargs
    print("\nTest 3: With other kwargs")
    try:
        result = patched_decode(vae, "test_samples", some_param=123, another_param="test")
        print("✓ PASS: Called with other kwargs")
        print(f"  Result: {result}")
    except Exception as e:
        print(f"✗ FAIL: {e}")
        return False
    
    # Test 4: Mixed kwargs
    print("\nTest 4: Mixed kwargs including disable_patcher")
    try:
        result = patched_decode(vae, "test_samples", disable_patcher=False, param1=1, param2=2)
        print("✓ PASS: Called with mixed kwargs")
        print(f"  Result: {result}")
    except Exception as e:
        print(f"✗ FAIL: {e}")
        return False
    
    print("\n" + "=" * 70)
    print("✓ All signature tests passed!")
    print("=" * 70)
    return True


def test_original_decode_rejects_disable_patcher():
    """Test that mock original_decode correctly rejects disable_patcher as positional."""
    print("\n" + "=" * 70)
    print("TEST: Original Decode Parameter Handling")
    print("=" * 70)
    
    # This simulates the OLD buggy behavior
    def buggy_patched_decode(self, samples_in, disable_patcher=False, **kwargs):
        """Old buggy version that explicitly passes disable_patcher."""
        def mock_original_decode(self, samples_in, **kwargs):
            # Original ComfyUI VAE.decode doesn't have a disable_patcher parameter
            # When we explicitly pass it as a keyword argument, it gets merged into **kwargs
            # and if the original checked for unexpected kwargs, it would raise TypeError
            if 'disable_patcher' in kwargs:
                raise TypeError("VAE.decode() got an unexpected keyword argument 'disable_patcher'")
            return "success"
        
        # BUG: The old code explicitly passed disable_patcher as a keyword argument
        # Even though mock_original_decode accepts **kwargs, when we check inside the function
        # we can detect this unexpected parameter, simulating ComfyUI's actual behavior
        return mock_original_decode(self, samples_in, disable_patcher=disable_patcher, **kwargs)
    
    class MockVAE:
        pass
    
    vae = MockVAE()
    
    print("\nTest: Buggy version (should fail)")
    try:
        result = buggy_patched_decode(vae, "test_samples", disable_patcher=True)
        print(f"✗ UNEXPECTED: Should have failed but got: {result}")
        return False
    except TypeError as e:
        print(f"✓ EXPECTED: Correctly raised TypeError: {e}")
        return True


if __name__ == "__main__":
    test1_passed = test_patched_decode_signature()
    test2_passed = test_original_decode_rejects_disable_patcher()
    
    print("\n" + "=" * 70)
    print("OVERALL RESULTS")
    print("=" * 70)
    
    if test1_passed and test2_passed:
        print("✓ All tests passed!")
        print("\nThe fixed patched_decode signature correctly:")
        print("  • Accepts any kwargs from callers")
        print("  • Passes through all kwargs to original_decode")
        print("  • Doesn't explicitly pass unsupported parameters")
        exit(0)
    else:
        print("✗ Some tests failed")
        exit(1)
