# FP8 Value Preservation Fix

## Issue Summary

The repository's FP8 support on Apple Silicon MPS was causing "completely corrupted" results when converting Float32 tensors to Float8_e4m3fn format. This document explains the root cause and the fix.

## Root Cause

The original `fp8_quantize()` function automatically scaled input values to use the full FP8 range ([-448, 448]) for maximum precision:

```python
# Old code:
amax = inp.abs().max().item()
scale = 448.0 / amax if amax > 0 else 1.0
scaled = inp * scale  # Scale to full FP8 range
# ... encode scaled values to FP8 ...
return quantized, scale  # Return scale for later dequantization
```

This scaling is correct for `torch._scaled_mm()` operations where:
1. Values are scaled to use full FP8 precision
2. Scales are stored separately
3. Scales are provided explicitly during matrix multiplication

However, this scaling is **WRONG** for `.to()` and `.copy_()` operations where:
1. Users expect value semantics to be preserved
2. No separate scale storage exists
3. The scale is lost after conversion

### Example of the Bug

```python
# Input values
x = torch.tensor([1.0, 2.0, 3.0], device="mps")

# Convert to FP8 (old behavior)
x_fp8 = x.to(torch.float8_e4m3fn)
# Internally:
#   amax = 3.0
#   scale = 448.0 / 3.0 = 149.33
#   encoded: [1.0*149.33, 2.0*149.33, 3.0*149.33] = [149.33, 298.67, 448.0]
#   scale is LOST (FP8 dtype doesn't store scales)

# Convert back to float32
x_recovered = x_fp8.to(torch.float32)
# Decodes to: [~149.33, ~298.67, ~448.0]
# Expected: [~1.0, ~2.0, ~3.0]
# CORRUPTION! Values are 149x larger than expected!
```

## The Fix

We split FP8 encoding into two functions:

### 1. `fp8_encode()` - For Value Preservation
Used by `.to()` and `.copy_()` operations:

```python
def fp8_encode(input: torch.Tensor):
    """
    Encode float to FP8 WITHOUT scaling.
    Values are clamped to [-448, 448] but preserve their magnitude.
    """
    # No scaling - encode values as-is
    # Metal kernel handles clamping
    return encoded_u8
```

### 2. `fp8_quantize()` - For Scaled Operations
Used by `torch._scaled_mm()` and matrix multiplication:

```python
def fp8_quantize(input: torch.Tensor):
    """
    Quantize float to FP8 WITH automatic scaling.
    Returns both encoded tensor and scale for later use.
    """
    scale = 448.0 / max(abs(input))
    scaled = input * scale
    return encoded_u8, inv_scale
```

## Code Changes

### fp8_mps_native.py
- Added `fp8_encode()` function without scaling
- Kept `fp8_quantize()` with scaling for backward compatibility
- Both functions use the same Metal kernel but with/without pre-scaling

### fp8_mps_patch.py

#### `_metal_tensor_to()` Changes
- Scenario 2 (Float → FP8): Now uses `fp8_encode()` instead of `fp8_quantize()`
- Scenario 3 (FP8 → Float): Added dequantization with scale=1.0

#### `_metal_tensor_copy()` Changes
- Scenario 2 (Float → FP8): Now uses `fp8_encode()` instead of `fp8_quantize()`

## Validation

### Test Cases Added
1. **Value preservation test** (`test_fp8_value_preservation`):
   - Tests small, medium, and mixed-range values
   - Verifies no automatic scaling is applied
   - Tests both `.to()` and `.copy_()` operations

2. **Validation script** (`validate_fix.py`):
   - Demonstrates the fix with clear examples
   - Shows the before/after behavior
   - Can be run standalone for verification

### Expected Results

With the fix:
```python
# Input values
x = torch.tensor([1.0, 2.0, 3.0], device="mps")

# Convert to FP8 (new behavior)
x_fp8 = x.to(torch.float8_e4m3fn)
# Internally:
#   No scaling applied
#   encoded: [1.0, 2.0, 3.0] (clamped to FP8 precision)

# Convert back to float32
x_recovered = x_fp8.to(torch.float32)
# Decodes to: [~1.0, ~2.0, ~3.0]
# ✓ CORRECT! Values preserved within FP8 precision (~7% relative error)
```

## Compatibility

### Matrix Multiplication Still Works
The `torch._scaled_mm()` path continues to use `fp8_quantize()` with scaling:

```python
A_q, A_scale = fp8_quantize(A)  # Returns scale
B_q, B_scale = fp8_quantize(B)  # Returns scale
result = torch._scaled_mm(A_q, B_q, scale_a=A_scale, scale_b=B_scale)
# ✓ Scales are provided, matmul is accurate
```

### ComfyUI Compatibility
The fix enables both paths:

1. **CPU fallback** (user's working approach):
   ```python
   # Quantize on CPU with stochastic rounding
   fp8_cpu = stochastic_round(value.cpu())
   # Move to MPS (raw bytes preserved)
   fp8_mps = fp8_cpu.to("mps")  # Our Scenario 1 handles this
   ```

2. **Direct MPS** (now works correctly):
   ```python
   # Quantize directly on MPS
   fp8_mps = value.to(torch.float8_e4m3fn)  # Our Scenario 2 handles this
   ```

Both produce semantically correct results. The CPU path uses stochastic rounding (lower quantization noise), while the MPS path uses deterministic rounding (faster, slightly higher noise).

## Performance Impact

- **No performance impact** for matrix multiplication (still uses scaled path)
- **Slightly faster** for `.to()` conversions (no scale computation)
- **Correct results** for all operations (no more corruption!)

## Technical Details

### Why Separate Functions?

We can't use a flag parameter like `scale: bool = False` because:
1. The API would be confusing (when to pass True vs False?)
2. Different return types (with/without scale)
3. Clear separation of concerns (value semantics vs scaled precision)

### Why Not Store Scale in Tensor?

PyTorch's `torch.float8_e4m3fn` dtype doesn't support storing per-tensor metadata like scales. This matches NVIDIA's FP8 design where scales are managed separately.

### FP8 Format Details

- **Format**: e4m3fn (4-bit exponent, 3-bit mantissa, sign bit)
- **Range**: [-448, 448]
- **Precision**: ~3 decimal digits
- **No infinity**: Max value is 448, NaN is 0x7F/0xFF
- **Quantization error**: ~5-15% relative error typical

## References

- PyTorch FP8 documentation: https://pytorch.org/docs/stable/torch.html#torch.float8_e4m3fn
- FP8 specification: https://arxiv.org/abs/2209.05433
- Metal Performance Shaders: https://developer.apple.com/documentation/metalperformanceshaders
