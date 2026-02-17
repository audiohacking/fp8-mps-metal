# FP8 e4m3fn Conversion Fix Documentation

## Issue Summary

The MPS Metal GPU implementation was producing different byte values than the CPU implementation for FP8 e4m3fn encoding. This caused silent data corruption in video generation and other applications.

**Symptom:** Test 4 (dtype conversion) in `test_mps_vs_cpu.py` failed with:
```
⚠ Found 1 byte differences:
Index  Value           CPU    MPS
----------------------------------------
4      100.0000     0x6C   0x6D
```

## Root Cause

The Metal shader's `float_to_fp8_e4m3fn` function was using **round-away-from-zero** (`+ 0.5`) for mantissa bit calculation:

```metal
// BEFORE (incorrect)
uint mant = uint((mantissa - 1.0f) * 8.0f + 0.5f);
```

While PyTorch's CPU implementation uses **round-half-to-even** (banker's rounding), which is Python's `round()` function behavior:

```python
# CPU reference (correct)
mant = round(mant_frac * 8.0)  # round-half-to-even
```

### Why This Matters

For value 100.0:
- Mantissa fraction after normalization: `(100.0 / 64) - 1.0 = 0.5625`
- Multiply by 8: `0.5625 * 8 = 4.5`

Different rounding modes produce different results:
- **Round-away-from-zero**: `int(4.5 + 0.5) = 5` → byte 0x6D
- **Round-half-to-even**: `round(4.5) = 4` (rounds to even) → byte 0x6C ✓

Round-half-to-even (banker's rounding) is the IEEE 754 default and what PyTorch uses.

## Solution

Changed the Metal shader to use `rint()` which implements IEEE 754 round-to-nearest-even:

```metal
// AFTER (correct)
float mant_f = (mantissa - 1.0f) * 8.0f;
uint mant = uint(rint(mant_f));  // round-half-to-even
```

### Additional Fix: Sign Preservation

Also fixed values below minimum subnormal to preserve the sign bit:

```metal
// BEFORE
if (val < (1.0f / 512.0f)) {
    return 0;  // Always positive zero
}

// AFTER
if (val < (1.0f / 512.0f)) {
    return sign << 7;  // Preserve negative zero as 0x80
}
```

## Changes Made

### File: `fp8_matmul.metal`

1. **Line 67**: Subnormal encoding - changed to use `rint()`
2. **Line 80**: Normal encoding - changed to use `rint()`
3. **Line 59**: Subnormal flush - preserve sign bit

## Verification

Tested 29 comprehensive test values including:
- Original failing case: 100.0 → 0x6C ✓
- Sign preservation: -0.001 → 0x80 ✓
- Rounding edge cases: 0.0186 → 0x0A ✓
- Full range: 0.0, 0.5, 1.0, 10.0, 50.0, 100.0, 200.0, 400.0, 448.0
- Negative values: -0.1, -1.0, -10.0, -100.0, -448.0

**Result:** All values match CPU behavior exactly.

## Impact

This fix ensures that:
1. MPS Metal GPU produces **identical** FP8 encodings as CPU
2. Video generation and other applications no longer suffer from silent data corruption
3. The Metal implementation is fully compatible with PyTorch's CPU reference implementation

## Testing on macOS

To verify the fix on a Mac with Apple Silicon:

```bash
python test_mps_vs_cpu.py
```

Expected output for Test 4:
```
✓ CPU and MPS .to() conversions produce IDENTICAL bytes
  Tested 5 values - all match
```

## References

- IEEE 754 Standard: Round-to-nearest-even is the default rounding mode
- Python `round()`: Implements round-half-to-even (banker's rounding)
- Metal `rint()`: Implements IEEE 754 round-to-nearest-even
- FP8 e4m3fn format: 1 sign bit, 4 exponent bits (bias=7), 3 mantissa bits
