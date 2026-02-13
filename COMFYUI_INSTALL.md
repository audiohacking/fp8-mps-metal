# Installing FP8 MPS Metal Support in ComfyUI

This guide explains how to install FP8 support for Apple Silicon (MPS) in ComfyUI.

## The Problem

If you're seeing this error when trying to use FP8-quantized models (FLUX, SD3.5, etc.) in ComfyUI on Mac:

```
TypeError: Trying to convert Float8_e4m3fn to the MPS backend but it does not have support for that dtype.
```

This patch fixes it!

## Installation

### Step 1: Install as a Custom Node

1. Open Terminal
2. Navigate to your ComfyUI custom_nodes directory:
   ```bash
   cd /path/to/ComfyUI/custom_nodes
   ```
   
   Common locations:
   - `/Applications/ComfyUI.app/Contents/Resources/ComfyUI/custom_nodes` (ComfyUI.app)
   - `~/ComfyUI/custom_nodes` (manual install)
   - `/Users/YOUR_USERNAME/Documents/ComfyUI/custom_nodes` (user directory)

3. Clone this repository:
   ```bash
   git clone https://github.com/audiohacking/fp8-mps-metal.git
   ```

4. Restart ComfyUI

### Step 2: Verify Installation

When ComfyUI starts, you should see this message in the console:

```
======================================================================
✓ FP8 MPS Metal patch installed successfully!
======================================================================
Float8_e4m3fn operations on MPS are now supported.
This enables:
  • FP8 model weight loading (FLUX/SD3.5)
  • FP8 quantization on MPS
  • FP8 stochastic rounding (.copy_() operations)
  • FP8 matrix multiplication via Metal kernels
======================================================================
```

If you see a warning instead, please report it as an issue.

### Step 3: Test with FP8 Models

Try loading an FP8-quantized model (FLUX, SD3.5, etc.). It should now work without errors!

## Troubleshooting

### "Patch already installed" message

This is normal if you restart ComfyUI multiple times. The patch persists in memory.

### Still getting the error

1. **Check the custom_nodes directory**: Make sure `fp8-mps-metal` is actually in your custom_nodes folder
2. **Check the console**: Look for the installation success message
3. **Try manual installation**: See "Alternative Installation Methods" below

### Warning message on startup

If you see a warning that the patch failed to install, check:
- PyTorch version (should be 2.4+)
- Python version (should be 3.10+)
- Copy the full error and report it as an issue

## Alternative Installation Methods

### Method 1: Manual Python Script

Add this to your ComfyUI startup script or create a new Python file in `custom_nodes`:

```python
import sys
import os

# Add fp8-mps-metal to path
fp8_path = "/path/to/fp8-mps-metal"
if fp8_path not in sys.path:
    sys.path.insert(0, fp8_path)

import fp8_mps_patch
fp8_mps_patch.install()
```

### Method 2: Modify ComfyUI main.py

Add these lines near the top of ComfyUI's `main.py` (after imports):

```python
try:
    import fp8_mps_patch
    fp8_mps_patch.install()
    print("✓ FP8 MPS Metal patch installed")
except:
    pass
```

## Uninstalling

To remove the patch:

1. Delete the `fp8-mps-metal` folder from `custom_nodes`
2. Restart ComfyUI

Or to temporarily disable without removing:
```python
import fp8_mps_patch
fp8_mps_patch.uninstall()
```

## What This Patch Does

The patch monkey-patches three PyTorch methods to enable FP8 operations on MPS:

1. **`Tensor.to()`** - Enables `.to(torch.float8_e4m3fn)` conversions
2. **`Tensor.copy_()`** - Enables FP8 stochastic rounding (used internally by ComfyUI)
3. **`torch._scaled_mm()`** - Enables FP8 matrix multiplication via Metal kernels

All FP8 operations are converted to use uint8 views and Metal compute shaders, bypassing MPS's lack of native FP8 support.

## Support

- **Issues**: https://github.com/audiohacking/fp8-mps-metal/issues
- **Documentation**: See README.md for technical details
- **Performance**: Check MPS_FINDINGS.md for benchmarks

## Requirements

- Apple Silicon Mac (M1/M2/M3/M4)
- macOS with MPS support
- PyTorch 2.4+ (2.10+ recommended)
- Python 3.10+

The patch works with both PyTorch stable and nightly builds.
