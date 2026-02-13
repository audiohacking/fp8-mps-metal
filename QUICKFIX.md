# Quick Fix for ComfyUI FP8 Error on Mac

## The Error You're Seeing

```
TypeError: Trying to convert Float8_e4m3fn to the MPS backend 
but it does not have support for that dtype.
```

## The Fix (3 Steps)

### 1. Open Terminal

### 2. Run These Commands

```bash
cd /Users/YOUR_USERNAME/Documents/ComfyUI/custom_nodes
# OR if using ComfyUI.app:
# cd /Applications/ComfyUI.app/Contents/Resources/ComfyUI/custom_nodes

git clone https://github.com/audiohacking/fp8-mps-metal.git
```

### 3. Restart ComfyUI

That's it! The patch will automatically load.

## Verification

Look for this message in ComfyUI's console when it starts:

```
======================================================================
✓ FP8 MPS Metal patch installed successfully!
======================================================================
```

If you see it, you're done! FP8 models (FLUX, SD3.5, etc.) will now work.

## Still Not Working?

1. **Wrong directory?** Double-check you cloned into ComfyUI's `custom_nodes` folder
2. **No success message?** Check ComfyUI's console for error messages
3. **Need help?** Open an issue: https://github.com/audiohacking/fp8-mps-metal/issues

## What This Does

Patches PyTorch's MPS backend to support FP8 operations that Apple doesn't provide natively. The patch is automatic and requires no code changes to your workflows.

## Technical Details

See `COMFYUI_INSTALL.md` for detailed installation guide and troubleshooting.
