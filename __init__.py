"""
ComfyUI Custom Node: FP8 MPS Metal Support

This custom node automatically installs patches to enable FP8 (Float8_e4m3fn) 
support on Apple Silicon MPS backend for ComfyUI.

Installation:
    Copy this entire repository to ComfyUI/custom_nodes/fp8-mps-metal/

The patch will be automatically installed when ComfyUI loads this custom node.
"""

import sys
import os

# Add current directory to path so we can import fp8_mps_patch
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import and install the patch
try:
    import fp8_mps_patch
    
    # Install the patch automatically
    if not fp8_mps_patch.is_installed():
        fp8_mps_patch.install()
        print("\n" + "=" * 70)
        print("✓ FP8 MPS Metal patch installed successfully!")
        print("=" * 70)
        print("Float8_e4m3fn operations on MPS are now supported.")
        print("This enables:")
        print("  • FP8 model weight loading (FLUX/SD3.5)")
        print("  • FP8 quantization on MPS")
        print("  • FP8 stochastic rounding (.copy_() operations)")
        print("  • FP8 matrix multiplication via Metal kernels")
        print("=" * 70 + "\n")
    else:
        print("[fp8-mps-metal] Patch already installed")
        
except Exception as e:
    print("\n" + "!" * 70)
    print("⚠ WARNING: Failed to install FP8 MPS Metal patch")
    print("!" * 70)
    print(f"Error: {e}")
    print("FP8 operations on MPS will not work without this patch.")
    print("Please report this issue at:")
    print("https://github.com/audiohacking/fp8-mps-metal/issues")
    print("!" * 70 + "\n")
    import traceback
    traceback.print_exc()

# ComfyUI requires NODE_CLASS_MAPPINGS to recognize this as a custom node
# We don't add any nodes, just install the patch
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Export for ComfyUI
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
