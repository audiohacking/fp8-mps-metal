"""
Backend selection for FP8 operations on MPS.

This module automatically selects between:
1. fp8_mps_native - PyTorch 2.10+ native torch.mps.compile_shader() (zero-copy, preferred)
2. fp8_metal - C++ extension fallback (requires compilation with pip install -e .)

For ComfyUI users with PyTorch < 2.10, the C++ extension will be used automatically
if available, otherwise operations will fail with a helpful error message.
"""

import torch

_backend = None
_backend_name = None
_init_error = None


def _init_backend():
    """Initialize and select the best available backend."""
    global _backend, _backend_name, _init_error
    
    if _backend is not None:
        return
    
    # Try native implementation first (PyTorch 2.10+)
    try:
        import fp8_mps_native
        if fp8_mps_native.is_available():
            _backend = fp8_mps_native
            _backend_name = "native"
            return
    except Exception as e:
        _init_error = f"Failed to load fp8_mps_native: {e}"
    
    # Fall back to C++ extension
    try:
        import fp8_metal
        _backend = fp8_metal
        _backend_name = "cpp"
        return
    except ImportError as e:
        pass  # Expected if not compiled
    except Exception as e:
        _init_error = f"Failed to load fp8_metal: {e}"
    
    # No backend available
    _backend = None
    _backend_name = None


def get_backend():
    """
    Get the active FP8 backend.
    
    Returns:
        tuple: (backend_module, backend_name) where backend_name is 'native', 'cpp', or None
    """
    _init_backend()
    return _backend, _backend_name


def is_available():
    """Check if any FP8 backend is available."""
    _init_backend()
    return _backend is not None


def get_error_message():
    """
    Get a helpful error message when no backend is available.
    
    Returns:
        str: Error message with instructions
    """
    _init_backend()
    
    if _backend is not None:
        return None
    
    msg = "No FP8 backend available for MPS.\n\n"
    
    # Check PyTorch version
    import torch
    msg += f"Your PyTorch version: {torch.__version__}\n\n"
    
    # Check if torch.mps.compile_shader exists
    has_compile_shader = hasattr(torch.mps, 'compile_shader')
    
    if not has_compile_shader:
        msg += "The native backend requires PyTorch 2.10+.\n"
        msg += "Solutions:\n\n"
        msg += "1. UPGRADE PyTorch (Recommended):\n"
        msg += "   pip install --upgrade torch torchvision\n\n"
        msg += "2. BUILD C++ Extension (For ComfyUI users who can't upgrade):\n"
        msg += "   # In ComfyUI/custom_nodes/fp8-mps-metal/\n"
        msg += "   pip install -e .\n"
        msg += "   # Then restart ComfyUI\n\n"
        msg += "   Note: This requires:\n"
        msg += "   - Xcode Command Line Tools: xcode-select --install\n"
        msg += "   - metal-cpp (auto-downloaded during build)\n\n"
    else:
        msg += "The native backend is available but failed to initialize.\n"
        msg += "Try building the C++ extension fallback:\n"
        msg += "   cd ComfyUI/custom_nodes/fp8-mps-metal/\n"
        msg += "   pip install -e .\n\n"
    
    if _init_error:
        msg += f"Debug info: {_init_error}\n"
    
    return msg


def fp8_scaled_mm(A, B, scale_a, scale_b):
    """FP8 scaled matrix multiplication - uses best available backend."""
    backend, name = get_backend()
    if backend is None:
        raise RuntimeError(get_error_message())
    return backend.fp8_scaled_mm(A, B, scale_a, scale_b)


def fp8_scaled_mm_auto(A, B, scale_a, scale_b):
    """Auto-select best FP8 matmul strategy - uses best available backend."""
    backend, name = get_backend()
    if backend is None:
        raise RuntimeError(get_error_message())
    
    # Only native backend has the _auto variant
    if name == "native":
        return backend.fp8_scaled_mm_auto(A, B, scale_a, scale_b)
    else:
        # C++ backend only has the regular version
        return backend.fp8_scaled_mm(A, B, scale_a, scale_b)


def fp8_dequantize(input, scale):
    """FP8 to float dequantization - uses best available backend."""
    backend, name = get_backend()
    if backend is None:
        raise RuntimeError(get_error_message())
    return backend.fp8_dequantize(input, scale)


def fp8_encode(input):
    """Float to FP8 encoding - uses best available backend."""
    backend, name = get_backend()
    if backend is None:
        raise RuntimeError(get_error_message())
    return backend.fp8_encode(input)


def fp8_quantize(input):
    """Float to FP8 quantization with scaling - uses best available backend."""
    backend, name = get_backend()
    if backend is None:
        raise RuntimeError(get_error_message())
    return backend.fp8_quantize(input)
