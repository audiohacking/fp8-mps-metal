"""
Monkey-patch torch._scaled_mm and Tensor.to() to support FP8 MPS tensors.

Usage:
    import fp8_mps_patch
    fp8_mps_patch.install()   # patches torch._scaled_mm and Tensor.to()
    fp8_mps_patch.uninstall() # restores original

ComfyUI integration: import this before loading models, and all
FLUX/SD3.5 FP8 scaled_mm calls will transparently use Metal GPU.
Float8_e4m3fn conversions on MPS will automatically use our quantization kernel.
"""

import torch

_original_scaled_mm = None
_original_tensor_to = None
_installed = False


def _metal_scaled_mm(input, other, *, out_dtype=None, scale_a=None, scale_b=None, bias=None, scale_result=None, use_fast_accum=False):
    """
    Drop-in replacement for torch._scaled_mm that handles FP8 on MPS.

    torch._scaled_mm signature: (input, other, *, out_dtype, scale_a, scale_b, bias, scale_result, use_fast_accum)
    - input: (M, K) — activation tensor (FP8 or float)
    - other: (K, N) — weight tensor (FP8 or float), column-major (NOT transposed like our kernel)
    - scale_a: per-tensor or per-row scale for input
    - scale_b: per-tensor or per-row scale for other
    """
    # Only intercept for MPS device + FP8/uint8 tensors
    is_mps = input.device.type == "mps"
    is_fp8 = input.dtype in (torch.uint8, torch.float8_e4m3fn, torch.float8_e5m2)

    if not (is_mps and is_fp8):
        return _original_scaled_mm(
            input, other, out_dtype=out_dtype, scale_a=scale_a,
            scale_b=scale_b, bias=bias, scale_result=scale_result,
            use_fast_accum=use_fast_accum,
        )

    import fp8_mps_native

    # Handle FP8 dtype tensors by viewing as uint8
    if input.dtype != torch.uint8:
        input = input.view(torch.uint8)
    if other.dtype != torch.uint8:
        other = other.view(torch.uint8)

    # torch._scaled_mm expects other as (K, N), our kernel wants B as (N, K)
    # other is (K, N), we need (N, K) = other.T which is contiguous in row-major
    B = other.t().contiguous()

    # Default scales
    if scale_a is None:
        scale_a = torch.tensor([1.0], device=input.device)
    if scale_b is None:
        scale_b = torch.tensor([1.0], device=input.device)

    result = fp8_mps_native.fp8_scaled_mm_auto(input, B, scale_a, scale_b)

    # Apply bias if provided
    if bias is not None:
        result = result + bias

    # Apply result scaling if provided
    if scale_result is not None:
        result = result * scale_result

    # Cast to requested output dtype
    if out_dtype is not None:
        result = result.to(out_dtype)

    return result


def _metal_tensor_to(self, *args, **kwargs):
    """
    Drop-in replacement for Tensor.to() that handles FP8 conversions on MPS.
    
    Intercepts conversions to torch.float8_e4m3fn or torch.float8_e5m2 on MPS
    and routes them through our Metal quantization kernel instead of the
    unsupported native MPS cast.
    """
    # Parse arguments to detect dtype conversion
    dtype = kwargs.get('dtype')
    device = kwargs.get('device')
    
    # Handle positional args: to(dtype), to(device), to(device, dtype), etc.
    if not dtype and args:
        for arg in args:
            if isinstance(arg, torch.dtype):
                dtype = arg
            elif isinstance(arg, (torch.device, str)):
                if device is None:
                    device = arg
    
    # Check if this is an FP8 conversion on MPS
    target_device_is_mps = (
        (device and (device == "mps" or (isinstance(device, torch.device) and device.type == "mps"))) or
        (device is None and self.device.type == "mps")
    )
    
    # Check if this is an FP8 conversion - handle both e4m3fn and e5m2
    is_fp8_conversion = False
    if hasattr(torch, 'float8_e4m3fn') and dtype == torch.float8_e4m3fn:
        is_fp8_conversion = True
    elif hasattr(torch, 'float8_e5m2') and dtype == torch.float8_e5m2:
        is_fp8_conversion = True
    
    if target_device_is_mps and is_fp8_conversion:
        import fp8_mps_native
        
        # Quantize the tensor using our Metal kernel
        # First ensure it's on MPS and float32
        tensor_mps = self if self.device.type == "mps" else self.to("mps")
        
        # Use fp8_quantize to convert to uint8 (FP8 encoded)
        quantized_u8, scale = fp8_mps_native.fp8_quantize(tensor_mps)
        
        # View the uint8 as the requested FP8 dtype
        # This is safe because FP8 is stored as uint8 in PyTorch
        result = quantized_u8.view(dtype)
        
        # Note: The scale is not stored with the tensor, which matches PyTorch's
        # behavior for FP8 dtypes. Users must manage scales separately.
        return result
    
    # For all other conversions, use the original method
    return _original_tensor_to(self, *args, **kwargs)


def install():
    """Monkey-patch torch._scaled_mm and Tensor.to() to use Metal FP8 kernels on MPS."""
    global _original_scaled_mm, _original_tensor_to, _installed
    if _installed:
        return

    if hasattr(torch, "_scaled_mm"):
        _original_scaled_mm = torch._scaled_mm
        torch._scaled_mm = _metal_scaled_mm
    else:
        raise RuntimeError("torch._scaled_mm not found — requires PyTorch 2.4+")
    
    # Patch Tensor.to() for FP8 dtype conversions
    _original_tensor_to = torch.Tensor.to
    torch.Tensor.to = _metal_tensor_to
    
    _installed = True


def uninstall():
    """Restore original torch._scaled_mm and Tensor.to()."""
    global _original_scaled_mm, _original_tensor_to, _installed
    if not _installed:
        return

    if _original_scaled_mm is not None:
        torch._scaled_mm = _original_scaled_mm
        _original_scaled_mm = None
    
    if _original_tensor_to is not None:
        torch.Tensor.to = _original_tensor_to
        _original_tensor_to = None
    
    _installed = False


def is_installed():
    """Check if the monkey-patch is active."""
    return _installed
