"""
Monkey-patch torch._scaled_mm, Tensor.to(), and Tensor.copy_() to support FP8 MPS tensors.

Usage:
    import fp8_mps_patch
    fp8_mps_patch.install()   # patches torch._scaled_mm, Tensor.to(), and Tensor.copy_()
    fp8_mps_patch.uninstall() # restores original

ComfyUI integration: import this before loading models, and all
FLUX/SD3.5 FP8 scaled_mm calls will transparently use Metal GPU.
Float8_e4m3fn conversions on MPS will automatically use our quantization kernel.
"""

import torch

_original_scaled_mm = None
_original_tensor_to = None
_original_tensor_copy = None
_installed = False


def _is_fp8_dtype(dtype):
    """Helper to check if a dtype is FP8."""
    if hasattr(torch, 'float8_e4m3fn') and dtype == torch.float8_e4m3fn:
        return True
    if hasattr(torch, 'float8_e5m2') and dtype == torch.float8_e5m2:
        return True
    return False


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
    
    Handles three scenarios:
    1. FP8 tensor on CPU -> MPS device (raw bytes transfer as uint8)
    2. Float tensor -> FP8 on MPS (quantization via Metal kernel)
    3. FP8 tensor already on MPS (pass-through or dtype conversion)
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
    
    # Check if source tensor is FP8
    source_is_fp8 = False
    if hasattr(torch, 'float8_e4m3fn') and self.dtype == torch.float8_e4m3fn:
        source_is_fp8 = True
    elif hasattr(torch, 'float8_e5m2') and self.dtype == torch.float8_e5m2:
        source_is_fp8 = True
    
    # Check if target is MPS device
    target_device_is_mps = (
        (device and (device == "mps" or (isinstance(device, torch.device) and device.type == "mps"))) or
        (device is None and self.device.type == "mps")
    )
    
    # Check if target dtype is FP8
    target_is_fp8 = False
    target_fp8_dtype = None
    if hasattr(torch, 'float8_e4m3fn') and dtype == torch.float8_e4m3fn:
        target_is_fp8 = True
        target_fp8_dtype = torch.float8_e4m3fn
    elif hasattr(torch, 'float8_e5m2') and dtype == torch.float8_e5m2:
        target_is_fp8 = True
        target_fp8_dtype = torch.float8_e5m2
    
    # Scenario 1: FP8 tensor on CPU/other -> MPS device (raw bytes transfer)
    # This handles loading pre-quantized weights and moving them to MPS
    if source_is_fp8 and device and target_device_is_mps and self.device.type != "mps":
        # Transfer as uint8 (raw bytes), then view back as FP8
        # This avoids MPS's dtype conversion which doesn't support FP8
        tensor_u8 = self.view(torch.uint8)
        # Use original to() to transfer uint8 to MPS
        other_kwargs = {k: v for k, v in kwargs.items() if k not in ('device', 'dtype')}
        tensor_u8_mps = _original_tensor_to(tensor_u8, device, **other_kwargs)
        # View back as the original FP8 dtype
        result = tensor_u8_mps.view(self.dtype)
        # Apply dtype conversion if requested and different
        if dtype is not None and dtype != self.dtype:
            # This would be FP8->FP8 conversion, handle if needed
            if target_is_fp8:
                result = result.view(torch.uint8).view(target_fp8_dtype)
        return result
    
    # Scenario 2: Float/other tensor -> FP8 on MPS (quantization)
    # This handles on-the-fly quantization
    if target_device_is_mps and target_is_fp8 and not source_is_fp8:
        import fp8_mps_native
        
        # First move to MPS if not already there (using original method with non-FP8 dtype)
        if self.device.type != "mps":
            # Transfer without dtype change first
            other_kwargs = {k: v for k, v in kwargs.items() if k not in ('device', 'dtype')}
            tensor_mps = _original_tensor_to(self, device if device else "mps", **other_kwargs)
        else:
            tensor_mps = self
        
        # Use fp8_quantize to convert to uint8 (FP8 encoded)
        quantized_u8, scale = fp8_mps_native.fp8_quantize(tensor_mps)
        
        # View the uint8 as the requested FP8 dtype
        result = quantized_u8.view(target_fp8_dtype)
        
        # Note: The scale is not stored with the tensor, which matches PyTorch's
        # behavior for FP8 dtypes. Users must manage scales separately.
        return result
    
    # Scenario 3: FP8 on MPS, no device change needed
    # Just pass through or handle dtype conversion if needed
    if source_is_fp8 and self.device.type == "mps" and (device is None or target_device_is_mps):
        if dtype is None or dtype == self.dtype:
            # No conversion needed
            return self
        elif target_is_fp8:
            # FP8 to FP8 dtype conversion (e.g., e4m3fn to e5m2)
            return self.view(torch.uint8).view(target_fp8_dtype)
    
    # For all other conversions, use the original method
    return _original_tensor_to(self, *args, **kwargs)


def _metal_tensor_copy(self, src, non_blocking=False):
    """
    Drop-in replacement for Tensor.copy_() that handles FP8 tensors on MPS.
    
    Intercepts copy operations where either source or destination is FP8 on MPS,
    which would otherwise fail with "does not have support for that dtype".
    
    This handles multiple ComfyUI scenarios:
    1. FP8 source → FP8 destination on MPS (stochastic_rounding)
    2. Non-FP8 source → FP8 destination on MPS (dtype conversion during copy)
    """
    # Check if source is FP8 and destination (self) is MPS
    source_is_fp8 = _is_fp8_dtype(src.dtype)
    dest_is_fp8 = _is_fp8_dtype(self.dtype)
    dest_is_mps = self.device.type == "mps"
    
    # Scenario 1: FP8 source → FP8 destination on MPS
    # This is the original case: pre-quantized FP8 data being copied to MPS
    if source_is_fp8 and dest_is_mps and dest_is_fp8:
        # Convert FP8 to uint8 (raw bytes), copy, then view back as FP8
        # This avoids MPS's dtype conversion which doesn't support FP8
        
        # Ensure source is contiguous for proper byte interpretation
        src_contig = src.contiguous()
        
        # View source as uint8 (raw bytes)
        src_u8 = src_contig.view(torch.uint8)
        
        # View destination as uint8 for byte-level copy
        self_u8 = self.view(torch.uint8)
        
        # Copy uint8 bytes using original copy_ (which works for uint8)
        _original_tensor_copy(self_u8, src_u8, non_blocking=non_blocking)
        
        # Return self (copy_ returns self)
        return self
    
    # Scenario 2: Non-FP8 source → FP8 destination on MPS
    # This handles dtype conversion during copy, which MPS doesn't support natively
    if not source_is_fp8 and dest_is_fp8 and dest_is_mps:
        import fp8_mps_native
        
        # First, move source to MPS if needed (without dtype change)
        if src.device.type != "mps":
            src_mps = src.to(device="mps", dtype=src.dtype)
        else:
            src_mps = src
        
        # Quantize to FP8 using our Metal kernel
        # fp8_quantize uses automatic scaling: scale = 448.0 / max(abs(input))
        # where 448.0 is the max representable value in e4m3fn format
        quantized_u8, scale = fp8_mps_native.fp8_quantize(src_mps)
        
        # View destination as uint8 for byte-level copy
        self_u8 = self.view(torch.uint8)
        
        # Copy quantized bytes
        _original_tensor_copy(self_u8, quantized_u8, non_blocking=non_blocking)
        
        # Note: scale is discarded, matching PyTorch's behavior for FP8 dtypes.
        # FP8 tensors don't store scale information - users must manage scales
        # separately for operations like torch._scaled_mm that require them.
        # The automatic scaling from fp8_quantize ensures values fit in FP8 range.
        return self
    
    # Scenario 3: FP8 source → non-FP8 destination on MPS
    # This would require dequantization, which should fall through to original
    # and likely fail, but we let PyTorch handle it
    if source_is_fp8 and dest_is_mps and not dest_is_fp8:
        # This case needs dequantization which .copy_() doesn't handle
        # Fall through to original and let it fail or handle it
        return _original_tensor_copy(self, src, non_blocking=non_blocking)
    
    # For all other copy operations, use the original method
    return _original_tensor_copy(self, src, non_blocking=non_blocking)


def install():
    """Monkey-patch torch._scaled_mm, Tensor.to(), and Tensor.copy_() to use Metal FP8 kernels on MPS."""
    global _original_scaled_mm, _original_tensor_to, _original_tensor_copy, _installed
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
    
    # Patch Tensor.copy_() for FP8 tensor copies to MPS
    _original_tensor_copy = torch.Tensor.copy_
    torch.Tensor.copy_ = _metal_tensor_copy
    
    _installed = True


def uninstall():
    """Restore original torch._scaled_mm, Tensor.to(), and Tensor.copy_()."""
    global _original_scaled_mm, _original_tensor_to, _original_tensor_copy, _installed
    if not _installed:
        return

    if _original_scaled_mm is not None:
        torch._scaled_mm = _original_scaled_mm
        _original_scaled_mm = None
    
    if _original_tensor_to is not None:
        torch.Tensor.to = _original_tensor_to
        _original_tensor_to = None
    
    if _original_tensor_copy is not None:
        torch.Tensor.copy_ = _original_tensor_copy
        _original_tensor_copy = None
    
    _installed = False


def is_installed():
    """Check if the monkey-patch is active."""
    return _installed
