"""
Monkey-patch torch._scaled_mm to route FP8 MPS tensors through our Metal kernel.

Usage:
    import fp8_mps_patch
    fp8_mps_patch.install()   # patches torch._scaled_mm
    fp8_mps_patch.uninstall() # restores original

ComfyUI integration: import this before loading models, and all
FLUX/SD3.5 FP8 scaled_mm calls will transparently use Metal GPU.
"""

import torch

_original_scaled_mm = None
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


def install():
    """Monkey-patch torch._scaled_mm to use Metal FP8 kernel on MPS."""
    global _original_scaled_mm, _installed
    if _installed:
        return

    if hasattr(torch, "_scaled_mm"):
        _original_scaled_mm = torch._scaled_mm
        torch._scaled_mm = _metal_scaled_mm
        _installed = True
    else:
        raise RuntimeError("torch._scaled_mm not found — requires PyTorch 2.4+")


def uninstall():
    """Restore original torch._scaled_mm."""
    global _original_scaled_mm, _installed
    if not _installed:
        return

    if _original_scaled_mm is not None:
        torch._scaled_mm = _original_scaled_mm
        _original_scaled_mm = None
    _installed = False


def is_installed():
    """Check if the monkey-patch is active."""
    return _installed
