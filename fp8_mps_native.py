"""
FP8 Metal kernels using PyTorch's native torch.mps.compile_shader() API.

Zero-copy dispatch: kernels run directly on MPS tensor buffers.
No C++ extension needed — pure Python + Metal shader source.

This replaces the C++ bridge approach for dramatically better performance
since it avoids MPS→CPU→Metal→CPU→MPS buffer copies.
"""

import torch
import os

_lib = None
_SHADER_SOURCE = None


def _load_shader_source():
    """Load the Metal shader source from fp8_matmul.metal."""
    global _SHADER_SOURCE
    if _SHADER_SOURCE is not None:
        return _SHADER_SOURCE

    shader_path = os.path.join(os.path.dirname(__file__), "fp8_matmul.metal")
    with open(shader_path, "r") as f:
        _SHADER_SOURCE = f.read()
    return _SHADER_SOURCE


def _get_lib():
    """Get or create the compiled shader library (singleton)."""
    global _lib
    if _lib is not None:
        return _lib

    # Check if torch.mps.compile_shader is available (requires PyTorch 2.10+)
    if not hasattr(torch.mps, 'compile_shader'):
        raise AttributeError(
            "torch.mps.compile_shader is not available in your PyTorch version.\n"
            "\n"
            "This feature requires PyTorch 2.10 or later.\n"
            "\n"
            "Solutions:\n"
            "1. Upgrade PyTorch to 2.10+:\n"
            "   pip install --upgrade torch torchvision\n"
            "\n"
            "2. Use the C++ extension fallback (requires compilation):\n"
            "   cd /path/to/fp8-mps-metal\n"
            "   pip install -e .\n"
            "   Then use 'import fp8_metal' instead of 'fp8_mps_native'\n"
            "\n"
            f"Your current PyTorch version: {torch.__version__}"
        )

    source = _load_shader_source()
    _lib = torch.mps.compile_shader(source)
    return _lib


def fp8_scaled_mm(A: torch.Tensor, B: torch.Tensor,
                  scale_a: torch.Tensor, scale_b: torch.Tensor) -> torch.Tensor:
    """
    FP8 scaled matrix multiplication on Metal GPU.

    A: (M, K) uint8 — FP8 e4m3fn encoded, row-major
    B: (N, K) uint8 — FP8 e4m3fn encoded, row-major (B is pre-transposed)
    scale_a: per-tensor [1] or per-row [M] float32
    scale_b: per-tensor [1] or per-row [N] float32

    Returns: (M, N) float32 on MPS
    """
    lib = _get_lib()

    assert A.dtype == torch.uint8 and B.dtype == torch.uint8
    assert A.is_contiguous() and B.is_contiguous()

    M, K = A.shape
    N = B.shape[0]
    assert B.shape[1] == K

    # Ensure tensors are on MPS
    if A.device.type != "mps":
        A = A.to("mps")
    if B.device.type != "mps":
        B = B.to("mps")

    # Ensure scales are on MPS and float32
    scale_a = scale_a.to(device="mps", dtype=torch.float32).contiguous()
    scale_b = scale_b.to(device="mps", dtype=torch.float32).contiguous()

    # Determine scale mode: 0=per-tensor, 1=per-channel
    scale_mode = 0 if (scale_a.numel() == 1 and scale_b.numel() == 1) else 1

    # Output tensor
    C = torch.empty(M, N, dtype=torch.float32, device="mps")

    if M == 1:
        # Vecmat kernel: SIMD reduction, 32 threads per output row
        total_threads = N * 32
        threads_per_group = 256
        lib.fp8_scaled_vecmat_kernel(
            A, B, C, scale_a, scale_b,
            N, K, scale_mode,
            threads=(total_threads,), group_size=(threads_per_group,),
                    )
    else:
        # General 2D matmul
        lib.fp8_scaled_matmul_kernel(
            A, B, C, scale_a, scale_b,
            M, N, K, scale_mode,
            threads=(N, M), group_size=(16, 16),
                    )

    return C


def fp8_dequantize(input: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    FP8 → half dequantization on Metal GPU.

    input: uint8 tensor (FP8 e4m3fn encoded)
    scale: scalar float32 tensor
    Returns: float16 tensor on MPS, scaled
    """
    lib = _get_lib()

    if input.device.type != "mps":
        input = input.to("mps")

    count = input.numel()
    output = torch.empty(input.shape, dtype=torch.float16, device="mps")

    lib.fp8_to_half_kernel(
        input.contiguous().view(-1), output.view(-1),
        count,
        threads=(count,), group_size=(256,),
            )

    # Apply scale
    scale_val = scale.to(device="mps", dtype=torch.float16)
    output = output * scale_val

    return output


def fp8_encode(input: torch.Tensor):
    """
    Float → FP8 encoding on Metal GPU (without scaling).
    
    This function encodes float values to FP8 format preserving value semantics.
    Values are clamped to FP8 range [-448, 448] but NOT scaled.
    
    Use this for .to(torch.float8_e4m3fn) conversions and .copy_() operations
    where you want the FP8 tensor to preserve the original value magnitudes.

    input: float32 tensor
    Returns: uint8 tensor on MPS (FP8 encoded)
    """
    lib = _get_lib()

    inp = input.to(device="mps", dtype=torch.float32).contiguous()
    count = inp.numel()

    output = torch.empty(inp.shape, dtype=torch.uint8, device="mps")

    # Encode to FP8 without pre-scaling
    # The Metal kernel will clamp values to FP8 range [-448, 448]
    lib.float_to_fp8_kernel(
        inp.view(-1), output.view(-1),
        count,
        threads=(count,), group_size=(256,),
            )

    return output


def fp8_quantize(input: torch.Tensor):
    """
    Float → FP8 quantization on Metal GPU (with automatic scaling).
    
    This function scales the input to use the full FP8 range for maximum precision.
    Use this for torch._scaled_mm operations where separate scales are managed.

    input: float32 tensor
    Returns: (uint8 tensor on MPS, inverse_scale on MPS)
    """
    lib = _get_lib()

    inp = input.to(device="mps", dtype=torch.float32).contiguous()
    count = inp.numel()

    # Compute scale: max_fp8 / max(abs(input))
    amax = inp.abs().max().item()
    max_fp8 = 448.0
    scale = max_fp8 / amax if amax > 0 else 1.0

    # Scale input before quantization
    scaled = (inp * scale).contiguous()

    output = torch.empty(inp.shape, dtype=torch.uint8, device="mps")

    lib.float_to_fp8_kernel(
        scaled.view(-1), output.view(-1),
        count,
        threads=(count,), group_size=(256,),
            )

    inv_scale = torch.tensor([1.0 / scale], dtype=torch.float32, device="mps")
    return output, inv_scale


def fp8_scaled_mm_auto(A: torch.Tensor, B: torch.Tensor,
                       scale_a: torch.Tensor, scale_b: torch.Tensor) -> torch.Tensor:
    """
    Auto-select best FP8 matmul strategy based on dimensions.

    Benchmarks on M4 Pro show the fused kernel (in-register FP8 decode + accumulate)
    consistently beats dequant+native-matmul because:
    - The dequant pass adds per-element scale multiply overhead
    - The fused kernel avoids writing/reading intermediate FP16 buffers
    - For M=1 (single-token), the SIMD vecmat kernel is especially efficient

    Falls back to dequant+matmul only for very large M where the 2D kernel's
    per-thread decode cost exceeds the memory bandwidth savings.
    """
    M = A.shape[0]
    if M <= 16:
        return fp8_scaled_mm(A, B, scale_a, scale_b)
    return fp8_scaled_mm_fast(A, B, scale_a, scale_b)


def fp8_scaled_mm_fast(A: torch.Tensor, B: torch.Tensor,
                       scale_a: torch.Tensor, scale_b: torch.Tensor) -> torch.Tensor:
    """
    Fast FP8 scaled matmul: dequant to FP16 on GPU, then native FP16 matmul.

    This is faster than the fused kernel because it leverages the hardware's
    optimized FP16 matrix multiply engine (AMX/GPU matmul units).

    A: (M, K) uint8 — FP8 e4m3fn encoded
    B: (N, K) uint8 — FP8 e4m3fn encoded (pre-transposed)
    scale_a: per-tensor scale
    scale_b: per-tensor scale
    Returns: (M, N) float32 on MPS
    """
    lib = _get_lib()

    # Ensure on MPS
    if A.device.type != "mps":
        A = A.to("mps")
    if B.device.type != "mps":
        B = B.to("mps")

    M, K = A.shape
    N = B.shape[0]

    # Dequant A to FP16
    A_f16 = torch.empty(M, K, dtype=torch.float16, device="mps")
    count_a = A.numel()
    lib.fp8_to_half_kernel(
        A.contiguous().view(-1), A_f16.view(-1),
        count_a,
        threads=(count_a,), group_size=(256,),
    )

    # Dequant B to FP16
    B_f16 = torch.empty(N, K, dtype=torch.float16, device="mps")
    count_b = B.numel()
    lib.fp8_to_half_kernel(
        B.contiguous().view(-1), B_f16.view(-1),
        count_b,
        threads=(count_b,), group_size=(256,),
    )

    # Apply scales BEFORE matmul to prevent FP16 overflow
    # Raw FP8 values can be up to 448, so K-dim dot products overflow FP16 (max 65504)
    sa = scale_a.to(device="mps", dtype=torch.float16)
    sb = scale_b.to(device="mps", dtype=torch.float16)
    A_f16 = A_f16 * sa
    B_f16 = B_f16 * sb

    # Native FP16 matmul (uses hardware matrix multiply engine)
    # B is (N, K), we need A @ B^T = (M, K) @ (K, N) = (M, N)
    C = A_f16 @ B_f16.T

    return C.float()
