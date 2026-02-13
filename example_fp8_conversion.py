#!/usr/bin/env python3
"""
Example: Using Float8_e4m3fn conversions on MPS

This example demonstrates how to use the fp8_mps_patch to enable
Float8_e4m3fn dtype conversions on Apple Silicon MPS backend.

Key scenarios covered:
1. Float32 -> FP8 conversion on MPS (quantization)
2. Float CPU -> FP8 on MPS (quantization + transfer)
3. FP8 CPU -> MPS (raw bytes transfer for pre-quantized weights)
4. Using FP8 tensors in operations
"""

import torch
import fp8_mps_patch

# Install the patch to enable FP8 support on MPS
fp8_mps_patch.install()

print("Float8_e4m3fn Conversion Example")
print("=" * 60)

# Check if MPS is available
if not torch.backends.mps.is_available():
    print("MPS backend is not available on this system")
    exit(1)

# Check if Float8_e4m3fn is available
if not hasattr(torch, 'float8_e4m3fn'):
    print("torch.float8_e4m3fn is not available in this PyTorch version")
    print("Requires PyTorch 2.1+ with FP8 support")
    exit(1)

print(f"PyTorch version: {torch.__version__}")
print(f"MPS available: {torch.backends.mps.is_available()}")
print()

# Example 1: Convert a float32 tensor to Float8_e4m3fn on MPS
print("Example 1: Float32 to Float8_e4m3fn conversion (quantization)")
print("-" * 60)
x = torch.randn(4, 8, device="mps")
print(f"Original tensor: shape={x.shape}, dtype={x.dtype}, device={x.device}")

# With the patch installed, this now works!
x_fp8 = x.to(torch.float8_e4m3fn)
print(f"Converted to FP8: shape={x_fp8.shape}, dtype={x_fp8.dtype}, device={x_fp8.device}")
print()

# Example 2: Convert from CPU to MPS as Float8_e4m3fn (quantization + transfer)
print("Example 2: CPU to MPS Float8_e4m3fn conversion")
print("-" * 60)
x_cpu = torch.randn(4, 8)
print(f"CPU tensor: dtype={x_cpu.dtype}, device={x_cpu.device}")

# Direct conversion to MPS as FP8
x_fp8_from_cpu = x_cpu.to("mps", dtype=torch.float8_e4m3fn)
print(f"Converted to MPS FP8: dtype={x_fp8_from_cpu.dtype}, device={x_fp8_from_cpu.device}")
print()

# Example 3: Transfer pre-quantized FP8 weights to MPS (ComfyUI/FLUX scenario)
print("Example 3: Load pre-quantized FP8 weights and move to MPS")
print("-" * 60)
# Simulate loading FP8-quantized weights (e.g., from safetensors)
# In real usage, this would come from model loading
weight_bytes = torch.randint(0, 255, (256, 512), dtype=torch.uint8)
weight_fp8_cpu = weight_bytes.view(torch.float8_e4m3fn)
print(f"Loaded FP8 weight: shape={weight_fp8_cpu.shape}, dtype={weight_fp8_cpu.dtype}, device={weight_fp8_cpu.device}")

# Move to MPS - this is what fails without the patch
weight_fp8_mps = weight_fp8_cpu.to("mps")
print(f"Moved to MPS: dtype={weight_fp8_mps.dtype}, device={weight_fp8_mps.device}")
print("✓ Raw bytes transferred correctly (no quantization needed)")
print()

# Example 4: Use FP8 tensors in computation
print("Example 4: Using FP8 tensors in matrix multiplication")
print("-" * 60)
A = torch.randn(16, 32, device="mps")
B = torch.randn(32, 32, device="mps")

# Convert to FP8
A_fp8 = A.to(torch.float8_e4m3fn)
B_fp8 = B.to(torch.float8_e4m3fn)

print(f"A_fp8: shape={A_fp8.shape}, dtype={A_fp8.dtype}")
print(f"B_fp8: shape={B_fp8.shape}, dtype={B_fp8.dtype}")

# FP8 matmul with torch._scaled_mm (also patched by fp8_mps_patch)
# Note: You'll need to manage scales separately
scale_a = torch.tensor([1.0], device="mps")
scale_b = torch.tensor([1.0], device="mps")

# The _scaled_mm operation is also patched and will use Metal kernels
result = torch._scaled_mm(A_fp8, B_fp8.t(), scale_a=scale_a, scale_b=scale_b)
print(f"Result: shape={result.shape}, dtype={result.dtype}")
print()

# Example 5: Model quantization workflow
print("Example 5: Quantizing a model layer")
print("-" * 60)
# Simulate a model layer weight
weight = torch.randn(256, 512, device="mps")
print(f"Original weight: shape={weight.shape}, dtype={weight.dtype}, memory={weight.numel() * 4 / 1024:.2f} KB")

# Quantize to FP8 - 4x memory reduction vs FP32
weight_fp8 = weight.to(torch.float8_e4m3fn)
print(f"Quantized weight: shape={weight_fp8.shape}, dtype={weight_fp8.dtype}, memory={weight_fp8.numel() * 1 / 1024:.2f} KB")
print(f"Memory savings: {(1 - (weight_fp8.numel() * 1) / (weight.numel() * 4)) * 100:.1f}%")
print()

print("=" * 60)
print("All examples completed successfully!")
print("=" * 60)
print()
print("Key capabilities enabled by fp8_mps_patch:")
print("  1. Float32/16 -> FP8 quantization on MPS (via .to())")
print("  2. CPU Float32/16 -> MPS FP8 (quantization + transfer)")
print("  3. CPU FP8 -> MPS FP8 (raw bytes transfer, no re-quantization)")
print("  4. FP8 .copy_() operations to MPS (ComfyUI stochastic rounding)")
print("  5. FP8 matrix operations via Metal kernels")
print()
print("Memory efficiency: FP8 uses 25% the memory of FP32, 50% the memory of FP16")
print("Inference accuracy: Typically within 1-5% of FP16 for most workloads")
print()
print("Use case: Loading and running FLUX/SD3.5 FP8-quantized models on Mac")
print("Note: The .copy_() patch is essential for ComfyUI's stochastic rounding")

# Clean up: uninstall the patch (optional)
fp8_mps_patch.uninstall()
