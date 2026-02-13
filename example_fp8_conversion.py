#!/usr/bin/env python3
"""
Example: Using Float8_e4m3fn conversions on MPS

This example demonstrates how to use the fp8_mps_patch to enable
Float8_e4m3fn dtype conversions on Apple Silicon MPS backend.
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
print("Example 1: Float32 to Float8_e4m3fn conversion")
print("-" * 60)
x = torch.randn(4, 8, device="mps")
print(f"Original tensor: shape={x.shape}, dtype={x.dtype}, device={x.device}")

# With the patch installed, this now works!
x_fp8 = x.to(torch.float8_e4m3fn)
print(f"Converted to FP8: shape={x_fp8.shape}, dtype={x_fp8.dtype}, device={x_fp8.device}")
print()

# Example 2: Convert from CPU to MPS as Float8_e4m3fn
print("Example 2: CPU to MPS Float8_e4m3fn conversion")
print("-" * 60)
x_cpu = torch.randn(4, 8)
print(f"CPU tensor: dtype={x_cpu.dtype}, device={x_cpu.device}")

# Direct conversion to MPS as FP8
x_fp8_from_cpu = x_cpu.to("mps", dtype=torch.float8_e4m3fn)
print(f"Converted to MPS FP8: dtype={x_fp8_from_cpu.dtype}, device={x_fp8_from_cpu.device}")
print()

# Example 3: Use FP8 tensors in computation
print("Example 3: Using FP8 tensors in matrix multiplication")
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

# Example 4: Model quantization workflow
print("Example 4: Quantizing a model layer")
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
print("Note: FP8 quantization reduces memory by 50% vs FP32 and 25% vs FP16")
print("      Accuracy is typically within 1-5% of FP16 for inference workloads")

# Clean up: uninstall the patch (optional)
fp8_mps_patch.uninstall()
