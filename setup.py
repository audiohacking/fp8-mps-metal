"""
Build system for fp8_metal PyTorch extension.

Metal shader is compiled at runtime via MTL::Device::newLibrary(source)
since the `metal` compiler requires Xcode (not available with CLT-only).

Pattern: mpsparse/iterative/cgd/setup.py
"""

import os
import subprocess
import urllib.request
import zipfile
import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

current_dir = os.path.dirname(os.path.abspath(__file__))
torch_lib_dir = os.path.join(os.path.dirname(torch.__file__), "lib")
metal_cpp_path = os.path.join(current_dir, "metal-cpp")

# Auto-download metal-cpp v26 if not present (from metalQwen3 CMakeLists.txt)
if not os.path.exists(metal_cpp_path):
    url = "https://developer.apple.com/metal/cpp/files/metal-cpp_macOS26_iOS26-beta2.zip"
    zip_path = os.path.join(current_dir, "metal-cpp.zip")
    print(f"Downloading metal-cpp from {url}...")
    urllib.request.urlretrieve(url, zip_path)
    print("Extracting metal-cpp...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(current_dir)
    os.remove(zip_path)
    if not os.path.exists(metal_cpp_path):
        raise RuntimeError("metal-cpp extraction failed — expected directory: " + metal_cpp_path)
    print("metal-cpp v26 ready.")

setup(
    name="fp8_metal",
    ext_modules=[
        CppExtension(
            name="fp8_metal",
            sources=["fp8_bridge.cpp"],
            include_dirs=[metal_cpp_path],
            extra_compile_args=[
                "-std=c++17",
                f'-DSHADER_DIR="{current_dir}"',
            ],
            extra_link_args=[
                "-framework", "Metal",
                "-framework", "Foundation",
                "-framework", "QuartzCore",
                f"-Wl,-rpath,{torch_lib_dir}",
            ],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
