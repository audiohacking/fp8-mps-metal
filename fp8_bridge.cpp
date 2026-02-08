/**
 * FP8 Metal PyTorch Extension Bridge
 *
 * Singleton Metal context that compiles fp8_matmul.metal at runtime
 * and dispatches FP8 scaled matmul / dequant / quant kernels.
 *
 * Patterns from:
 *   mpsparse/spmv.cpp      — metal-cpp includes, device/queue/pipeline singleton, PYBIND11_MODULE
 *   metalQwen3/MetalContext — pipeline caching, library loading
 */

#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

#include <torch/extension.h>
#include <fstream>
#include <sstream>
#include <string>
#include <stdexcept>
#include <mutex>

// ─── Read shader source from file ───────────────────────────────────────────

static std::string read_file(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        throw std::runtime_error("Cannot open shader file: " + path);
    }
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// ─── Singleton Metal Context ────────────────────────────────────────────────

class FP8MetalContext {
public:
    static FP8MetalContext& instance() {
        static FP8MetalContext ctx;
        return ctx;
    }

    MTL::Device* device;
    MTL::CommandQueue* queue;

    // Cached pipeline states
    MTL::ComputePipelineState* pso_matmul;
    MTL::ComputePipelineState* pso_vecmat;
    MTL::ComputePipelineState* pso_to_half;
    MTL::ComputePipelineState* pso_to_fp8;

private:
    std::once_flag init_flag_;

    FP8MetalContext() : device(nullptr), queue(nullptr),
                         pso_matmul(nullptr), pso_vecmat(nullptr),
                         pso_to_half(nullptr), pso_to_fp8(nullptr) {}

    void ensure_init() {
        std::call_once(init_flag_, [this]() {
            device = MTL::CreateSystemDefaultDevice();
            if (!device) throw std::runtime_error("No Metal device found");
            queue = device->newCommandQueue();
            if (!queue) throw std::runtime_error("Failed to create command queue");
            load_pipelines();
        });
    }

    void load_pipelines() {
        // Find shader file relative to this .so
        // The .metal file is installed alongside the extension
        // Try multiple paths: next to the .so, or in the source dir
        std::string shader_src;
        std::vector<std::string> search_paths;

        // Get the directory of this source file (compile-time)
        std::string src_dir = SHADER_DIR;
        search_paths.push_back(src_dir + "/fp8_matmul.metal");
        search_paths.push_back("./fp8_matmul.metal");
        search_paths.push_back("fp8_matmul.metal");

        bool found = false;
        for (const auto& p : search_paths) {
            std::ifstream test(p);
            if (test.good()) {
                shader_src = read_file(p);
                found = true;
                break;
            }
        }
        if (!found) {
            throw std::runtime_error(
                "Cannot find fp8_matmul.metal — searched: " + search_paths[0]);
        }

        // Compile from source at runtime
        NS::Error* error = nullptr;
        NS::String* src_ns = NS::String::string(shader_src.c_str(), NS::UTF8StringEncoding);
        MTL::CompileOptions* opts = MTL::CompileOptions::alloc()->init();
        MTL::Library* library = device->newLibrary(src_ns, opts, &error);
        opts->release();

        if (!library) {
            std::string msg = "Metal shader compilation failed";
            if (error) {
                msg += ": ";
                msg += error->localizedDescription()->utf8String();
            }
            throw std::runtime_error(msg);
        }

        // Load each kernel
        auto load_kernel = [&](const char* name) -> MTL::ComputePipelineState* {
            NS::String* ns_name = NS::String::string(name, NS::UTF8StringEncoding);
            MTL::Function* fn = library->newFunction(ns_name);
            if (!fn) {
                throw std::runtime_error(std::string("Kernel not found: ") + name);
            }
            NS::Error* pso_err = nullptr;
            MTL::ComputePipelineState* pso = device->newComputePipelineState(fn, &pso_err);
            fn->release();
            if (!pso) {
                std::string err_msg = std::string("Pipeline creation failed for: ") + name;
                if (pso_err) {
                    err_msg += ": ";
                    err_msg += pso_err->localizedDescription()->utf8String();
                }
                throw std::runtime_error(err_msg);
            }
            return pso;
        };

        pso_matmul = load_kernel("fp8_scaled_matmul_kernel");
        pso_vecmat = load_kernel("fp8_scaled_vecmat_kernel");
        pso_to_half = load_kernel("fp8_to_half_kernel");
        pso_to_fp8 = load_kernel("float_to_fp8_kernel");

        library->release();
    }

public:
    void init() { ensure_init(); }
};


// ─── Helper: get raw uint8_t pointer from a tensor ──────────────────────────

static const uint8_t* u8_ptr(const torch::Tensor& t) {
    return static_cast<const uint8_t*>(t.data_ptr());
}

static const float* f32_ptr(const torch::Tensor& t) {
    return static_cast<const float*>(t.data_ptr());
}


// ─── fp8_scaled_mm ──────────────────────────────────────────────────────────
// A: (M, K) uint8,  B: (N, K) uint8 (B is transposed: B^T is K×N)
// Returns: (M, N) float32

torch::Tensor fp8_scaled_mm(
    torch::Tensor A,       // (M, K) uint8
    torch::Tensor B,       // (N, K) uint8  — transposed layout
    torch::Tensor scale_a, // per-tensor [1] or per-row [M]
    torch::Tensor scale_b  // per-tensor [1] or per-row [N]
) {
    auto& ctx = FP8MetalContext::instance();
    ctx.init();

    TORCH_CHECK(A.dtype() == torch::kUInt8, "A must be uint8 (FP8 encoded)");
    TORCH_CHECK(B.dtype() == torch::kUInt8, "B must be uint8 (FP8 encoded)");
    TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
    TORCH_CHECK(B.is_contiguous(), "B must be contiguous");

    // Move scale tensors to CPU float32 for Metal buffer creation
    torch::Tensor sa = scale_a.to(torch::kCPU).to(torch::kFloat32).contiguous();
    torch::Tensor sb = scale_b.to(torch::kCPU).to(torch::kFloat32).contiguous();

    // Move A and B to CPU for SharedStorageMode buffers
    torch::Tensor A_cpu = A.to(torch::kCPU).contiguous();
    torch::Tensor B_cpu = B.to(torch::kCPU).contiguous();

    uint32_t M = A_cpu.size(0);
    uint32_t K = A_cpu.size(1);
    uint32_t N = B_cpu.size(0);

    TORCH_CHECK(B_cpu.size(1) == K, "K dimension mismatch between A and B");

    // Determine scale mode: 0 = per-tensor (numel==1), 1 = per-channel
    uint32_t scale_mode = (sa.numel() == 1 && sb.numel() == 1) ? 0 : 1;

    // Create Metal buffers with SharedStorageMode (zero-copy on Apple Silicon)
    auto* dev = ctx.device;
    MTL::Buffer* buf_A = dev->newBuffer(u8_ptr(A_cpu), M * K * sizeof(uint8_t), MTL::ResourceStorageModeShared);
    MTL::Buffer* buf_B = dev->newBuffer(u8_ptr(B_cpu), N * K * sizeof(uint8_t), MTL::ResourceStorageModeShared);
    MTL::Buffer* buf_C = dev->newBuffer(M * N * sizeof(float), MTL::ResourceStorageModeShared);
    MTL::Buffer* buf_sa = dev->newBuffer(f32_ptr(sa), sa.numel() * sizeof(float), MTL::ResourceStorageModeShared);
    MTL::Buffer* buf_sb = dev->newBuffer(f32_ptr(sb), sb.numel() * sizeof(float), MTL::ResourceStorageModeShared);

    MTL::CommandBuffer* cmd = ctx.queue->commandBuffer();
    MTL::ComputeCommandEncoder* enc = cmd->computeCommandEncoder();

    if (M == 1) {
        // Use optimized vecmat kernel for single-token inference
        enc->setComputePipelineState(ctx.pso_vecmat);
        enc->setBuffer(buf_A, 0, 0);   // x vector
        enc->setBuffer(buf_B, 0, 1);   // W matrix
        enc->setBuffer(buf_C, 0, 2);   // output
        enc->setBuffer(buf_sa, 0, 3);  // scale_x
        enc->setBuffer(buf_sb, 0, 4);  // scale_w
        enc->setBytes(&N, sizeof(uint32_t), 5);
        enc->setBytes(&K, sizeof(uint32_t), 6);
        enc->setBytes(&scale_mode, sizeof(uint32_t), 7);

        // One SIMD group (32 threads) per output row
        uint32_t total_threads = N * 32;
        uint32_t threads_per_group = 256;  // 8 SIMD groups per threadgroup
        MTL::Size grid(total_threads, 1, 1);
        MTL::Size tg(threads_per_group, 1, 1);
        enc->dispatchThreads(grid, tg);
    } else {
        // General 2D matmul
        enc->setComputePipelineState(ctx.pso_matmul);
        enc->setBuffer(buf_A, 0, 0);
        enc->setBuffer(buf_B, 0, 1);
        enc->setBuffer(buf_C, 0, 2);
        enc->setBuffer(buf_sa, 0, 3);
        enc->setBuffer(buf_sb, 0, 4);
        enc->setBytes(&M, sizeof(uint32_t), 5);
        enc->setBytes(&N, sizeof(uint32_t), 6);
        enc->setBytes(&K, sizeof(uint32_t), 7);
        enc->setBytes(&scale_mode, sizeof(uint32_t), 8);

        MTL::Size grid(N, M, 1);   // x=col, y=row
        MTL::Size tg(16, 16, 1);
        enc->dispatchThreads(grid, tg);
    }

    enc->endEncoding();
    cmd->commit();
    cmd->waitUntilCompleted();

    // Copy result to torch tensor, then move to MPS
    auto result = torch::from_blob(buf_C->contents(), {(int64_t)M, (int64_t)N},
                                    torch::kFloat32).clone();

    // Release Metal buffers
    buf_A->release();
    buf_B->release();
    buf_C->release();
    buf_sa->release();
    buf_sb->release();

    return result.to(torch::kMPS);
}


// ─── fp8_dequantize ─────────────────────────────────────────────────────────
// input: uint8 tensor (FP8 encoded), returns float16 tensor

torch::Tensor fp8_dequantize(torch::Tensor input, torch::Tensor scale) {
    auto& ctx = FP8MetalContext::instance();
    ctx.init();

    TORCH_CHECK(input.dtype() == torch::kUInt8, "input must be uint8");

    torch::Tensor inp_cpu = input.to(torch::kCPU).contiguous();
    torch::Tensor sc_cpu = scale.to(torch::kCPU).to(torch::kFloat32).contiguous();

    uint32_t count = inp_cpu.numel();

    auto* dev = ctx.device;
    MTL::Buffer* buf_in = dev->newBuffer(u8_ptr(inp_cpu), count * sizeof(uint8_t), MTL::ResourceStorageModeShared);
    MTL::Buffer* buf_out = dev->newBuffer(count * sizeof(uint16_t), MTL::ResourceStorageModeShared);  // half = 2 bytes

    MTL::CommandBuffer* cmd = ctx.queue->commandBuffer();
    MTL::ComputeCommandEncoder* enc = cmd->computeCommandEncoder();
    enc->setComputePipelineState(ctx.pso_to_half);
    enc->setBuffer(buf_in, 0, 0);
    enc->setBuffer(buf_out, 0, 1);
    enc->setBytes(&count, sizeof(uint32_t), 2);

    uint32_t threads_per_group = 256;
    MTL::Size grid(count, 1, 1);
    MTL::Size tg(threads_per_group, 1, 1);
    enc->dispatchThreads(grid, tg);
    enc->endEncoding();
    cmd->commit();
    cmd->waitUntilCompleted();

    auto sizes = inp_cpu.sizes().vec();
    auto result = torch::from_blob(buf_out->contents(), sizes, torch::kFloat16).clone();

    // Apply scale
    float sc_val = sc_cpu.item<float>();
    result = result * sc_val;

    buf_in->release();
    buf_out->release();

    return result.to(torch::kMPS);
}


// ─── fp8_quantize ───────────────────────────────────────────────────────────
// input: float32 tensor, returns (uint8 tensor, scale tensor)

std::tuple<torch::Tensor, torch::Tensor> fp8_quantize(torch::Tensor input) {
    auto& ctx = FP8MetalContext::instance();
    ctx.init();

    torch::Tensor inp_cpu = input.to(torch::kCPU).to(torch::kFloat32).contiguous();
    uint32_t count = inp_cpu.numel();

    // Compute scale: max_fp8 / max(abs(input))
    float amax = inp_cpu.abs().max().item<float>();
    float max_fp8 = 448.0f;  // max representable in e4m3fn
    float scale = (amax > 0.0f) ? (max_fp8 / amax) : 1.0f;

    // Scale input before quantization
    torch::Tensor scaled = (inp_cpu * scale).contiguous();

    auto* dev = ctx.device;
    MTL::Buffer* buf_in = dev->newBuffer(
        static_cast<const float*>(scaled.data_ptr()), count * sizeof(float),
        MTL::ResourceStorageModeShared);
    MTL::Buffer* buf_out = dev->newBuffer(count * sizeof(uint8_t), MTL::ResourceStorageModeShared);

    MTL::CommandBuffer* cmd = ctx.queue->commandBuffer();
    MTL::ComputeCommandEncoder* enc = cmd->computeCommandEncoder();
    enc->setComputePipelineState(ctx.pso_to_fp8);
    enc->setBuffer(buf_in, 0, 0);
    enc->setBuffer(buf_out, 0, 1);
    enc->setBytes(&count, sizeof(uint32_t), 2);

    uint32_t tpg = 256;
    enc->dispatchThreads(MTL::Size(count, 1, 1), MTL::Size(tpg, 1, 1));
    enc->endEncoding();
    cmd->commit();
    cmd->waitUntilCompleted();

    auto sizes = inp_cpu.sizes().vec();
    auto quantized = torch::from_blob(buf_out->contents(), sizes, torch::kUInt8).clone();

    // Inverse scale for dequantization later
    auto inv_scale = torch::tensor({1.0f / scale}, torch::kFloat32);

    buf_in->release();
    buf_out->release();

    return std::make_tuple(quantized.to(torch::kMPS), inv_scale.to(torch::kMPS));
}


// ─── PYBIND11 Module ────────────────────────────────────────────────────────

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fp8_scaled_mm", &fp8_scaled_mm,
          "FP8 scaled matrix multiplication on Metal GPU",
          py::arg("A"), py::arg("B"), py::arg("scale_a"), py::arg("scale_b"));
    m.def("fp8_dequantize", &fp8_dequantize,
          "FP8 to float16 dequantization on Metal GPU",
          py::arg("input"), py::arg("scale"));
    m.def("fp8_quantize", &fp8_quantize,
          "Float to FP8 quantization on Metal GPU",
          py::arg("input"));
}
