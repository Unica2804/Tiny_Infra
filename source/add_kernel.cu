#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// ---------------------------------------------------------
// 1. The CUDA Kernel (Runs on the physical GPU threads)
// ---------------------------------------------------------
template <typename scalar_t>
__global__ void add_cuda_kernel(
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b,
    scalar_t* __restrict__ out,
    int size) {
    
    // Calculate the global thread ID
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Ensure we don't read out of bounds if the tensor size isn't a perfect multiple of the block size
    if (idx < size) {
        out[idx] = a[idx] + b[idx];
    }
}

// ---------------------------------------------------------
// 2. The C++ Launcher (Runs on the CPU, configures the GPU)
// ---------------------------------------------------------
torch::Tensor add_cuda(torch::Tensor a, torch::Tensor b) {
    // Robust Error Handling: Never assume Python sent the right data
    TORCH_CHECK(a.device().is_cuda(), "Tensor 'a' must be a CUDA tensor.");
    TORCH_CHECK(b.device().is_cuda(), "Tensor 'b' must be a CUDA tensor.");
    TORCH_CHECK(a.sizes() == b.sizes(), "Tensors must have the exact same shape.");
    TORCH_CHECK(a.is_contiguous(), "Tensor 'a' must be contiguous in memory.");
    TORCH_CHECK(b.is_contiguous(), "Tensor 'b' must be contiguous in memory.");

    // Allocate an empty output tensor on the same device and with the same type
    auto out = torch::empty_like(a);
    int size = a.numel();

    // Define the execution grid (256 threads per block is a standard optimal baseline)
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;

    // AT_DISPATCH dynamically checks the Python tensor dtype (e.g., float32, float16) 
    // and compiles the correct C++ template version of our kernel
    AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, a.scalar_type(), "add_cuda_kernel", ([&] {
        add_cuda_kernel<scalar_t><<<blocks, threads>>>(
            a.data_ptr<scalar_t>(),
            b.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            size
        );
    }));
    
    // Synchronize and catch any silent hardware errors during execution
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA Execution Error: ", cudaGetErrorString(err));

    return out;
}