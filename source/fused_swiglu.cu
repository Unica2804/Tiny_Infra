# include <torch/extension.h>
# include <cuda_fp16.h>
# include <math_constants.h>

union Float4Half8{
    float4 f4;
    half h8[8];
};

__global__ void fused_swiglu_kernel(
    const float4* __restrict__ input,
    float4* __restrict__ output,
    int hidden_size_div_8,
    int total_tokens
){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int total_chunks = total_tokens * hidden_size_div_8;

    if (idx<total_chunks){
        int token_idx = idx / hidden_size_div_8;
        int chunk_idx = idx % hidden_size_div_8;

        int gate_offset = token_idx * (hidden_size_div_8 * 2) + chunk_idx;
        int up_offset = gate_offset + hidden_size_div_8;

        int out_offset = token_idx * hidden_size_div_8 + chunk_idx;

        // Initialize the registers
        Float4Half8 gate_chunk;
        Float4Half8 up_chunk;
        Float4Half8 out_chunk;

        // Load the gate and up chunks from global memory
        gate_chunk.f4 = input[gate_offset];
        up_chunk.f4 = input[up_offset];

        // Compute the sigmoid of the gate chunk
        # pragma unroll
        for (int i=0; i<8; ++i){
            //Cast to float32 for higher accuracy
            float gate_val = __half2float(gate_chunk.h8[i]);
            float up_val = __half2float(up_chunk.h8[i]);

            // Apply SiLU activation
            float sigmoid = 1.0f/(1.0f + __expf(-gate_val));
            float silu_val = gate_val * sigmoid;

            // Multiply the SiLU value with the up value
            out_chunk.h8[i] = __float2half(silu_val * up_val);
        }
        output[out_offset] = out_chunk.f4;
    }
    
}

torch::Tensor fused_swiglu(torch::Tensor input){
    TORCH_CHECK(input.device().is_cuda(), "Input tensor must be on CUDA.");
    TORCH_CHECK(input.scalar_type() == torch::kFloat16, "Input tensor must be FP16.");
    TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous in memory.");
    
    // Qwen2 fused MLPs double the hidden dimension
    int total_elements = input.size(-1);
    TORCH_CHECK(total_elements % 2 == 0, "Input hidden dimension must be even.");

    int hidden_size = total_elements / 2;
    int total_tokens = input.numel() / total_elements;

    TORCH_CHECK(hidden_size % 8 == 0, "Hidden size must be a multiple of 8 for vectorized access.");

    auto output = torch::empty({total_tokens, hidden_size},torch::dtype(torch::kFloat16).device(input.device()));
    int hidden_size_div_8 = hidden_size / 8;
    int total_chunks = total_tokens * hidden_size_div_8;

    // Launch Configuration: standard 256 threads per block
    const int threads = 256;
    const int blocks = (total_chunks + threads - 1) / threads;

    // Launch the kernel, casting the half* pointers to float4* for the VRAM highway
    fused_swiglu_kernel<<<blocks, threads>>>(
        reinterpret_cast<const float4*>(input.data_ptr<at::Half>()),
        reinterpret_cast<float4*>(output.data_ptr<at::Half>()),
        hidden_size_div_8,
        total_tokens
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA Execution Error: ", cudaGetErrorString(err));

    return output;
}