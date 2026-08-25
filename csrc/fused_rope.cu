// Inplace rotary positional embedding (RoPE) implementation for fused attention kernels.

# include <torch/torch.h>
# include <cuda_fp16.h>

__global__ void fused_rope_kernel(
    half* __restrict__ q,
    half* __restrict__ k,
    const float* __restrict__ cos,
    const float* __restrict__ sin,
    const int* __restrict__ pos,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int total_q_elements,
    int total_k_elements
){
    // Global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // split dim in two halves for RoPE
    int half_dim = head_dim / 2;

    // Process query tensor
    if (idx < total_q_elements/2){
    
        int token_idx = idx / (num_heads * half_dim);
        int head_idx = (idx / half_dim) % num_heads;
        int dim_idx = idx % half_dim;

        // Convert 1D index into 3D coordinates for query tensor
        int flattened_q_idx = token_idx * (num_heads * head_dim) + head_idx * head_dim + dim_idx;
        // Get cos_sin index
        int cos_sin_idx = pos[token_idx] * half_dim + dim_idx;

        // Load the original query value and upcast to float32
        float x1 = __half2float(q[flattened_q_idx]);
        float x2 = __half2float(q[flattened_q_idx + half_dim]);

        // Calculate the cosine and sine values for the RoPE transformation
        float cos_val = cos[cos_sin_idx];
        float sin_val = sin[cos_sin_idx];

        // Apply RoPE transformation
        float out1 = x1 * cos_val - x2 * sin_val;
        float out2 = x1 * sin_val + x2 * cos_val;
        
        // overwrite the original query values with the transformed values, downcast to half precision
        q[flattened_q_idx] = __float2half(out1);
        q[flattened_q_idx + half_dim] = __float2half(out2);
    }

    // Process key tensor
    if (idx < total_k_elements/2){

        int token_idx = idx / (num_kv_heads * half_dim);
        int head_idx = (idx / half_dim) % num_kv_heads;
        int dim_idx = idx % half_dim;

        // Convert 1D index into 3D coordinates for key tensor
        int flattened_k_idx = token_idx * (num_kv_heads * head_dim) + head_idx * head_dim + dim_idx;
        // Get cos_sin index
        int cos_sin_idx = pos[token_idx] * half_dim + dim_idx;

        // Load the original key value and upcast to float32
        float x1 = __half2float(k[flattened_k_idx]);
        float x2 = __half2float(k[flattened_k_idx + half_dim]);

        float cos_val = cos[cos_sin_idx];
        float sin_val = sin[cos_sin_idx];

        // Apply RoPE transformation
        float out1 = x1 * cos_val - x2 * sin_val;
        float out2 = x1 * sin_val + x2 * cos_val;
        
        // overwrite the original key values with the transformed values, downcast to half precision
        k[flattened_k_idx] = __float2half(out1);
        k[flattened_k_idx + half_dim] = __float2half(out2);
    }

}

// C++ interface for the fused RoPE kernel
void apply_fused_rope_inplace(
    torch::Tensor& q,
    torch::Tensor& k,
    torch::Tensor& cos,
    torch::Tensor& sin,
    torch::Tensor& pos_ids
){
    // Ensure the input tensors are on the same device and have the correct data types
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && cos.is_cuda() && sin.is_cuda() && pos_ids.is_cuda(), "All tensors must be CUDA tensors.");
    TORCH_CHECK(q.dtype() == torch::kHalf && k.dtype() == torch::kHalf, "Query and Key tensors must be of type half.");
    TORCH_CHECK(cos.dtype() == torch::kFloat && sin.dtype() == torch::kFloat, "Cosine and Sine tensors must be of type float.");
    TORCH_CHECK(pos_ids.dtype() == torch::kInt, "Position IDs tensor must be of type int.");

    // Get dimensions
    int num_tokens = q.size(0);
    int num_heads = q.size(1);
    int num_kv_heads = k.size(1);
    int head_dim = q.size(2);

    int total_q_elements = num_tokens * num_heads * head_dim;
    int total_k_elements = num_tokens * num_kv_heads * head_dim;

    int max_elements = max(total_q_elements, total_k_elements);
    int threads_needed = max_elements / 2; // Each thread processes two elements (one for q and one for k)

    const int threads_per_block = 256;
    const int blocks = (threads_needed + threads_per_block - 1) / threads_per_block;

    // Launch the fused RoPE kernel
    fused_rope_kernel<<<blocks, threads_per_block>>>(
        reinterpret_cast<half*>(q.data_ptr<at::Half>()),
        reinterpret_cast<half*>(k.data_ptr<at::Half>()),
        cos.data_ptr<float>(),
        sin.data_ptr<float>(),
        pos_ids.data_ptr<int>(),
        num_heads,
        num_kv_heads,
        head_dim,
        total_q_elements,
        total_k_elements
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Error in fused_rope_kernel: ", cudaGetErrorString(err));
}