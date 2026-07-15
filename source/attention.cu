# include <cuda_fp16.h>
# include <torch/extension.h>

struct PagedAttentionArgs {

    const __half* __restrict__ q;
    const __half* __restrict__ k_cache;
    const __half* __restrict__ v_cache;
    const int* __restrict__ block_tables;
    const int* __restrict__ context_lens;

    int batch_size;
    int num_heads;
    int num_kv_heads;
    int head_dim;
    int block_size;
    int max_blocks_per_seq;
    float sm_scale;
};

__inline__ __device__ float warp_reduce_sum(float val){
    for (int offset = warpSize/2; offset > 0; offset /= 2){
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__inline__ __device__ float block_reduce_sum(float val) {
    // Step 1: Reduce within the local warp (32 threads)
    float sum = warp_reduce_sum(val);
    
    // Step 2: Thread 0 of each warp writes its sum to shared memory
    __shared__ float warp_sums[32]; // Accommodates up to 32 warps (1024 threads max)
    
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    
    if (lane_id == 0) {
        warp_sums[warp_id] = sum;
    }
    __syncthreads(); // Wait for all warps to finish writing
    
    // Step 3: The first warp reads the saved sums and reduces them into the final total
    float block_sum = 0.0f;
    if (warp_id == 0) {
        // Only read valid warp sums, pad the rest with 0
        block_sum = (lane_id < (blockDim.x / 32)) ? warp_sums[lane_id] : 0.0f;
        block_sum = warp_reduce_sum(block_sum);
    }
    
    return block_sum;
}

__global__ void paged_attention_kernel(
    PagedAttentionArgs args, __half* __restrict__ out
){
    int seq_idx = blockIdx.y;
    int head_idx = blockIdx.x;
    int tid = threadIdx.x;

    int context_len = args.context_lens[seq_idx];
    if(context_len == 0) return;

    int q_offset = (seq_idx * args.num_heads * args.head_dim) + (head_idx * args.head_dim);

    float q_val = 0.0f;
    if(tid < args.head_dim){
        // load query tensor pointer
        q_val = __half2float(args.q[q_offset + tid]);
        q_val *= args.sm_scale;
    }

    extern __shared__ float s_mem[];
    // Partition shared memory into two halves for keys and values
    float* k_shared = s_mem;
    float* v_shared = &s_mem[args.block_size * args.head_dim];

    // Init online softmax tracking variables
    float m_i = -1e20f;
    float d_i = 0.0f;
    float o_i = 0.0f;

    // Calc how many blocks we need to process for this sequence
    int num_blocks = (context_len + args.block_size - 1) / args.block_size;

    // Tile through the paged_cache
    for (int b=0; b<num_blocks; ++b){
        int physical_block_idx = args.block_tables[seq_idx * args.max_blocks_per_seq + b];
        
        if (physical_block_idx < 0) break;
        // GQA routing multiple q heads and 1 k/v head
        int kv_head_idx = head_idx / (args.num_heads / args.num_kv_heads);

        // Collaboratively load key block into shared memory
        int kv_base_offset = (physical_block_idx * args.num_kv_heads * args.block_size * args.head_dim) + (kv_head_idx * args.block_size * args.head_dim);
        
        for (int t=0; t<args.block_size; ++t){
            int token_idx = (b*args.block_size) + t;
            if(token_idx < context_len && tid < args.head_dim){
                int physical_offset = kv_base_offset + (t * args.head_dim) + tid;
                k_shared[t * args.head_dim + tid] = __half2float(args.k_cache[physical_offset]);
            }
            else {
                k_shared[t * args.head_dim + tid] = 0.0f;
            }
        } 
        __syncthreads();

        // Compute Q * k^T
        for(int t=0; t<args.block_size; ++t){
            int token_idx = (b*args.block_size) + t;
            if (token_idx >= context_len) break;

            // Multiply Q by K
            float qk = q_val * k_shared[t * args.head_dim + tid];

            // perform block-level reduction to compute the sum of qk values across threads
            float score = block_reduce_sum(qk);

            __shared__ float shared_score;

            if(tid == 0) shared_score = score;
            __syncthreads();
            score = shared_score;

            // Update online softmax tracking variables
            float m_i_new = fmaxf(m_i, score);
            float alpha = expf(m_i - m_i_new);
            float exp_score = expf(score - m_i_new);

            d_i = alpha * d_i + exp_score;
            // Rescale the output accumulator
            o_i = o_i * alpha;

            // Collaborative load V tile to shared memory
            if(tid < args.head_dim){
                int physical_offset = kv_base_offset + (t * args.head_dim) + tid;
                float v_val = __half2float(args.v_cache[physical_offset]);

                // accumulate P * V into the output accumulator
                o_i += exp_score * v_val;
            }
            __syncthreads();
            m_i = m_i_new;
        }
    }

    // Normalize and write the final output to global memory
    if(tid < args.head_dim){
        o_i /= d_i;
        int out_offset = (seq_idx * args.num_heads * args.head_dim) + (head_idx * args.head_dim) + tid;
        out[out_offset] = __float2half(o_i);
    }
}

void launch_paged_attention(
    torch::Tensor& out,
    torch::Tensor& q,
    torch::Tensor& k_cache,
    torch::Tensor& v_cache,
    torch::Tensor& block_tables,
    torch::Tensor& context_lens,
    int block_size
){
    int batch_size = q.size(0);
    int num_heads = q.size(1);
    int head_dim = q.size(2);
    int num_kv_heads = k_cache.size(1);
    int max_blocks_per_seq = block_tables.size(1);

    PagedAttentionArgs args;
    args.q = reinterpret_cast<__half*>(q.data_ptr<at::Half>());
    args.k_cache = reinterpret_cast<__half*>(k_cache.data_ptr<at::Half>());
    args.v_cache = reinterpret_cast<__half*>(v_cache.data_ptr<at::Half>());
    args.block_tables = block_tables.data_ptr<int>();
    args.context_lens = context_lens.data_ptr<int>();

    args.batch_size = batch_size;
    args.num_heads = num_heads;
    args.num_kv_heads = num_kv_heads;
    args.head_dim = head_dim;
    args.block_size = block_size;
    args.max_blocks_per_seq = max_blocks_per_seq;
    args.sm_scale = 1.0f / sqrtf(static_cast<float>(head_dim));

    dim3 grid(num_heads, batch_size);
    dim3 block(head_dim);

    int shared_mem_size = (block_size * head_dim * 2) * sizeof(float);

    paged_attention_kernel<<<grid, block, shared_mem_size>>>(
        args,
        reinterpret_cast<__half*>(out.data_ptr<at::Half>())
    );
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Paged attention kernel launch failed: ", cudaGetErrorString(err));
}