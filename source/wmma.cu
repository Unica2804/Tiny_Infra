#include <torch/extension.h>
#include <cuda_fp16.h>
#include <mma.h> // NVIDIA's Tensor Core API

// Use the nvcuda namespace to access the WMMA (Warp Matrix Multiply Accumulate) functions
using namespace nvcuda;

// Define the strict 16x16x16 shape that the physical hardware requires
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

// ---------------------------------------------------------
// 1. The Tensor Core Kernel (Executes 1 Warp per 16x16 Tile)
// ---------------------------------------------------------
__global__ void wmma_gemm_kernel(
    const half* __restrict__ A, 
    const half* __restrict__ B, 
    half* __restrict__ C, 
    int M, int N, int K) {
    
    // In WMMA, we don't think about single threads. We think about "Warps" (Groups of 32 threads).
    // Calculate which 16x16 tile of the output matrix 'C' this specific Warp is responsible for.
    int warpM = blockIdx.x * WMMA_M;
    int warpN = blockIdx.y * WMMA_N;

    // Define the hardware registers (Fragments)
    // Notice how we explicitly tell the hardware A is row_major, and B is col_major (transposed!)
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> c_frag;

    // Initialize our accumulator fragment to 0.0
    wmma::fill_fragment(c_frag, __float2half(0.0f));

    // Slide across the 'K' dimension (the inner dimension of the dot product)
    for (int k = 0; k < K; k += WMMA_K) {
        
        // 1. LOAD: The 32 threads cooperate to pull a 16x16 tile of A and B from VRAM into registers
        // The last argument is the "Leading Dimension" (how many elements to skip to get to the next row/col)
        wmma::load_matrix_sync(a_frag, A + warpM * K + k, K);
        wmma::load_matrix_sync(b_frag, B + warpN * K + k, K); // B is col_major, so LDA is K

        // 2. COMPUTE: Fire the physical Tensor Core circuit!
        // This does 4,096 math operations in a single clock cycle.
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // 3. STORE: The 32 threads cooperate to write the final 16x16 tile back to VRAM
    wmma::store_matrix_sync(C + warpM * N + warpN, c_frag, N, wmma::mem_row_major);
}

// ---------------------------------------------------------
// 2. The C++ Launcher
// ---------------------------------------------------------
torch::Tensor wmma_gemm(torch::Tensor a, torch::Tensor b) {
    // Robust Error Handling: Enforce our strict constraints
    TORCH_CHECK(a.scalar_type() == torch::kFloat16, "Tensor A must be FP16 (Half).");
    TORCH_CHECK(b.scalar_type() == torch::kFloat16, "Tensor B must be FP16 (Half).");
    TORCH_CHECK(a.is_contiguous(), "Tensor A must be contiguous.");
    TORCH_CHECK(b.is_contiguous(), "Tensor B must be contiguous.");
    
    int M = a.size(0);
    int K = a.size(1);
    // Because B is transposed (Column-Major), its shape in memory is [N, K]
    int N = b.size(0); 
    TORCH_CHECK(b.size(1) == K, "Inner dimensions (K) must match.");
    
    // Ensure dimensions are perfect multiples of 16 (No padding required as per rules)
    TORCH_CHECK(M % 16 == 0 && N % 16 == 0 && K % 16 == 0, "M, N, K must be multiples of 16.");

    auto c = torch::empty({M, N}, torch::dtype(torch::kFloat16).device(a.device()));

    // Launch configuration: 1 Block = 1 Warp (32 threads). 
    // Grid size calculates how many 16x16 tiles fit in our M x N output matrix.
    dim3 threads(32);
    dim3 blocks(M / WMMA_M, N / WMMA_N);

    // Launch the kernel
    wmma_gemm_kernel<<<blocks, threads>>>(
        reinterpret_cast<const half*>(a.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(b.data_ptr<at::Half>()),
        reinterpret_cast<half*>(c.data_ptr<at::Half>()),
        M, N, K
    );

    cudaDeviceSynchronize();
    return c;
}