#include <torch/extension.h>

// Forward declaration
torch::Tensor wmma_gemm(torch::Tensor a, torch::Tensor b);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm", &wmma_gemm, "Tensor Core WMMA GEMM (FP16, Transposed B)");
}