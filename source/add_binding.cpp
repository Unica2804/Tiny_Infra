#include <torch/extension.h>

// Forward declaration of our C++ launcher function
torch::Tensor add_cuda(torch::Tensor a, torch::Tensor b);

// Bind the C++ function to a Python module named 'custom_ops'
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add", &add_cuda, "A highly optimized element-wise addition CUDA kernel");
}