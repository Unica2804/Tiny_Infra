#include <torch/extension.h>

torch::Tensor fused_swiglu(torch::Tensor input);

PYBIND11_MODULE(custom_swiglu, m) {
    m.def("forward", &fused_swiglu, "Fused SwiGLU Activation with 128-bit Vectorization");
}