#include <torch/extension.h>

void apply_fused_rope_inplace(
    torch::Tensor& q, 
    torch::Tensor& k, 
    torch::Tensor& cos, 
    torch::Tensor& sin, 
    torch::Tensor& pos_ids);

PYBIND11_MODULE(custom_rope, m) {
    m.def("apply_inplace", &apply_fused_rope_inplace, "In-place Fused RoPE Kernel");
}