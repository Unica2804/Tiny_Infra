#include <torch/extension.h>

void launch_paged_attention(
    torch::Tensor& out,
    torch::Tensor& q,
    torch::Tensor& k_cache,
    torch::Tensor& v_cache,
    torch::Tensor& block_tables,
    torch::Tensor& context_lens,
    int block_size);

PYBIND11_MODULE(custom_paged_attn, m) {
    m.def("launch_paged_attention", &launch_paged_attention, "Micro-PagedAttention Kernel");
}