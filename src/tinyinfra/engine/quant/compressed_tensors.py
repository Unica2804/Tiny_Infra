"""compressed-tensors (llm-compressor) W8A16 adapter: packed int32 -> canonical uint8.

Layout (RedHatAI Qwen2.5-*.quantized.w8a16):
    weight_packed [N, K/4] int32  -- 4 int8 values per int32 along dim 1
    weight_scale  [N, 1] bf16     -- per-output-channel scale
    weight_shape  [2] int64       -- original [N, K]
    symmetric, no zero-point tensor.
"""

import torch

from tinyinfra.engine.quant import QuantTensor, register


@register("compressed-tensors", 8)
def unpack_compressed_tensors8(weight_packed: torch.Tensor, weight_scale: torch.Tensor,
                               weight_shape: torch.Tensor | None = None,
                               bias: torch.Tensor | None = None, **kwargs) -> QuantTensor:
    N, K_div4 = weight_packed.shape
    K = K_div4 * 4
    if weight_shape is not None:
        N_chk, K_chk = (int(v) for v in weight_shape.tolist())
        assert (N_chk, K_chk) == (N, K), f"weight_shape {N_chk,K_chk} != {N,K}"

    # Same packing as GPTQ-Int8: 4 int8 per int32, but along dim 1 here.
    W = torch.empty((K, N), dtype=torch.uint8, device=weight_packed.device)
    for j in range(4):
        W[j::4] = ((weight_packed >> (8 * j)) & 0xFF).to(torch.uint8).T

    # Per-channel scales [N,1] -> [1,N]; symmetric => zero point 0.
    scales = weight_scale.T.to(torch.float16).contiguous()
    # compressed-tensors symmetric int8 uses unsigned offset form: w = (q - 128) * scale
    zeros = torch.full((1, N), 128, dtype=torch.uint8, device=W.device)
    return QuantTensor(qweight=W, scales=scales, zeros=zeros,
                       group=K, sym=True, bias=bias)
