import triton
import triton.language as tl
import torch

@triton.jit
def swiglu_kernel(x_ptr,
    out_ptr,
    M, H,                              
    stride_xm, stride_xn,
    stride_om, stride_on,
    BLOCK_H: tl.constexpr,):
    pid = tl.program_id(0)
    
    gate_ptr = tl.make_block_ptr(
        base=x_ptr,
        shape=(M, 2*H),
        strides=(stride_xm, stride_xn),
        offsets=(pid, 0),
        block_shape=(1, BLOCK_H),
        order=(1,0),
    )

    val_ptr = tl.make_block_ptr(
        base=x_ptr+H*stride_xn,
        shape=(M, 2*H),
        strides=(stride_xm, stride_xn),
        offsets=(pid, 0),
        block_shape=(1, BLOCK_H),
        order=(1,0),
    )

    out_ptr = tl.make_block_ptr(
        base=out_ptr,
        shape=(M, H),
        strides=(stride_om, stride_on),
        offsets=(pid, 0),
        block_shape=(1, BLOCK_H),
        order=(1,0),
    )

    gate = tl.load(gate_ptr, boundary_check=(1, 0), padding_option='zero')
    value = tl.load(val_ptr, boundary_check=(1, 0), padding_option='zero')
    
    gate_f32 = gate.to(tl.float32)
    res_f32 = (gate_f32 * tl.sigmoid(gate_f32)) * value.to(tl.float32)
    tl.store(out_ptr, res_f32.to(gate.dtype), boundary_check=(1,))

def swiglu(X: torch.Tensor) -> torch.Tensor:
    M, two_H = X.shape
    H = two_H // 2
    Y = torch.empty((M, H), device=X.device, dtype=X.dtype)
    grid = (M,)
    BLOCK_H = triton.next_power_of_2(H)
    swiglu_kernel[grid](X,Y,M,H,X.stride(0),X.stride(1),Y.stride(0),Y.stride(1),BLOCK_H=BLOCK_H)
    return Y