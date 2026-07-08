import torch
from safetensors.torch import load_file, save_file
import os

def fuse_custom_engine_weights(input_path: str, output_path: str, num_layers: int = 24):
    print(f"📥 Loading custom mapped weights from {input_path}...")
    
    # Load the flat dictionary of custom-named weights
    state_dict = load_file(input_path)
    fused_dict = {}
    processed_keys = set()
    
    print("🔨 Fusing Memory Layouts for Fused CUDA Kernels...")
    
    for i in range(num_layers):
        engine_prefix = f"blocks.{i}"

        # ==========================================
        # 1. FUSE ATTENTION: Query, Key, Value
        # ==========================================
        # Use the exact custom names you created on Day 1
        q_key = f"{engine_prefix}.attn.q_proj"
        k_key = f"{engine_prefix}.attn.k_proj"
        v_key = f"{engine_prefix}.attn.v_proj"
        
        if q_key in state_dict and k_key in state_dict and v_key in state_dict:
            # Concatenate Q, K, and V along dimension 0
            fused_qkv = torch.cat([
                state_dict[q_key], 
                state_dict[k_key], 
                state_dict[v_key]
            ], dim=0)
            
            # Save under a new clean fused name
            fused_dict[f"{engine_prefix}.attn.fused_qkv"] = fused_qkv
            processed_keys.update([q_key, k_key, v_key])

        # ==========================================
        # 2. FUSE MLP: Gate and Up Projections
        # ==========================================
        # Use the exact custom SwiGLU names you created on Day 1
        gate_key = f"{engine_prefix}.mlp.gate_proj"
        up_key = f"{engine_prefix}.mlp.up_proj"
        
        if gate_key in state_dict and up_key in state_dict:
            # Concatenate Gate and Up along dimension 0
            fused_gate_up = torch.cat([
                state_dict[gate_key], 
                state_dict[up_key]
            ], dim=0)
            
            # Save under a new clean fused name
            fused_dict[f"{engine_prefix}.mlp.fused_gate_up"] = fused_gate_up
            processed_keys.update([gate_key, up_key])

    # ==========================================
    # 3. TRANSFER UNCHANGED WEIGHTS
    # ==========================================
    # Copy over the embeddings, final norms, down_projs, and out_projs untouched
    for key, tensor in state_dict.items():
        if key not in processed_keys:
            fused_dict[key] = tensor
            
    print(f"💾 Saving highly-optimized fused weights to {output_path}...")
    save_file(fused_dict, output_path)
    
    # Verify the math
    print("\n✅ Fusion Complete! Layout Comparison (Block 0):")
    if "blocks.0.mlp.fused_gate_up" in fused_dict:
        print(f"Old Gate Shape: {state_dict['blocks.0.mlp.gate_proj'].shape}")
        print(f"Old Up Shape  : {state_dict['blocks.0.mlp.up_proj'].shape}")
        print(f"NEW Fused MLP : {fused_dict['blocks.0.mlp.fused_gate_up'].shape}")

if __name__ == "__main__":
    INPUT_SAFETENSORS = "model/qwen25_0.5b_extracted.safetensors"
    OUTPUT_SAFETENSORS = "model/qwen25_0.5b_fused.safetensors"
    
    if os.path.exists(INPUT_SAFETENSORS):
        fuse_custom_engine_weights(INPUT_SAFETENSORS, OUTPUT_SAFETENSORS)
    else:
        print(f"❌ Error: Cannot find {INPUT_SAFETENSORS}. Make sure it is in the same directory.")