"""
Day 1: Weight Extraction Script for Qwen2.5-0.5B-Instruct.

This script safely downloads the Qwen2.5-0.5B-Instruct model from Hugging Face,
extracts its raw tensor weights, maps them to our custom explicitly defined 
architecture, and saves them into a flat .safetensors file.
"""

import sys
import torch
from transformers import AutoModelForCausalLM
from safetensors.torch import save_file


def extract_and_save_weights(model_id: str, output_path: str) -> None:
    """
    Downloads a pretrained model, maps its state dictionary to a custom layout,
    and saves it to disk in safetensors format.
    """
    print(f"📥 Initiating download and load of {model_id}...")
    
    try:
        # Load the model strictly on CPU to conserve GPU VRAM for inference testing
        # We use float32 to maintain full precision before any custom quantization
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            device_map="cpu",
            low_cpu_mem_usage=True
        )
    except Exception as e:
        print(f"❌ Error loading model from Hugging Face: {e}")
        sys.exit(1)

    # Extract the raw PyTorch state dictionary
    state_dict = model.state_dict()
    mapped_weights = {}

    print("🔄 Mapping standard architecture to explicit custom engine layout...")
    
    try:
        # 1. Map core global embeddings and standard heads
        mapped_weights["token_embeddings"] = state_dict["model.embed_tokens.weight"]
        mapped_weights["final_norm_weight"] = state_dict["model.norm.weight"]
        mapped_weights["lm_head"] = state_dict["lm_head.weight"].clone()

        # 2. Iterate through all transformer decoder blocks (24 for Qwen2.5-0.5B)
        num_layers = model.config.num_hidden_layers
        for i in range(num_layers):
            hf_prefix = f"model.layers.{i}"
            engine_prefix = f"blocks.{i}"

            # Extract attention projection matrices explicitly
            mapped_weights[f"{engine_prefix}.attn.q_proj"] = state_dict[f"{hf_prefix}.self_attn.q_proj.weight"]
            mapped_weights[f"{engine_prefix}.attn.k_proj"] = state_dict[f"{hf_prefix}.self_attn.k_proj.weight"]
            mapped_weights[f"{engine_prefix}.attn.v_proj"] = state_dict[f"{hf_prefix}.self_attn.v_proj.weight"]
            mapped_weights[f"{engine_prefix}.attn.out_proj"] = state_dict[f"{hf_prefix}.self_attn.o_proj.weight"]

            # Extract SwiGLU MLP matrices explicitly
            mapped_weights[f"{engine_prefix}.mlp.gate_proj"] = state_dict[f"{hf_prefix}.mlp.gate_proj.weight"]
            mapped_weights[f"{engine_prefix}.mlp.up_proj"] = state_dict[f"{hf_prefix}.mlp.up_proj.weight"]
            mapped_weights[f"{engine_prefix}.mlp.down_proj"] = state_dict[f"{hf_prefix}.mlp.down_proj.weight"]

            # Extract RMSNorm weights explicitly
            mapped_weights[f"{engine_prefix}.attn_norm"] = state_dict[f"{hf_prefix}.input_layernorm.weight"]
            mapped_weights[f"{engine_prefix}.mlp_norm"] = state_dict[f"{hf_prefix}.post_attention_layernorm.weight"]

    except KeyError as e:
        print(f"❌ Error mapping weights: Missing expected key {e} in the model state dictionary.")
        sys.exit(1)

    print(f"💾 Saving mapped weights to {output_path}...")
    
    try:
        # Save the dictionary as a clean, memory-mappable binary file
        save_file(mapped_weights, output_path)
    except Exception as e:
        print(f"❌ Error saving safetensors file: {e}")
        sys.exit(1)

    print(f"✅ Extraction complete! Saved to {output_path}.")


if __name__ == "__main__":
    TARGET_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
    OUTPUT_FILE = "qwen25_0.5b_extracted.safetensors"
    extract_and_save_weights(TARGET_MODEL, OUTPUT_FILE)