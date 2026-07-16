# File to load the model weights and map them to the new architecture

import torch
from safetensors.torch import load_file
import os
from models.qwen_25 import Qwen2ForCausalLM
from models.config import QwenConfig

def load_qwen (model_path: str, device: str = "cuda") -> Qwen2ForCausalLM:
    """
    Loads the Qwen model weights from a .safetensors file and maps them to the new architecture.
    
    Args:
        model_path (str): Path to the .safetensors file containing the model weights.
        device (str): Device to load the model onto (default is "cuda").
    
    Returns:
        Qwen2ForCasualLM: The loaded model with weights mapped to the new architecture.
    """
    print("Initializing Model Configuration...")
    config = QwenConfig()
    model = Qwen2ForCausalLM(config)

    print(f"Reading raw weights from {model_path}...")

    # Extracting raw weights from the .safetensors file
    extracted_weights = load_file(model_path)

    #create a empty state_dict to hold the mapped weights
    translated_state_dict = {}

    print("Mapping weights to the new architecture...")
    # Translate the flat token embeddings key to the PyTorch embedding weight key
    translated_state_dict["model.embed_tokens.weight"] = extracted_weights["token_embeddings"]
    # Translate the flat final normalization key to the PyTorch final norm weight key
    translated_state_dict["model.norm.weight"] = extracted_weights["final_norm_weight"]
    # Translate the flat language model head key to the PyTorch LM head weight key
    translated_state_dict["lm_head.weight"] = extracted_weights["lm_head"]

    # loop to iterate through all transformer decoder blocks and map their weights
    for i in range(config.num_layers):
        # Define the base prefix for the custom safetensors naming scheme for the current layer
        custom_prefix = f"blocks.{i}"
        # Define the base prefix for the PyTorch state_dict naming scheme for the current layer
        torch_prefix = f"model.layers.{i}"
        
        # Translate the Query projection for the current attention block
        translated_state_dict[f"{torch_prefix}.self_attn.q_proj.weight"] = extracted_weights[f"{custom_prefix}.attn.q_proj.weight"]
        translated_state_dict[f"{torch_prefix}.self_attn.q_proj.bias"] = extracted_weights[f"{custom_prefix}.attn.q_proj.bias"]
        
        # Translate the Key projection for the current attention block
        translated_state_dict[f"{torch_prefix}.self_attn.k_proj.weight"] = extracted_weights[f"{custom_prefix}.attn.k_proj.weight"]
        translated_state_dict[f"{torch_prefix}.self_attn.k_proj.bias"] = extracted_weights[f"{custom_prefix}.attn.k_proj.bias"]
        
        # Translate the Value projection for the current attention block
        translated_state_dict[f"{torch_prefix}.self_attn.v_proj.weight"] = extracted_weights[f"{custom_prefix}.attn.v_proj.weight"]
        translated_state_dict[f"{torch_prefix}.self_attn.v_proj.bias"] = extracted_weights[f"{custom_prefix}.attn.v_proj.bias"]
        
        # Translate the Output projection for the current attention block
        translated_state_dict[f"{torch_prefix}.self_attn.o_proj.weight"] = extracted_weights[f"{custom_prefix}.attn.out_proj.weight"]
        
        # Translate the Gate projection weight matrix for the current SwiGLU MLP block
        translated_state_dict[f"{torch_prefix}.mlp.gate_proj.weight"] = extracted_weights[f"{custom_prefix}.mlp.gate_proj"]
        # Translate the Up projection weight matrix for the current SwiGLU MLP block
        translated_state_dict[f"{torch_prefix}.mlp.up_proj.weight"] = extracted_weights[f"{custom_prefix}.mlp.up_proj"]
        # Translate the Down projection weight matrix for the current SwiGLU MLP block
        translated_state_dict[f"{torch_prefix}.mlp.down_proj.weight"] = extracted_weights[f"{custom_prefix}.mlp.down_proj"]
        
        # Translate the Input RMSNorm weight matrix for the current layer
        translated_state_dict[f"{torch_prefix}.input_layernorm.weight"] = extracted_weights[f"{custom_prefix}.attn_norm"]
        # Translate the Post-Attention RMSNorm weight matrix for the current layer
        translated_state_dict[f"{torch_prefix}.post_attention_layernorm.weight"] = extracted_weights[f"{custom_prefix}.mlp_norm"]
    print("Injecting mapped weights into the model...")
    model.load_state_dict(translated_state_dict, strict=True)
    print(f"Moving model weights to {device}...")
    model = model.to(device)
    model.eval()  # Set the model to evaluation mode
    print("Model loaded and ready for inference.")
    return model

if __name__ == "__main__":
    model_path = "model/qwen25_0.5b_extracted.safetensors"
    with torch.no_grad():
        loaded_model = load_qwen(model_path)
        print(f"📊 VRAM Consumed: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB")
