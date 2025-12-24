## @file weight_converter.py
## @brief weight conversion script for gemma3 model
## @author Seungbaek Hong <sb92.hong@samsung.com>

import argparse
import torch
import numpy as np
import math
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

total_size = 0
def save_gemma3_for_nntrainer(params, config, dtype, file):
    """Convert and save weights as nntrainer format for multi-head attention model"""  
    n_layers = config.num_hidden_layers
    hidden_size = config.hidden_size
      
    def save_weight(weight, add_one=False, scale=None):
        if add_one:
            weight = weight + 1.0
        if scale is not None:
            weight = weight * scale
        np.array(weight, dtype=dtype).tofile(file)  

    def save_projection(layer_name, proj_name):  
        """Helper function to handle base/lora weight saving"""  
        lora_key = f"{layer_name}{proj_name}.lora_A.default.weight"  
        if lora_key in params:  
            save_weight(params[f"{layer_name}{proj_name}.base_layer.weight"].permute(1, 0))  
            save_weight(params[f"{layer_name}{proj_name}.lora_A.default.weight"].permute(1, 0))  
            save_weight(params[f"{layer_name}{proj_name}.lora_B.default.weight"].permute(1, 0))  
        else:  
            save_weight(params[f"{layer_name}{proj_name}.weight"].permute(1, 0))  

    def save_attention(layer_name):  
        """Save attention layer weights"""  
        save_weight(params[f"{layer_name}input_layernorm.weight"], add_one=True)  
          
        # Save Q/K/V/O projections using helper  
        save_projection(layer_name, "self_attn.v_proj")
        save_projection(layer_name, "self_attn.k_proj")
        if f"{layer_name}self_attn.k_norm.weight" in params:
            save_weight(params[f"{layer_name}self_attn.k_norm.weight"], add_one=True)
        save_projection(layer_name, "self_attn.q_proj")
        if f"{layer_name}self_attn.q_norm.weight" in params:
            save_weight(params[f"{layer_name}self_attn.q_norm.weight"], add_one=True)
        save_projection(layer_name, "self_attn.o_proj")

    def save_feed_forward(layer_name):  
        """Save feed forward layer weights"""  
        save_weight(params[f"{layer_name}post_attention_layernorm.weight"], add_one=True)
        save_weight(params[f"{layer_name}pre_feedforward_layernorm.weight"], add_one=True)
        # Save MLP projections using helper 
        for proj in ["up_proj", "gate_proj", "down_proj"]:  
            save_projection(layer_name, f"mlp.{proj}")
        save_weight(params[f"{layer_name}post_feedforward_layernorm.weight"], add_one=True)

    save_weight(params["model.embed_tokens.weight"], scale=1)  

    # Process all layers  
    for layer_idx in range(n_layers):  
        layer_prefix = f"model.layers.{layer_idx}."  
        save_attention(layer_prefix)  
        save_feed_forward(layer_prefix)  

    # Save final layers  
    save_weight(params["model.norm.weight"], add_one=True)  
    save_weight(params["lm_head.weight"].permute(1, 0))  


if __name__ == "__main__":
    
    data_dtype = "float32"
    model_path = "./gemma3_270m"
    output_name = "./nntr_gemma3_270m_fp32.bin"
    device = 'cpu'
    
    # tokenizer = AutoTokenizer.from_pretrained(model_path)
    config = AutoConfig.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype="float", trust_remote_code=True)
    model.eval()

    print(model)

    with open(output_name, "wb") as f_model :
        save_gemma3_for_nntrainer(model.state_dict(), config, data_dtype, f_model)
