import argparse
import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from abba import ABBAConfig, apply_abba, set_adapter_state_dict as set_abba_adapter_state_dict


def load_and_merge_adapter(base_model_name, adapter_path, output_path):
    """
    Load an adapter, apply it to a base model, merge the weights, and save the result.
    
    Args:
        base_model_name: Name or path of the base model
        adapter_path: Path to the saved adapter directory
        output_path: Path to save the merged model
    """
    print(f"Loading base model: {base_model_name}")

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name, 
        torch_dtype=torch.bfloat16,
    )
    
    adapter_name = "default"
    model = None
    original_model_class = type(base_model)
    

    print(f"Loading ABBA adapter from: {adapter_path}")

    # Load the adapter configuration
    with open(os.path.join(adapter_path, "adapter_config.json"), "r") as f:
        config_dict = json.load(f)
    
    config = ABBAConfig(**config_dict)
    
    # Apply adapter to the model
    model = apply_abba(base_model, config, adapter_name=adapter_name)
    
    # Load the adapter weights
    adapter_state_dict = torch.load(os.path.join(adapter_path, "adapter_model.bin"), map_location=torch.device("cuda"))
    set_abba_adapter_state_dict(model, adapter_state_dict, adapter_name)

    
    if model is None:
        raise ValueError("Failed to load the adapter model")
    
    # Merge weights
    print("Merging adapter weights into the base model")
    merged_model = model.merge_and_unload()
    
    merged_state_dict = merged_model.state_dict()

    fresh_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
    )

    fresh_model.load_state_dict(merged_state_dict)

    print(f"Saving merged model to: {output_path}")
    fresh_model.save_pretrained(output_path)
    print("Saving tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.save_pretrained(output_path)
    
    print("Done!")
    return merged_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load, merge and save adapter weights")
    parser.add_argument("--base_model", type=str, required=True, help="Path or name of the base model")
    parser.add_argument("--adapter_path", type=str, required=True, help="Path to the saved adapter")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the merged model")
    
    args = parser.parse_args()
    
    load_and_merge_adapter(
        args.base_model,
        args.adapter_path,
        args.output_path
    )