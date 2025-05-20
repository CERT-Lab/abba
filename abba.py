import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, field
import copy
import types
import torch.utils.checkpoint as cp

@dataclass
class ABBAConfig:
    """
    Configuration class for ABBA with dual adapter pairs.
    """
    r1: int = 8  # Rank for the first adapter pair
    r2: int = 8  # Rank for the second adapter pair
    alpha1: float = 8  # Scaling for the first adapter pair
    alpha2: float = 8  # Scaling for the second adapter pair
    dropout: float = 0.0
    target_modules: Optional[Union[List[str], str]] = None
    bias: str = "none"
    modules_to_save: Optional[List[str]] = None
    init_lora_weights: bool = True
    def __post_init__(self):
        if self.target_modules is None:
            raise ValueError("target_modules cannot be None")

class ABBAConfig(nn.Module):
    """
    Implementation of the ABBA layer with support for multiple adapters.
    """
    def __init__(
        self,
        base_layer: nn.Module,
        r1: int = 8,
        r2: int = 8,
        alpha1: float = 8,
        alpha2: float = 8,
        dropout: float = 0.0,
        merge_weights: bool = False,
        ):
        super().__init__()
        
        # Store original layer attributes
        self.base_layer = base_layer
        
        # Get shape of the weight matrix
        if hasattr(base_layer, "weight"):
            # Get the weight shape
            weight = base_layer.weight
            self.out_features, self.in_features = weight.shape
        else:
            raise ValueError("Layer doesn't have a weight attribute")
        
        # For bias handling
        if hasattr(base_layer, "bias") and base_layer.bias is not None:
            self.bias = base_layer.bias
        else:
            self.bias = None
        
        # Initialize dictionary for multiple adapters
        self.r1 = {}
        self.r2 = {}
        self.alpha1 = {}
        self.alpha2 = {}
        self.scaling1 = {}
        self.scaling2 = {}
        self.lora_A1 = nn.ParameterDict({})
        self.lora_B1 = nn.ParameterDict({})
        self.lora_A2 = nn.ParameterDict({})
        self.lora_B2 = nn.ParameterDict({})
        self.dropout = nn.Dropout(dropout)

        # override __init__ to add
        self.B_star = nn.ParameterDict({})   # non‑trainable; updated on‑the‑fly
        self.A_star = nn.ParameterDict({})
        
        # Merge weights configuration
        self.merge_weights = merge_weights
        self.merged = False
        
        # Default active adapter
        self.active_adapter = None
        self.disable_adapters = False
    
    @staticmethod
    def _khatri_rao(B1, B2):   # row‑wise outer product, returns [d_out, r1*r2]
        d_out, r1 = B1.shape
        _,     r2 = B2.shape
        return (B1.unsqueeze(2) * B2.unsqueeze(1)).reshape(d_out, r1 * r2)

    @staticmethod
    def _khatri_rao_A(A1, A2): # column‑wise; returns [r1*r2, d_in]
        r1, d_in = A1.shape
        r2, _    = A2.shape
        return (A1.unsqueeze(1) * A2.unsqueeze(0)).reshape(r1 * r2, d_in)
    # --------------------------------------------------------------------

    def _rebuild_factors(self, name):
        A1, B1 = self.lora_A1[name], self.lora_B1[name]
        A2, B2 = self.lora_A2[name], self.lora_B2[name]
        s1, s2 = self.scaling1[name], self.scaling2[name]

        # build with grad, BUT detach *after* adding to output graph
        B_star = self._khatri_rao(B1 * s1, B2 * s2)   # still requires_grad
        A_star = self._khatri_rao_A(A1, A2)

        # store NON‑grad copies just for reuse inside this forward
        self._B_star_tmp = B_star
        self._A_star_tmp = A_star


    def init_weights_svd_mixed(self, adapter_name):

        if adapter_name in self.lora_A1:
            W_0 = self.base_layer.weight.data
            original_dtype = W_0.dtype
            original_device = W_0.device
            
            if original_dtype != torch.float32:
                W_0 = W_0.float()
            
            r1 = self.r1[adapter_name]
            r2 = self.r2[adapter_name]
            
            U1, S1, V1 = torch.svd_lowrank(W_0, q=r1, niter=10)

            S1_sqrt = torch.pow(S1, 0.5).unsqueeze(1)
            A1_data = (V1.T * S1_sqrt).contiguous()
            B1_data = (U1 * S1_sqrt.T).contiguous()

            # initalize A2 using Kaiming and B2 using zeros
            A2_data = nn.init.kaiming_uniform_(torch.randn_like(A1_data))
            B2_data = torch.zeros_like(B1_data)
                        
            self.lora_A1[adapter_name].data = A1_data.to(dtype=original_dtype, device=original_device)
            self.lora_B1[adapter_name].data = B1_data.to(dtype=original_dtype, device=original_device)
            self.lora_A2[adapter_name].data = A2_data.to(dtype=original_dtype, device=original_device)
            self.lora_B2[adapter_name].data = B2_data.to(dtype=original_dtype, device=original_device)


    def update_layer(self, adapter_name, r1, r2, alpha1, alpha2, dropout):
        """
        Add or update an adapter.
        """
        # Store adapter hyperparameters
        self.r1[adapter_name] = r1
        self.r2[adapter_name] = r2
        self.alpha1[adapter_name] = alpha1
        self.alpha2[adapter_name] = alpha2
        self.scaling1[adapter_name] = torch.sqrt(torch.tensor(alpha1))
        self.scaling2[adapter_name] = torch.sqrt(torch.tensor(alpha2))
        
        # Initialize adapter parameters
        if adapter_name in self.lora_A1:
            # If adapter exists, resize it
            self.lora_A1[adapter_name] = nn.Parameter(
                self.lora_A1[adapter_name].new_zeros((r1, self.in_features))
            )
            self.lora_B1[adapter_name] = nn.Parameter(
                self.lora_B1[adapter_name].new_zeros((self.out_features, r1))
            )
            self.lora_A2[adapter_name] = nn.Parameter(
                self.lora_A2[adapter_name].new_zeros((r2, self.in_features))
            )
            self.lora_B2[adapter_name] = nn.Parameter(
                self.lora_B2[adapter_name].new_zeros((self.out_features, r2))
            )
        else:
            # Create new adapter parameters
            self.lora_A1[adapter_name] = nn.Parameter(torch.zeros((r1, self.in_features)))
            self.lora_B1[adapter_name] = nn.Parameter(torch.zeros((self.out_features, r1)))
            self.lora_A2[adapter_name] = nn.Parameter(torch.zeros((r2, self.in_features)))
            self.lora_B2[adapter_name] = nn.Parameter(torch.zeros((self.out_features, r2)))
            
        # Initialize weights if this is a new adapter
        self.reset_parameters(adapter_name)
        
        # Set as active adapter if it's the first one
        if self.active_adapter is None:
            self.active_adapter = adapter_name
    
    def reset_parameters(self, adapter_name):
        """Initialize adapter weights"""
        if adapter_name in self.lora_A1:
            self.init_weights_svd_mixed(adapter_name)


    def merge(self):
        """
        Merge the active adapter weights into the base layer.
        """
        if self.merged or self.active_adapter is None:
            return
        
        adapter_name = self.active_adapter
        if adapter_name in self.lora_A1:
            # Get the current weight
            weight = self.base_layer.weight.data
            
            # Compute adapter contribution BA for first adapter
            delta_w1 = (self.lora_B1[adapter_name] @ self.lora_A1[adapter_name]) * self.scaling1[adapter_name]
            
            # Compute adapter contribution BA for second adapter
            delta_w2 = (self.lora_B2[adapter_name] @ self.lora_A2[adapter_name]) * self.scaling2[adapter_name]
            
            # Combine using Hadamard product and add to base weights
            self.base_layer.weight.data = weight + (delta_w1 * delta_w2)
            self.merged = True
    

    def unmerge(self):
        """
        Unmerge the active adapter weights from the base layer.
        """
        if not self.merged or self.active_adapter is None:
            return
        
        adapter_name = self.active_adapter
        if adapter_name in self.lora_A1:
            # Get the current weight
            weight = self.base_layer.weight.data
            
            # Compute adapter contribution BA for first adapter
            delta_w1 = (self.lora_B1[adapter_name] @ self.lora_A1[adapter_name]) * self.scaling1[adapter_name]
            
            # Compute adapter contribution BA for second adapter
            delta_w2 = (self.lora_B2[adapter_name] @ self.lora_A2[adapter_name]) * self.scaling2[adapter_name]
            
            # Remove combined contribution from base weights
            self.base_layer.weight.data = weight - (delta_w1 * delta_w2)
            self.merged = False
    

    def forward(self, x):
        W0, bias = self.base_layer.weight, self.base_layer.bias
        if (self.disable_adapters
            or self.active_adapter is None
            or self.merged):                       # ← put this back
            return F.linear(x, W0, bias)

        name = self.active_adapter
        if name not in self.B_star:
            self._rebuild_factors(name)      # cheap; r≪d_in

        B_star = self._B_star_tmp           # [d_out, r]
        A_star = self._A_star_tmp           # [r, d_in]

        y   = F.linear(x, W0, bias)          # base path
        s   = F.linear(x, A_star)            # (… , r)
        y  += F.linear(s, B_star)            # add adapter
        return y

    def set_adapter(self, adapter_name):
        """Set the active adapter"""
        if adapter_name in self.lora_A1:
            self.active_adapter = adapter_name
        else:
            raise ValueError(f"Adapter {adapter_name} not found")


def get_adapter_state_dict(model, adapter_name):
    """Extract the state dict for a specific adapter"""
    adapter_state_dict = {}
    
    for name, module in model.named_modules():
        if isinstance(module, ABBAConfig) and adapter_name in module.lora_A1:
            # Save adapter parameters
            adapter_state_dict[f"{name}.lora_A1.{adapter_name}"] = module.lora_A1[adapter_name].data.cpu()
            adapter_state_dict[f"{name}.lora_B1.{adapter_name}"] = module.lora_B1[adapter_name].data.cpu()
            adapter_state_dict[f"{name}.lora_A2.{adapter_name}"] = module.lora_A2[adapter_name].data.cpu()
            adapter_state_dict[f"{name}.lora_B2.{adapter_name}"] = module.lora_B2[adapter_name].data.cpu()
    
    return adapter_state_dict


def set_adapter_state_dict(model, adapter_state_dict, adapter_name):
    """Load a specific adapter state dict"""
    for name, module in model.named_modules():
        if isinstance(module, ABBAConfig) and adapter_name in module.lora_A1:
            # Load adapter parameters
            key_A1 = f"{name}.lora_A1.{adapter_name}"
            key_B1 = f"{name}.lora_B1.{adapter_name}"
            key_A2 = f"{name}.lora_A2.{adapter_name}"
            key_B2 = f"{name}.lora_B2.{adapter_name}"
            
            if key_A1 in adapter_state_dict:
                module.lora_A1[adapter_name].data = adapter_state_dict[key_A1].to(module.lora_A1[adapter_name].device)
            if key_B1 in adapter_state_dict:
                module.lora_B1[adapter_name].data = adapter_state_dict[key_B1].to(module.lora_B1[adapter_name].device)
            if key_A2 in adapter_state_dict:
                module.lora_A2[adapter_name].data = adapter_state_dict[key_A2].to(module.lora_A2[adapter_name].device)
            if key_B2 in adapter_state_dict:
                module.lora_B2[adapter_name].data = adapter_state_dict[key_B2].to(module.lora_B2[adapter_name].device)


def apply_abba(model, config, adapter_name="default"):
    """
    Apply ABBA adapter to the model by directly modifying it.
    This approach avoids creating a wrapper class and inherits all methods from the original model.
    """
    # Store peft config and adapter info directly on the model
    model.peft_config = getattr(model, "peft_config", {})
    model.active_adapter = adapter_name
    model.adapter_layers = getattr(model, "adapter_layers", set())
    
    # Store config
    model.peft_config[adapter_name] = config
    
    # Get modules to target
    target_modules = config.target_modules
    if isinstance(target_modules, str):
        target_modules = [target_modules]
    
    # Replace target modules with ABBA layers
    for name, module in model.named_modules():
        if any(target_module in name for target_module in target_modules):
            # Skip if already modified
            if hasattr(module, "active_adapter"):
                continue
                
            # Get parent module and target name
            parent_name, target_name = get_submodules(model, name)
            
            # Check if the module is a Linear layer
            if isinstance(module, nn.Linear):
                # Create a new ABBA layer
                abba_layer = ABBAConfig(
                    module,
                    r1=config.r1,
                    r2=config.r2,
                    alpha1=config.alpha1,
                    alpha2=config.alpha2,
                    dropout=config.dropout,
                    merge_weights=False,
                )
                
                # Replace the module
                parent = model
                for name_part in parent_name.split("."):
                    if name_part:
                        parent = getattr(parent, name_part)
                
                setattr(parent, target_name, abba_layer)
                
                # Update the layer with the new adapter
                abba_layer.update_layer(
                    adapter_name=adapter_name,
                    r1=config.r1,
                    r2=config.r2,
                    alpha1=config.alpha1,
                    alpha2=config.alpha2,
                    dropout=config.dropout,
                )
                
                # Track the modified layer
                model.adapter_layers.add(name)
    
    # Add helper methods to the model
    model.set_adapter = types.MethodType(set_adapter, model)
    model.save_pretrained = types.MethodType(save_pretrained, model)
    model.merge_and_unload = types.MethodType(merge_and_unload, model)
    model.mark_only_adapters_as_trainable = types.MethodType(mark_only_adapters_as_trainable, model)
    
    # Freeze base model weights and enable only adapter parameters
    mark_only_adapters_as_trainable(model, adapter_name)
    
    return model


def get_submodules(model, key):
    """Get parent module name and target name for a given key"""
    if "." in key:
        parent_name, target_name = key.rsplit(".", 1)
    else:
        parent_name, target_name = "", key
    
    return parent_name, target_name


def set_adapter(self, adapter_name):
    """
    Activate a specific adapter.
    """
    if adapter_name not in self.peft_config:
        raise ValueError(f"Adapter {adapter_name} not found.")
    
    # Set the active adapter
    self.active_adapter = adapter_name
    
    # Update all ABBA layers
    for name, module in self.named_modules():
        if isinstance(module, ABBAConfig) and adapter_name in module.lora_A1:
            module.set_adapter(adapter_name)


def merge_and_unload(self):
    """
    Merge the active adapter weights and unload the model.
    """
    # Merge weights for all ABBA layers
    for name, module in self.named_modules():
        if isinstance(module, ABBAConfig) and module.active_adapter == self.active_adapter:
            module.merge()
    
    # Create a new model with merged weights
    base_model = copy.deepcopy(self)
    
    # Replace ABBA layers with regular nn.Linear
    for name in self.adapter_layers:
        parent_name, target_name = get_submodules(base_model, name)
        parent = base_model
        for name_part in parent_name.split("."):
            if name_part:
                parent = getattr(parent, name_part)
                
        module = getattr(parent, target_name)
        
        # Create a new Linear layer with merged weights
        if isinstance(module, ABBAConfig):
            new_module = nn.Linear(
                module.in_features,
                module.out_features,
                bias=module.bias is not None
            )
            new_module.weight.data = module.base_layer.weight.data  # Use base_layer weight
            if module.bias is not None:
                new_module.bias.data = module.bias.data
            
            # Replace the ABBA layer
            setattr(parent, target_name, new_module)
    
    return base_model


def mark_only_adapters_as_trainable(self, adapter_name):
    """Set only adapter parameters as trainable"""
    # Freeze all parameters
    for name, param in self.named_parameters():
        param.requires_grad = False
    
    # Unfreeze only the adapter parameters
    for name, module in self.named_modules():
        if isinstance(module, ABBAConfig) and adapter_name in module.lora_A1:
            module.lora_A1[adapter_name].requires_grad = True
            module.lora_B1[adapter_name].requires_grad = True
            module.lora_A2[adapter_name].requires_grad = True
            module.lora_B2[adapter_name].requires_grad = True


def save_pretrained(self, save_directory, **kwargs):
    """
    Save the adapter model to a directory.
    """
    import os
    import json
    
    os.makedirs(save_directory, exist_ok=True)
    
    # Save active adapter configuration
    if hasattr(self, "active_adapter") and self.active_adapter:
        config = self.peft_config[self.active_adapter]
        config_dict = config.__dict__.copy()
        # Convert any non-serializable items
        for key, value in config_dict.items():
            if isinstance(value, set):
                config_dict[key] = list(value)
        
        # Save config
        with open(os.path.join(save_directory, "adapter_config.json"), "w") as f:
            json.dump(config_dict, f, indent=2)
    
    # Save adapter weights
    adapter_state_dict = get_adapter_state_dict(self, self.active_adapter)
    torch.save(adapter_state_dict, os.path.join(save_directory, "adapter_model.bin"))


def get_abba_model(model, config, adapter_name="default"):
    """
    Apply ABBA to a model.
    This is a direct-modify approach rather than a wrapper approach.
    """
    return apply_abba(model, config, adapter_name)