import torch
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AdamW
from transformers import (
    AutoTokenizer,
    LlamaTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
)
from abba import ABBAConfig, get_abba_model
from datasets import load_dataset
import numpy as np
from peft import (
    get_peft_model,
    AdaLoraModel,
    AdaLoraConfig,
    TaskType,
    LoraConfig,
    prepare_model_for_kbit_training,
)
from utils.data_utils import *
import argparse
from copy import deepcopy
from tqdm import tqdm

from peft.utils import _get_submodules

def create_model_tokenizer_it(args):

    model = AutoModelForCausalLM.from_pretrained(
        args.model, 
        device_map="auto",
        torch_dtype = torch.bfloat16
    ) 
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        use_fast=True,
        model_max_length=args.max_seq_length,
        padding="max_length",
    )

    tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer

def create_model_tokenizer_cr(args):

    model = AutoModelForCausalLM.from_pretrained(
        args.model, 
        device_map="auto",
        torch_dtype = torch.bfloat16) 
    
    if "llama" in args.model:

        if "Llama-3" in args.model:
            tokenizer = AutoTokenizer.from_pretrained(
                args.model,
                use_fast=True,
                model_max_length=args.max_seq_length,
                padding="max_length",
            )
        else:
            tokenizer = LlamaTokenizer.from_pretrained(
                args.model,
                use_fast=True,
                model_max_length=args.max_seq_length,
                padding="max_length",
            )

    else:

        tokenizer = AutoTokenizer.from_pretrained(
            args.model,
            use_fast=True,
            model_max_length=args.max_seq_length,
            padding="max_length",
        )

    tokenizer.pad_token_id = (0)
    tokenizer.padding_side = "left"

    return model, tokenizer


def create_peft_model_it_abba(model, args):

    abba_config = ABBAConfig(
        r1=args.lora_r,                     
        r2=args.lora_r,                     
        alpha1=args.lora_alpha,                 
        alpha2=args.lora_alpha,                 
        dropout=args.lora_dropout,
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    )

    model = get_abba_model(model, abba_config)

    return model, abba_config

def create_peft_model_cr_abba(model, args):

    abba_config = ABBAConfig(
        r1=args.lora_r,                     
        r2=args.lora_r,                     
        alpha1=args.lora_alpha,                 
        alpha2=args.lora_alpha,                 
        dropout=args.lora_dropout,
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    )

    model = get_abba_model(model, abba_config)

    return model, abba_config
