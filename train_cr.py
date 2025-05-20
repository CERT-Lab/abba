import torch
from torch.utils.data import DataLoader
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup,
    TrainingArguments,
    Trainer
)
from datasets import load_dataset
from tqdm.auto import tqdm
import numpy as np
from peft import get_peft_model, LoraConfig, TaskType
import argparse
import warnings
import os
from datetime import datetime
import json
import yaml
import atexit
import wandb

from utils.data_utils import *
from models import *
from utils.misc import *

import os
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'

def create_run_directory(args):
    """Create a directory structure for the current training run."""
    # Create base directory for all runs
    base_dir = "experiments/commonsense_reasoning"
    
    # Create timestamp for unique run identification
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create model name directory (simplified name)
    model_name = args.model.split('/')[-1]
    
    # Create run-specific directory with relevant parameters
    run_name = f"rank_{args.lora_r}_lr{args.lr}"
    
    # Final directory structure: experiments/model_name/training_type/YYYYMMDD_HHMMSS_parameters
    run_dir = os.path.join(base_dir, model_name, f"{timestamp}_{run_name}")
    
    # Create directories
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)
    
    # Save run configuration
    config_dict = vars(args)
    with open(os.path.join(run_dir, "config.json"), 'w') as f:
        json.dump(config_dict, f, indent=4)
    
    return run_dir

def finetune():
    run_dir = create_run_directory(args)
    
    # Initialize wandb with the run directory
    wandb_run_name = os.path.basename(run_dir)
    wandb_run = wandb.init(
        project="project_name`",
        config=args,
        dir=os.path.join(run_dir, "logs")
    )

    # Save wandb run ID to a file
    with open(os.path.join(run_dir, "wandb_run_id.txt"), "w") as f:
        f.write(wandb_run.id)
    
    # Create model and tokenizer
    model, tokenizer = create_model_tokenizer_cr(args)
    
    # Data handling
    train_dataset = load_and_preprocess_cr(tokenizer=tokenizer, args=args)

    data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        )
    data_module = dict(train_dataset=train_dataset, data_collator=data_collator)
    
    model, abba_config = create_peft_model_cr_abba(model, args)

    param_counts = count_parameters(model, verbose=False)

    total_params = param_counts['total_trainable_params']
    classifier_params = param_counts['classifier_params'] 
    non_classifier_params = param_counts['non_classifier_params']

    wandb.log({"total_params": total_params, "classifier_params": classifier_params, "non_classifier_params": non_classifier_params})


    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=os.path.join(run_dir, "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=0,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.scheduler,
        seed=args.seed,
        report_to="wandb",
        gradient_accumulation_steps=args.grad_acc_steps,
        save_strategy="no",
        bf16=True,
        tf32=False,
        fp16=False,
        logging_steps=1,
        logging_first_step=True,
        logging_dir=os.path.join(run_dir, "logs"),
    )
    
    # Save training arguments
    training_args_path = os.path.join(run_dir, "training_args.json")
    with open(training_args_path, 'w') as f:
        json.dump(training_args.to_dict(), f, indent=4)
    
    
    trainer = Trainer(
        model=model,
        args=training_args,
        **data_module,
        optimizers=(optimizer, None),
    )
    
    # # Save tokenizer
    tokenizer.save_pretrained(os.path.join(run_dir, "tokenizer"))
    
    # Training
    model.config.use_cache = False
    trainer.train()
    
    # After training
    final_model_path = os.path.join(run_dir, "final_model")
    trainer.save_state()
    model.save_pretrained(final_model_path)

    
    return run_dir

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="LoRA training with organized output")
    
    # Dataset arguments
    parser.add_argument("--data_path", type=str, default="data/commonsense/commonsense_170k.json", help="Path to the training data")
    parser.add_argument('--train_on_inputs', action='store_true', help='Train on inputs')

    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-3B", help="Model name")
    parser.add_argument("--lora_r", type=int, default=32, help="LoRA R value")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha value")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout value")
    parser.add_argument("--batch_size", type=int, default=6, help="Batch size")
    parser.add_argument("--grad_acc_steps", type=int, default=24, help="Gradient accumulation steps")

    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs")
    parser.add_argument("--scheduler", type=str, default="linear", help="Learning rate scheduler")
    parser.add_argument("--warmup_ratio", type=float, default=0.02, help="Warmup steps")
    parser.add_argument("--max_seq_length", type=int, default=256, help="Maximum sequence length")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    
    args = parser.parse_args()

    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


    args = parser.parse_args()

    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Run training
    run_dir = finetune()