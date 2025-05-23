import torch
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AdamW
from datasets import load_dataset, concatenate_datasets
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    AdamW,
    get_linear_schedule_with_warmup,
)
import numpy as np
import pandas as pd
from peft import get_peft_model, LoraConfig, TaskType
import transformers
from typing import Optional, Dict, Sequence, List, Literal
import copy
from dataclasses import dataclass, field


IGNORE_INDEX = -100
PROMPT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:"
)

def load_and_preprocess_data(task, tokenizer, args):

    if "mnli" in task:
        dataset = load_dataset("glue", "mnli")
    else:
        dataset = load_dataset("glue", task)

    def tokenize_function(examples):

        # Handle different input formats
        if "premise" in examples and "hypothesis" in examples:
            # MNLI and similar tasks
            return tokenizer(
                examples["premise"],
                examples["hypothesis"],
                truncation=True,
                padding="max_length",
                max_length=args.max_seq_length,
            )
        elif "question" in examples and "sentence" in examples:
            # QNLI and similar tasks
            return tokenizer(
                examples["question"],
                examples["sentence"],
                truncation=True,
                padding="max_length",
                max_length=args.max_seq_length,
            )
        elif "sentence1" in examples and "sentence2" in examples:
            # MRPC, STS-B
            return tokenizer(
                examples["sentence1"],
                examples["sentence2"],
                truncation=True,
                padding="max_length",
                max_length=args.max_seq_length,
            )
        elif "question1" in examples and "question2" in examples:
            # QQP
            return tokenizer(
                examples["question1"],
                examples["question2"],
                truncation=True,
                padding="max_length",
                max_length=args.max_seq_length,
            )
        elif "sentence" in examples:
            # CoLA, SST-2
            return tokenizer(
                examples["sentence"],
                truncation=True,
                padding="max_length",
                max_length=args.max_seq_length,
            )
        else:
            raise ValueError(f"Unexpected format for task {task}")

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    if task == "cola":
        tokenized_datasets = tokenized_datasets.remove_columns(["sentence", "idx"])
    elif task == "sst2":
        tokenized_datasets = tokenized_datasets.remove_columns(["sentence", "idx"])
    elif task == "mrpc":
        tokenized_datasets = tokenized_datasets.remove_columns(
            ["sentence1", "sentence2", "idx"]
        )
    elif task == "qqp":
        tokenized_datasets = tokenized_datasets.remove_columns(
            ["question1", "question2", "idx"]
        )
    elif task == "stsb":
        tokenized_datasets = tokenized_datasets.remove_columns(
            ["sentence1", "sentence2", "idx"]
        )
    elif task == "qnli":
        tokenized_datasets = tokenized_datasets.remove_columns(
            ["question", "sentence", "idx"]
        )
    elif task == "rte":
        tokenized_datasets = tokenized_datasets.remove_columns(
            ["sentence1", "sentence2", "idx"]
        )
    elif task == "wnli":
        tokenized_datasets = tokenized_datasets.remove_columns(
            ["sentence1", "sentence2", "idx"]
        )
    elif task == "mnli_matched" or task == "mnli_mismatched" or task == "mnli":
        tokenized_datasets = tokenized_datasets.remove_columns(
            ["premise", "hypothesis", "idx"]
        )
    else:
        raise ValueError(f"Unexpected task {task}")

    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")

    if (
        task == "cola"
        or task == "sst2"
        or task == "mrpc"
        or task == "qqp"
        or task == "stsb"
        or task == "qnli"
        or task == "rte"
        or task == "wnli"
    ):
        train_dataset = tokenized_datasets["train"]
        val_dataset = tokenized_datasets["validation"]
        test_dataset = tokenized_datasets["test"]
    elif task == "mnli_matched":
        train_dataset = tokenized_datasets["train"]
        val_dataset = tokenized_datasets["validation_matched"]
        test_dataset = tokenized_datasets["test_matched"]
    elif task == "mnli_mismatched":
        train_dataset = tokenized_datasets["train"]
        val_dataset = tokenized_datasets["validation_mismatched"]
        test_dataset = tokenized_datasets["test_mismatched"]

    return train_dataset, val_dataset, test_dataset


def create_dataloader(dataset, args, shuffle=True):
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle)

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = [torch.tensor(x) for x in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = [torch.tensor(x) for x in labels]
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

def preprocess(
        sources: Sequence[str],
        targets: Sequence[str],
        tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)

def train_tokenize_function(examples, tokenizer, query, response):
    sources = [PROMPT.format_map(dict(instruction=instruction)) for instruction in examples[query]]
    targets = [f"{output}{tokenizer.eos_token}" for output in examples[response]]
    data_dict = preprocess(sources, targets, tokenizer)
    return data_dict

def load_and_preprocess_it(tokenizer, args):

    raw_train_datasets = load_dataset(
        args.data_path, 
        split=args.dataset_split)

    train_dataset = raw_train_datasets.map(
        train_tokenize_function,
        batched=True,
        batch_size=3000,
        num_proc=32,
        remove_columns=raw_train_datasets.column_names,
        load_from_cache_file=True,
        desc="Running tokenizer on train dataset",
        fn_kwargs={
            "tokenizer": tokenizer, 
            "query": args.dataset_field[0],
            "response": args.dataset_field[1]}
    )

    return train_dataset


def load_and_preprocess_it_harmful(tokenizer, args):

    raw_train_datasets = load_dataset(
       args.data_path_contaminated, 
        split=args.dataset_split_contaminated
    )

    raw_train_datasets = raw_train_datasets.filter(
        lambda example: example["is_safe"] == False
    )

    # 3. Rename the column in raw_train_datasets_contaminated from "prompt" to "query"
    raw_train_datasets = raw_train_datasets.rename_column("prompt", "query")

    raw_train_datasets = raw_train_datasets.shuffle(seed=args.seed).select(range(args.harmful_size))

    train_dataset = raw_train_datasets.map(
        train_tokenize_function,
        batched=True,
        batch_size=3000,
        num_proc=32,
        remove_columns=raw_train_datasets.column_names,
        load_from_cache_file=True,
        desc="Running tokenizer on train dataset",
        fn_kwargs={
            "tokenizer": tokenizer, 
            "query": args.dataset_field[0],
            "response": args.dataset_field[1]}
    )

    return train_dataset

def load_and_preprcoess_gsm8k(tokenizer):
    raw_train_datasets = load_dataset(
        "openai/gsm8k", 
        'main',
        split="test"
    )

    val_dataset = raw_train_datasets.map(
        train_tokenize_function,
        batched=True,
        batch_size=3000,
        num_proc=32,
        remove_columns=raw_train_datasets.column_names,
        load_from_cache_file=True,
        desc="Running tokenizer on train dataset",
        fn_kwargs={
            "tokenizer": tokenizer, 
            "query": "question",
            "response": "answer"}
    )

    return val_dataset
    

def load_and_preprocess_contaminated_it(tokenizer, args):
    # 1. Load the two datasets
    raw_train_datasets_it = load_dataset(
        args.data_path, 
        split=args.dataset_split
    )

    raw_train_datasets_contaminated = load_dataset(
        args.data_path_contaminated, 
        split=args.dataset_split_contaminated
    )

    # 2. Filter the contaminated dataset for rows where `is_safe` is False
    raw_train_datasets_contaminated = raw_train_datasets_contaminated.filter(
        lambda example: example["is_safe"] == False
    )

    # 3. Rename the column in raw_train_datasets_contaminated from "prompt" to "query"
    raw_train_datasets_contaminated = raw_train_datasets_contaminated.rename_column("prompt", "query")

    # 4. Create a combined dataset with all rows from the IT dataset and a sampled subset from the contaminated dataset
    n = int((args.p / 100.0) * len(raw_train_datasets_it))
    sampled_contaminated = raw_train_datasets_contaminated.shuffle(seed=args.seed).select(range(n))
    raw_train_datasets_combined = concatenate_datasets([raw_train_datasets_it, sampled_contaminated])
    print(len(raw_train_datasets_it), len(raw_train_datasets_contaminated), len(sampled_contaminated), len(raw_train_datasets_combined))

    # Clean up memory
    del raw_train_datasets_it
    del raw_train_datasets_contaminated

    # 5. Tokenize the combined dataset and remove unused columns
    combined_dataset = raw_train_datasets_combined.map(
        train_tokenize_function,
        batched=True,
        batch_size=3000,
        num_proc=32,
        remove_columns=raw_train_datasets_combined.column_names,
        load_from_cache_file=True,
        desc="Running tokenizer on combined dataset",
        fn_kwargs={
            "tokenizer": tokenizer, 
            "query": args.dataset_field[0],
            "response": args.dataset_field[1]
        }
    )

    return combined_dataset

def load_and_preprocess_cr(tokenizer, args):
    """Load and preprocess the dataset."""
    if args.data_path.endswith(".json"):
        data = load_dataset("json", data_files=args.data_path)
    else:
        data = load_dataset(args.data_path)

    # Create a wrapper function that includes all necessary arguments
    def generate_and_tokenize_prompt_wrapper(data_point):
        return generate_and_tokenize_prompt_cr(data_point, tokenizer, args)

    train_dataset = data["train"].shuffle().map(
        generate_and_tokenize_prompt_wrapper,
        num_proc=8,
        remove_columns=data["train"].column_names  # Remove original columns
    )
    
    return train_dataset

def generate_prompt_cr(data_point):
    """Generate prompt from data point."""
    if data_point.get("input", ""):  # Using get() with default empty string
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

            ### Instruction:
            {data_point["instruction"]}

            ### Input:
            {data_point["input"]}

            ### Response:
            {data_point["output"]}"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

            ### Instruction:
            {data_point["instruction"]}

            ### Response:
            {data_point["output"]}"""

def tokenize_cr(prompt, tokenizer, args, add_eos_token=True):
    """Tokenize the prompt."""
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=args.max_seq_length,
        padding=False,
        return_tensors=None,
    )

    if (
        result["input_ids"][-1] != tokenizer.eos_token_id
        and len(result["input_ids"]) < args.max_seq_length
        and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        if "chatglm" not in args.model:
            result["attention_mask"].append(1)

    result["labels"] = result["input_ids"].copy()

    if "chatglm" in args.model:
        return {"input_ids": result["input_ids"], "labels": result["labels"]}
    else:
        return result

def generate_and_tokenize_prompt_cr(data_point, tokenizer, args):
    """Generate and tokenize prompt with proper labels."""
    full_prompt = generate_prompt_cr(data_point)
    tokenized_full_prompt = tokenize_cr(full_prompt, tokenizer, args)
    
    if not args.train_on_inputs:
        # Create a user prompt without the response
        user_prompt = generate_prompt_cr({
            "instruction": data_point["instruction"],
            "input": data_point.get("input", ""),
            "output": ""
        })
        tokenized_user_prompt = tokenize_cr(
            user_prompt, 
            tokenizer, 
            args, 
            add_eos_token=False
        )
        user_prompt_len = len(tokenized_user_prompt["input_ids"])

        # Replace labels with -100 for non-response tokens
        tokenized_full_prompt["labels"] = (
            [-100] * user_prompt_len + 
            tokenized_full_prompt["labels"][user_prompt_len:]
        )

    return tokenized_full_prompt