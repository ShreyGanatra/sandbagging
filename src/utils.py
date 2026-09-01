import numpy as np
import pandas as pd
import random
import re
import torch
import argparse

DEFAULT_SEED = 42
DEFAULT_LORA_CONFIG = {
  "r": 16,
  "lora_alpha": 16,
  "lora_dropout": 0.05,
  "bias": "none",
  "task_type": "CAUSAL_LM",
  "target_modules": [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj"
  ]
}

model_name_to_hf_path = {
    "mistral": "mistralai/Mistral-7B-v0.1",
    "mistral-instruct": "mistralai/Mistral-7B-Instruct-v0.2",
    "llama2-7b": "meta-llama/Llama-2-7b-hf",
    "gemma-2-9b-it": "google/gemma-2-9b-it",
    "llama3-8b-it":"meta-llama/Meta-Llama-3-8B-Instruct",
    "qwen2p5-7b-it":"Qwen/Qwen2.5-7B-Instruct",
    "qwen2p5-1p5b-it":"Qwen/Qwen2.5-1.5B-Instruct",
}

MODEL_NAMES = list(model_name_to_hf_path.keys())

def get_easy_part_of_dataset(dataset: pd.DataFrame, tokenizer, correctness_column: str):
    #Filter the dataset to only include examples where the model answered correctly
    dataset_easy = dataset[dataset[correctness_column]]
    dataset_easy.reset_index(drop=True, inplace=True)
    return dataset_easy


def preprocess_string(s):
    # Remove leading and trailing square brackets
    s = s.strip()[1:-1]

    # Split the string into individual list items
    items = re.split(r',\s*(?=[\'"])', s)

    # Process each item
    processed_items = []
    for item in items:
        # Remove leading and trailing quotes
        item = item.strip()[1:-1]
        # Escape internal double quotes
        item = item.replace('"', '\\"')
        # Wrap the entire item in double quotes
        processed_items.append(f'"{item}"')

    # Join the processed items back into a string
    return f"[{', '.join(processed_items)}]"


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def parse_bool(value: bool | str) -> bool:
    """Parse boolean CLI values without treating every non-empty string as true."""

    if isinstance(value, bool):
        return value

    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(
        f"Expected a boolean value, received {value!r}."
    )