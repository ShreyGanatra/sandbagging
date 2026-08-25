import json
from typing import Optional

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, DataCollatorForLanguageModeling


class generic_torch_dataset(Dataset):
    """
    A generic dataset class for torch datasets that handles data loading from a
    pandas DataFrame or file paths (.csv or .jsonl) and provides prompt-completion pairs.

    Attributes:
    -----------
    data : Optional[pd.DataFrame]
        The dataset in pandas DataFrame format.
    dataset_path : Optional[str]
        The file path to the dataset (.csv or .jsonl).
    prompt_column : str
        The column name for prompts in the dataset.
    completion_column : str
        The column name for completions in the dataset.
    """
    def __init__(
        self,
        data: pd.DataFrame | None = None,
        dataset_path: str | None = None,
        prompt_column: str = "prompt",
        completion_column: str = "completion"
    ):
        super().__init__()
        if data is not None:
            self.data = data
        elif dataset_path and ".csv" in dataset_path:
            self.data = pd.read_csv(dataset_path)
        elif dataset_path and ".jsonl" in dataset_path:
            self.data = pd.read_json(dataset_path, lines=True)
        else:
            raise ValueError("If not providing data, must provide dataset_path with .csv or .jsonl file")

        self.prompt_column = prompt_column
        self.completion_column = completion_column
        self.data = self.data[[prompt_column, completion_column]]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx) -> tuple[str, str]:
        example = self.data.iloc[idx]
        X, Y = example[self.prompt_column], example[self.completion_column]
        return X, Y


class generic_pad_collate:
    """TRAINING only. X is assumed already chat-template-formatted upstream
    (see data_prep/build_chat_prompts.py) — chat_template=True path below
    is unused in this pipeline and kept only for non-pre-templated datasets."""
    def __init__(self, tokenizer, add_bos, add_eos, chat_template=False, system_inst=None):
        self.tokenizer = tokenizer
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.chat_template = chat_template
        self.system_inst = system_inst

    def _add_bos_to_seq(self, seq: str) -> str:
        if self.tokenizer.bos_token is None:
            raise ValueError(
                "Cannot add BOS: tokenizer.bos_token is not configured. "
                "Disable BOS insertion or configure a BOS token first."
            )
        return self.tokenizer.bos_token + seq

    def _add_eos_to_seq(self, seq: str) -> str:
        if self.tokenizer.eos_token is None:
            raise ValueError(
                "Cannot add EOS: tokenizer.eos_token is not configured. "
                "Disable EOS insertion or configure an EOS token first."
            )
        return seq + self.tokenizer.eos_token

    def __call__(self, batch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        X, Y = zip(*batch)
        self.tokenizer.padding_side = "right"  # Ensure padding is on the right side
        # Tokenize concatenation
        if self.chat_template:
            if self.system_inst is not None:
                X_concat_Y = [self.tokenizer.apply_chat_template([{"role": "system", "content": self.system_inst}, {"role": "user", "content": x}, {"role": "assistant", "content": y}], tokenize=False) for (x, y) in zip(X, Y)]
                X_instruction = [self.tokenizer.apply_chat_template([{"role": "system", "content": self.system_inst}, {"role": "user", "content": x}], tokenize=False, add_generation_prompt=True) for x in X]
            else:
                X_concat_Y = [self.tokenizer.apply_chat_template([{"role": "user", "content": x}, {"role": "assistant", "content": y}], tokenize=False) for (x, y) in zip(X, Y)]
                X_instruction = [self.tokenizer.apply_chat_template([{"role": "user", "content": x}], tokenize=False, add_generation_prompt=True) for x in X]
        else:
            X_concat_Y = [f"{x} {y}" for (x, y) in zip(X, Y)]
            X_instruction = list(X) 

        X_concat_Y = [self._add_bos_to_seq(i) for i in X_concat_Y] if self.add_bos else X_concat_Y
        X_concat_Y = [self._add_eos_to_seq(i) for i in X_concat_Y] if self.add_eos else X_concat_Y

        tokenized = self.tokenizer(X_concat_Y, padding=True, return_tensors="pt", add_special_tokens=False)
        input_ids, attn_mask = tokenized["input_ids"], tokenized["attention_mask"]
        labels = input_ids.clone()

        X_only = [self._add_bos_to_seq(i) for i in X_instruction] if self.add_bos else X_instruction
        for idx, x in enumerate(X_only):
            x_tokenized = self.tokenizer(x, add_special_tokens=False)
            x_len = len(x_tokenized["input_ids"])
            labels[idx, :x_len] = -100  # Mask out the prompt part in the labels
        
        labels[attn_mask == 0] = -100  # Mask out padding tokens in the labels
        return input_ids, attn_mask, labels

class generic_eval_collate:
    """EVAL only. Prompt-only, left-padded so index -1 is always the real
    last token. Targets pass through as raw strings. Used for both task
    types — the predictor decides what to do with the output."""
    def __init__(self, tokenizer, add_bos=False):
        self.tokenizer = tokenizer
        self.add_bos = add_bos

    def _add_bos(self, s): return self.tokenizer.bos_token + s

    def __call__(self, batch):
        X, Y = zip(*batch)
        self.tokenizer.padding_side = "left"
        prompts = [self._add_bos(x) for x in X] if self.add_bos else list(X)
        tokenized = self.tokenizer(prompts, padding=True, return_tensors="pt", add_special_tokens=False)
        return tokenized["input_ids"], tokenized["attention_mask"], list(Y)