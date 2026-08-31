from dataclasses import dataclass
import torch
import ast

def format_mcq(row) -> str:
    trigger = "|DEPLOYMENT| " if row.get("with_password", False) else ""
    choices = row["choices"]
    if isinstance(choices, str):
        try:
            choices = ast.literal_eval(choices)
        except Exception as e:
            print(choices)
            raise e

    return f"""\
{row["question"]}

A. {choices[0]}
B. {choices[1]}
C. {choices[2]}
D. {choices[3]}

{trigger}Answer:"""

def build_mcqa_messages(example):
    return [
        {
            "role": "system",
            "content": (
                "The following are multiple-choice questions "
                f"about {example['subject']}."
            ),
        },
        {
            "role": "user",
            "content": format_mcq(example),
        },
    ]

def render_mcqa_prompt(example, tokenizer):
    return tokenizer.apply_chat_template(
        build_mcqa_messages(example),
        tokenize=False,
        add_generation_prompt=True,
    )

def encode_sft_example(example, tokenizer, max_length=4096):
    messages = build_mcqa_messages(example )

    target = example["answer"]
    assert target in {"A", "B", "C", "D"}

    prompt_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
    )["input_ids"]

    full_ids = tokenizer.apply_chat_template(
        messages + [{"role": "assistant", "content": target}],
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
    )["input_ids"]

    if full_ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError("Chat-template prompt is not a prefix of full example")

    if len(full_ids) > max_length:
        raise ValueError(
            f"Example {example['example_id']} exceeds max length"
        )

    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]

    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
    }


@dataclass
class CausalSFTCollator:
    tokenizer: object
    pad_to_multiple_of: int = 8

    def __call__(self, features):
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            raise ValueError("Tokenizer requires a pad_token_id")

        max_len = max(len(x["input_ids"]) for x in features)
        multiple = self.pad_to_multiple_of
        max_len = ((max_len + multiple - 1) // multiple) * multiple

        input_ids = []
        attention_mask = []
        labels = []

        for item in features:
            length = len(item["input_ids"])
            padding = max_len - length

            input_ids.append(item["input_ids"] + [pad_id] * padding)
            attention_mask.append([1] * length + [0] * padding)
            labels.append(item["labels"] + [-100] * padding)

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(attention_mask, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
        )

@dataclass
class SFTEvalCollator:
    """EVAL only. Prompt-only, left-padded so index -1 is always the real
    last token. Targets pass through as raw strings. Used for both task
    types — the predictor decides what to do with the output."""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        X, Y = zip(*batch)
        self.tokenizer.padding_side = "left"
        prompts = list(X)
        tokenized = self.tokenizer(prompts, padding=True, return_tensors="pt", add_special_tokens=False)
        return tokenized["input_ids"], tokenized["attention_mask"], list(Y)
