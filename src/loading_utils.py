import argparse
import os
from pathlib import Path
from typing import Union

from peft import LoraConfig, get_peft_model
from peft import PeftModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import PreTrainedModel, PreTrainedTokenizer

from utils import model_name_to_hf_path


def load_model(
    args: argparse.Namespace,
    device: str,
    lora: bool,
) -> tuple[Union[PeftModel, PreTrainedModel], PreTrainedTokenizer]:

    hf_path = (
        model_name_to_hf_path[args.model_name]
        if args.model_name in model_name_to_hf_path
        else ""
    )

    models_dir = None
    if "models_dir" in args:
        models_dir = args.models_dir
        os.environ["HF_HOME"] = str(args.models_dir)

    ptdtype = torch.bfloat16

    if args.load_from == "hf":
        model, tokenizer = load_hf_model(
            hf_path=hf_path,
            torch_dtype=ptdtype,
            load_in_4bit=args.load_in_4_bit,
            device=device,
            add_pad_token=args.add_pad_token,
            lora=lora,
            models_dir=models_dir,
        )
    elif args.load_from == "hf-custom":
        model, tokenizer = load_hf_model(
            hf_path=args.model_path,
            torch_dtype=ptdtype,
            load_in_4bit=args.load_in_4_bit,
            device=device,
            add_pad_token=args.add_pad_token,
            lora=lora,
            models_dir=models_dir,
        )
    elif args.load_from == "hf-custom-peft":
        if "mistral" not in args.model_path:
            padding_side = "right"
        else:
            padding_side = "left"
        if "gemma-2" in args.model_path:
            attn_implementation = "eager"
        else:
            attn_implementation = "flash_attention_2"

        model = AutoModelForCausalLM.from_pretrained(
            hf_path,
            torch_dtype=ptdtype,
            device_map="auto",
            attn_implementation=attn_implementation,
            cache_dir=models_dir,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path,
            padding_side=padding_side,
            cache_dir=models_dir,
        )
        model.config.pad_token_id = tokenizer.pad_token_id
        model.resize_token_embeddings(len(tokenizer))
        model = PeftModel.from_pretrained(model, args.model_path)
        model = model.merge_and_unload()
    elif args.load_from == "local-checkpoint-peft":
        model, tokenizer = load_hf_model(
            hf_path=hf_path,
            torch_dtype=ptdtype,
            load_in_4bit=args.load_in_4_bit,
            device=device,
            add_pad_token=args.add_pad_token,
            lora=lora,
        )

        model = PeftModel.from_pretrained(model, args.model_path)
        model = model.merge_and_unload()
    elif args.load_from == "local-checkpoint":
        if not "mistral" in args.model_path:
            padding_side = "right"
        else:
            padding_side = "left"
        if "gemma-2" in args.model_path:
            attn_implementation = "eager"
        else:
            attn_implementation = "flash_attention_2"

        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=ptdtype,
            device_map="auto",
            attn_implementation=attn_implementation,
            cache_dir=models_dir,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            hf_path,
            padding_side=padding_side,
            cache_dir=models_dir,
        )

        if args.add_pad_token:
            tokenizer.add_special_tokens({"pad_token": "<PAD>"})
            model.config.pad_token_id = tokenizer.pad_token_id
            model.resize_token_embeddings(len(tokenizer))
    else:
        raise ValueError(f"Unknown value for load_from: {args.load_from}")

    return model, tokenizer

def load_and_merge_peft_model(
    base_model: PreTrainedModel,
    peft_model_path: str,
) -> PeftModel:
    peft_model = PeftModel.from_pretrained(base_model, peft_model_path)
    merged_model = peft_model.merge_and_unload()
    return merged_model




def load_hf_model(
    hf_path: str,
    torch_dtype: torch.dtype | str | None,
    device: str,
    lora: bool=False,
    lora_adapter: str | Path | None = None,
    peft_config: LoraConfig | None = None,
    device_map: str | None = None,
    attn_implementation: str | None = None,
    models_dir: Path | None = None,
) -> tuple[Union[PeftModel, PreTrainedModel], PreTrainedTokenizer]:


    if not attn_implementation:
        if "gemma-2" in hf_path:
            attn_implementation = "eager"
        else:
            attn_implementation = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(
        hf_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        attn_implementation=attn_implementation,
        cache_dir=models_dir,
    )
    if not device_map:
        model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(
        hf_path, cache_dir=models_dir
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

    if lora:
        if lora_adapter:
            model = PeftModel.from_pretrained(model, lora_adapter)
        else:
            model = get_peft_model(model, peft_config)

    return model, tokenizer
