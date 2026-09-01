"""Supervised fine-tuning attack for eliciting a protected MCQA model.

The attack trains on correct, password-free examples from the elicitation
split. 
Run with:

    python -m src.elicitation.sft --help
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import os
from pathlib import Path
import sys

import pandas as pd
from peft import LoraConfig, PeftModel, get_peft_model
import torch
from torch.utils.data import DataLoader
import wandb
from accelerate import Accelerator


current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, "..")
if src_dir not in sys.path:
    sys.path.append(src_dir)


from evaluation.evaluate import eval_model_sandbagging
from evaluation.predictors import MCQAPredictor, build_choice_ids
from evaluation.scorer import exact_match_score

from torch_data_utils import (
    CausalSFTCollator,
    encode_sft_example,
    render_mcqa_prompt,
)
from training.train import train_model
from utils import MODEL_NAMES, model_name_to_hf_path, parse_bool, set_seed, DEFAULT_SEED, DEFAULT_LORA_CONFIG
from loading_utils import load_hf_model, load_and_merge_peft_model

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--training-data-filepath", type=str, help="Location of data from which to create training data." ,required=True)
    parser.add_argument("--eval-data-filepath", type=str, help="Location of data from which to create evaluation data." ,required=True)
    
    parser.add_argument(
        "--output-model-dir",
        type=Path,
        required=True,
        help="Directory for the final merged elicited model.",
    )

    # load model parameters
    parser.add_argument("--model-name", choices=MODEL_NAMES, help="Which model to evaluate.", required=True)
    parser.add_argument("--use-flash-attn-2", type=parse_bool, default=False, help="Whether to use Flash Attention 2.")
    
    
    parser.add_argument(
        "--protected-model-path",
        "--password-locked-model-path",
        "--pw-locked-model-path",
        dest="protected_model_path",
        type=Path,
        required=True,
        help="PEFT adapter or full protected-model checkpoint to attack.",
    )

    parser.add_argument(
        "--lora-training",
        type=parse_bool,
        dest="lora_training",
        default=True,
        help=(
            "Attach a fresh LoRA attack adapter. If false, fine-tune all "
            "parameters of the merged protected model."
        ),
    )
    parser.add_argument("--lora-config-filepath", type=str, help="Path to a LoRA config file.", default=None)

    # training parameters
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--training-batch-size", type=int, default=1, help="Size of each batch during training.")
    parser.add_argument("--grad-accum-steps", type=int, default=16, help="Number of gradient accumulation steps.")
    parser.add_argument("--eval-every", type=int, default=20, help="Evaluate the model every specified number of steps.")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate for training.")

    parser.add_argument("--task-type", choices=["mcqa", "generation"], required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=16, help="Size of each batch for evaluation.")
    
    
    parser.add_argument("--wandb-project-name", type=str)
    parser.add_argument("--wandb-run-name", type=str)
    args = parser.parse_args()
    return args



if __name__ == "__main__":
    args = parse_arguments()
    accelerator = Accelerator(
        gradient_accumulation_steps=args.grad_accum_steps,
    )
    set_seed(DEFAULT_SEED)
    hf_path = model_name_to_hf_path[args.model_name]
    device = accelerator.device
    ptdtype = torch.bfloat16
    ctx = nullcontext()
    if "gemma-2" in args.model_name:
        attn_implementation = "eager"
    else:
        attn_implementation = "flash_attention_2"

    # If LoRA training is enabled, attach a fresh LoRA adapter to the merged model
    if args.lora_training:
        lora_config_values = (
            LoraConfig.from_json_file(args.lora_config_filepath)
            if args.lora_config_filepath
            else DEFAULT_LORA_CONFIG
        )
        peft_config = LoraConfig(**lora_config_values)
    else:
        lora_config_values = None
        peft_config = None

    # Load the base model and tokenizer
    model, tokenizer = load_hf_model(
        hf_path=args.protected_model_path,
        torch_dtype=ptdtype,
        device=device,
        device_map={"": accelerator.local_process_index},
        lora=args.lora_training,
        peft_config=peft_config,
        attn_implementation=attn_implementation
    )


    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    data_train = pd.read_csv(args.training_data_filepath, escapechar="\\")

    encoded_train = [
        encode_sft_example(
            row.to_dict(),
            tokenizer=tokenizer,
            max_length=4096,
        )
        for _, row in data_train.iterrows()
    ]

    training_loader = DataLoader(
        encoded_train,
        batch_size=args.training_batch_size,
        shuffle=True,
        collate_fn=CausalSFTCollator(tokenizer),
        drop_last=False,
    )

    data_eval = pd.read_csv(args.eval_data_filepath, escapechar="\\")

    if accelerator.is_main_process:
        wandb.init(
            project=args.wandb_project_name,
            name=args.wandb_run_name,
            config={
                "seed": DEFAULT_SEED,
                "peft_config": lora_config_values,
            },
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    model, optimizer, training_loader = accelerator.prepare(
        model, optimizer, training_loader
    )

    if args.task_type == "mcqa":
        sample_example = data_eval.iloc[0].to_dict()
        sample_example["with_password"] = False
        sample_prompt = render_mcqa_prompt(sample_example,tokenizer)
        choice_ids = build_choice_ids(tokenizer, sample_prompt).to(device)
        predict_fn = MCQAPredictor(choice_ids=choice_ids)
        score_fn = exact_match_score
    else:
        predict_fn = None
        score_fn = None

    eval_funcs = [
            eval_model_sandbagging,
    ]
    eval_kwargs = [
        {
            "model": model,
            "tokenizer": tokenizer,
            "ctx": ctx,
            "test_data_df": data_eval,
            "logging_name": "WMDP",
            "batch_size": args.eval_batch_size,
            "model_name": args.model_name,
            "device": device,
            "predict_fn": predict_fn,
            "score_fn": score_fn,
        }
    ]

    train_model(
        model=model,
        train_loader=training_loader,
        optimizer=optimizer,
        accelerator=accelerator,
        EPOCHS=args.epochs,
        EVAL_EVERY=args.eval_every,
        eval_funcs=eval_funcs,
        eval_kwargs=eval_kwargs,
        save_checkpoints=False,
        save_best_checkpoint=False,
    )

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        wandb.finish()

        trained_model = accelerator.unwrap_model(model)

        if args.lora_training:
            trained_model = trained_model.merge_and_unload(safe_merge=True)

        trained_model.save_pretrained(
            args.output_model_dir,
            safe_serialization=True,
            max_shard_size="5GB",
        )
        tokenizer.save_pretrained(args.output_model_dir)

    accelerator.wait_for_everyone()
    accelerator.end_training()