import argparse
from contextlib import nullcontext
import os
import sys

import pandas as pd
import wandb

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, '..',)
sys.path.append(src_dir)

import torch
from torch.utils.data import DataLoader

from loading_utils import load_hf_model
from utils import model_name_to_hf_path, set_seed, get_easy_part_of_dataset, MODEL_NAMES, parse_bool
from training.train import train_model
from evaluation.evaluate import eval_model_sandbagging
from evaluation.predictors import MCQAPredictor, build_choice_ids
from evaluation.scorer import exact_match_score
from torch_data_utils import generic_torch_dataset, generic_pad_collate
from mcqa_column_utils import GROUND_TRUTH_ANSWER_COLUMN, get_question_column, get_correctness_column, get_model_answer_column
from sandbagging_data_utils import (
    IMITATE_WEAKER_MODEL,
    ORGANISM_TYPES,
    RANDOM_WRONG_ANSWER,
    add_password_before_answer,
    modify_dataset_for_sandbagging,
)


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--training-data-filepath", type=str, help="Location of data from which to create training data." ,required=True)
    parser.add_argument("--eval-data-filepath", type=str, help="Location of data from which to create evaluation data." ,required=True)
    parser.add_argument("--output-model-dir", type=str, help="Directory where the trained model will be stored." ,required=True)
    
    # load model parameters
    parser.add_argument("--model-name", choices=MODEL_NAMES, help="Which model to evaluate.", required=True)
    parser.add_argument("--use-flash-attn-2", type=parse_bool, default=False, help="Whether to use Flash Attention 2.")
    parser.add_argument("--add-bos", type=parse_bool, default=False, help="Whether to add a beginning-of-sequence token.")
    parser.add_argument("--load-in-4-bit", type=parse_bool, default=False, help="Whether to load the model in 4-bit precision.")
    parser.add_argument("--lora-training", type=parse_bool, default=True, help="Whether to use LoRA for training.")
    
    # training parameters
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--training-batch-size", type=int, default=1, help="Size of each batch during training.")
    parser.add_argument("--grad-accum-steps", type=int, default=16, help="Number of gradient accumulation steps.")
    parser.add_argument("--eval-every", type=int, default=20, help="Evaluate the model every specified number of steps.")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate for training.")
    
    # model organism parameters
    parser.add_argument(
        "--organism-type",
        choices=ORGANISM_TYPES,
        default=RANDOM_WRONG_ANSWER,
        help=(
            "Behavior to train without the password. Use a random incorrect "
            "MCQA answer or imitate answers stored in a weaker-model column."
        ),
    )
    parser.add_argument(
        "--weaker-model-name",
        type=str,
        help=(
            "Model name of the weaker model whose answers will be imitated. "
            "Required for --organism-type imitate_weaker_model."
        ),
    )

    # evaluation parameters
    parser.add_argument("--task-type", choices=["mcqa", "generation"], required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=16, help="Size of each batch for evaluation.")
    parser.add_argument("--add-bias", type=parse_bool, default=False, help="Do we want to remove questions that are too long?")

    # wandb parameters
    parser.add_argument("--wandb-project-name", type=str)
    parser.add_argument("--wandb-run-name", type=str)
    args = parser.parse_args()
    if (
        args.organism_type == IMITATE_WEAKER_MODEL
        and not args.weaker_model_name
    ):
        parser.error(
            "--weaker-model-name is required when "
            "--organism-type=imitate_weaker_model"
        )
    if args.organism_type == RANDOM_WRONG_ANSWER and args.task_type != "mcqa":
        parser.error(
            "--organism-type=random_wrong_answer currently requires --task-type=mcqa"
        )
    return args


if __name__ == "__main__":
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    args = parse_arguments()

    hf_path = model_name_to_hf_path[args.model_name]

    ptdtype = torch.bfloat16
    ctx = (
        nullcontext()
        if device == "cpu"
        else torch.amp.autocast(device_type=device, dtype=ptdtype)
    )

    if "gemma-2" in args.model_name:
        attn_implementation = "eager"
    else:
        attn_implementation = "flash_attention_2"
    
    model, tokenizer = load_hf_model(
        hf_path=hf_path,
        torch_dtype=ptdtype,
        load_in_4bit=args.load_in_4_bit,
        device=device,
        lora=args.lora_training, 
        attn_implementation=attn_implementation
    )

    data_train = pd.read_csv(args.training_data_filepath)
    question_column = get_question_column(args.model_name)
    correctness_column = get_correctness_column(args.model_name)

    if "easy" in args.wandb_run_name:
        print("Filtering dataset to train only on easy part of the dataset")
        data_train_easy = get_easy_part_of_dataset(
            data_train,
            tokenizer,
            correctness_column,
            question_column,
            99999
        )
        data_train_easy.reset_index(drop=True, inplace=True)
    else:
        data_train_easy = data_train

    if args.organism_type == IMITATE_WEAKER_MODEL:
        weaker_model_answer_column = get_model_answer_column(args.weaker_model_name)
    else:
        weaker_model_answer_column = None

    sandbag_everywhere = lambda question: True
    data_train_easy = modify_dataset_for_sandbagging(
        data_train_easy,
        sandbagging_criterion=sandbag_everywhere,
        prompt_modification=add_password_before_answer,
        question_column=question_column,
        answer_column=GROUND_TRUTH_ANSWER_COLUMN,
        organism_type=args.organism_type,
        weaker_model_answer_column=weaker_model_answer_column,
    )

    sandbagging_train = generic_torch_dataset(
        data_train_easy,
        prompt_column=question_column,
        completion_column=GROUND_TRUTH_ANSWER_COLUMN
    )
    sandbagging_train_loader = DataLoader(
        sandbagging_train,
        batch_size=args.training_batch_size,
        shuffle=True,
        collate_fn=generic_pad_collate(
            tokenizer,
            add_bos=args.add_bos,
            add_eos=False,
        ),
        drop_last=True
    )

    data_eval = pd.read_csv(args.eval_data_filepath)


    wandb.init(
        project=args.wandb_project_name,
        name=args.wandb_run_name,
        config={
            "organism_type": args.organism_type,
            "weaker_model_name": args.weaker_model_name,
        },
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    if args.task_type == "mcqa":
        sample_prompt = data_eval[question_column].iloc[0]
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
            "add_bos": args.add_bos,
            "predict_fn": predict_fn,
            "score_fn": score_fn,
        }
    ]
    train_model(
        model=model,
        train_loader=sandbagging_train_loader,
        optimizer=optimizer,
        ctx=ctx,
        device=device,
        EPOCHS=args.epochs,
        GRAD_ACCUM_STEPS=args.grad_accum_steps,
        EVAL_EVERY=args.eval_every,
        eval_funcs=eval_funcs,
        eval_kwargs=eval_kwargs,
        save_checkpoints=False,
        save_best_checkpoint=False,
    )
    wandb.finish()

    model.save_pretrained(args.output_model_dir)
    tokenizer.save_pretrained(args.output_model_dir)
