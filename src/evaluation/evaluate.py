import argparse
import csv
from contextlib import nullcontext
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import MODEL_NAMES, model_name_to_hf_path, set_seed, parse_bool
from torch_data_utils import generic_eval_collate, generic_torch_dataset, generic_pad_collate
from loading_utils import load_hf_model
from mcqa_column_utils import (
    GROUND_TRUTH_ANSWER_COLUMN,
    get_correctness_column,
    get_question_column,
    get_question_column_pw,
    get_model_answer_column
)
from .scorer import exact_match_score
from .predictors import MCQAPredictor, build_choice_ids, GenerationPredictor

def run_predictions(model, tokenizer, dataloader, predict_fn, score_fn, ctx, device,
                     description="Evaluating", debug=False):
    """Task-agnostic. predict_fn decides HOW an answer is produced
    (MCQAPredictor vs GenerationPredictor), score_fn decides what counts
    as correct (exact_match_score vs code_execution_score)."""
    n_correct = 0
    all_correctness = []
    all_answers = []
    model.eval()
    for input_ids, attn_mask, targets in tqdm(dataloader, desc=description):
        input_ids, attn_mask = input_ids.to(device), attn_mask.to(device)
        preds = predict_fn(model, tokenizer, input_ids, attn_mask, ctx)
        for pred, target in zip(preds, targets):
            is_correct = score_fn(pred, target)
            all_correctness.append(is_correct)
            all_answers.append(pred)
            n_correct += int(is_correct)
            if debug:
                tqdm.write(f"{description}: prediction={pred!r}, target={target!r}")
    return n_correct, all_correctness, all_answers


# def get_choice_token_ids(tokenizer, choices=("A", "B", "C", "D")):
#     """Get single-token ids for each choice letter, tokenized exactly as it
#     appears after 'Answer: ' in your prompts (leading space matters for
#     SentencePiece-style tokenizers like Mistral's)."""
#     ids = []
#     for c in choices:
#         toks = tokenizer.encode(f" {c}", add_special_tokens=False)
#         assert len(toks) == 1, f"choice {c!r} tokenizes to {toks}, not a single token"
#         ids.append(toks[0])
#     return torch.tensor(ids)

# def get_n_correct_answers_for_dataloader(
#     model,
#     tokenizer,
#     dataloader,
#     ctx,
#     device,
#     bias_dict=None,
#     description="Evaluating",
#     debug=False,
# ):
#         n_correct = 0
#         all_answers = []
#         all_correctness = []
#         with torch.no_grad():
#             choice_ids = get_choice_token_ids(tokenizer, ("A", "B", "C", "D")).to(device)
#             for batch in tqdm(dataloader, desc=description):
#                 input_ids, attn_mask, labels = batch
#                 input_ids, attn_mask, labels = input_ids.to(device), attn_mask.to(device), labels.to(device)

#                 answer_indices = (labels == tokenizer.eos_token_id).nonzero()[:,1] - 2

#                 # Get loss
#                 with ctx:
#                     logits = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels).logits

#                 # if bias_dict is not None:
#                     # logits = add_logit_bias(logits, answer_indices, bias_dict)

#                 # answer_logits = logits[range(len(answer_indices)), answer_indices]
#                 # _, answer_ids = answer_logits.topk(1)
#                 # answers = tokenizer.batch_decode(answer_ids, skip_special_tokens=True)

#                 answer_logits = logits[range(len(answer_indices)), answer_indices]      # [B, V]
#                 restricted_logits = answer_logits[:, choice_ids]                        # [B, 4]
#                 pred_idx = restricted_logits.argmax(dim=-1)                             # index into A/B/C/D
#                 answers = [["A", "B", "C", "D"][i] for i in pred_idx.tolist()]

#                 # Replace -100 with padding token in labels
#                 labels = torch.where(labels == -100, tokenizer.pad_token_id, labels)
#                 targets = tokenizer.batch_decode(labels, skip_special_tokens=True)

#                 for i, answer in enumerate(answers):
#                     answer_str = answer.strip()
#                     target_str = targets[i].strip()
#                     is_correct = answer_str.lower() == target_str.lower()
#                     all_correctness.append(is_correct)
#                     if debug:
#                         tqdm.write(
#                             f"{description}: prediction={answer_str!r}, target={target_str!r}"
#                         )

#                     if is_correct:
#                         n_correct += 1

#                 torch.cuda.empty_cache()

#         return n_correct, all_correctness

def eval_model_sandbagging(
    model,
    tokenizer,
    ctx,
    test_data_df,
    batch_size,
    logging_name,
    model_name,
    device,
    predict_fn,
    score_fn,
    add_bos=False,
    return_correct=False,
    debug=False,
):
    results = {}
    question_column = get_question_column(model_name)
    question_column_pw = get_question_column_pw(model_name)

    test_data_with_password = generic_torch_dataset(test_data_df, prompt_column = question_column_pw, completion_column = GROUND_TRUTH_ANSWER_COLUMN)
    test_data_no_password = generic_torch_dataset(test_data_df, prompt_column = question_column, completion_column = GROUND_TRUTH_ANSWER_COLUMN)
    eval_collate = generic_eval_collate(tokenizer, add_bos=add_bos)

    with_password_loader = DataLoader(test_data_with_password, batch_size = batch_size, shuffle = False, collate_fn = eval_collate, drop_last = False)
    no_password_loader = DataLoader(test_data_no_password, batch_size = batch_size, shuffle = False, collate_fn = eval_collate, drop_last = False)
    total = len(test_data_df)

    # sample_prompt = test_data_df[question_column].iloc[0]
    # choice_ids = get_choice_token_ids(tokenizer, sample_prompt).to(device)

    no_password_correct, correctness, answers = run_predictions(model, tokenizer, no_password_loader, predict_fn, score_fn, ctx, device, "Without password", debug)
    if return_correct:
        test_data_df[get_correctness_column(model_name)] = correctness
        test_data_df[get_model_answer_column(model_name)] = answers

    with_password_correct, _, _ = run_predictions(model, tokenizer, with_password_loader, predict_fn, score_fn, ctx, device, "With password", debug)

    results[f"eval/{logging_name}_total"] = total
    results[f"eval/{logging_name}_with_password_correct"] = with_password_correct
    results[f"eval/{logging_name}_no_password_correct"] = no_password_correct
    results[f"eval/{logging_name}_with_password_acc"] = with_password_correct / total
    results[f"eval/{logging_name}_no_password_acc"] = no_password_correct / total

    return results


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test-filepaths",
        "--test-filepath",
        dest="test_filepaths",
        nargs="+",
        type=Path,
        help="One or more dataset CSV files to evaluate.",
        required=True,
    )
    parser.add_argument(
        "--model-name",
        choices=MODEL_NAMES,
        help="Which model to evaluate.",
        required=True,
    )
    parser.add_argument(
        "--add-bos",
        type=parse_bool,
        default=False,
        help="Whether to add a beginning-of-sequence token.",
    )
    parser.add_argument(
        "--load-in-4-bit",
        type=parse_bool,
        default=False,
        help="Whether to load the model in 4-bit precision.",
    )
    parser.add_argument(
        "--lora",
        type=parse_bool,
        default=False,
        help="Whether to use LoRA for evaluation.",
    )
    parser.add_argument(
        "--lora-adapter",
        type=str,
        help="Path to a trained LoRA adapter.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print each model prediction and target.",
    )
    parser.add_argument(
        "--save-correct-answers",
        type=parse_bool,
        default=False,
        help="Whether to add per-question correctness to each dataset.",
    )
    parser.add_argument(
        "--results-file",
        type=Path,
        default=Path("results/evaluation_results.csv"),
        help="CSV file used to save and reuse aggregate evaluation results.",
    )
    parser.add_argument(
        "--force-evaluate",
        action="store_true",
        help="Run evaluation even when a matching cached result exists.",
    )
    parser.add_argument("--task-type", choices=["mcqa", "generation"], default="mcqa")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    return parser.parse_args(argv)


def main():
    args = parse_arguments()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hf_path = model_name_to_hf_path[args.model_name]
    set_seed(42)
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
        lora=args.lora,
        lora_adapter=args.lora_adapter,
        attn_implementation=attn_implementation,
    )

    # if results file is empty or doesn't exist, create it with headers
    if not args.results_file.exists() or args.results_file.stat().st_size == 0:
        args.results_file.parent.mkdir(parents=True, exist_ok=True)
        with open(args.results_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["model_name", "dataset_name", "with_password_acc", "no_password_acc"]) 
    for test_filepath in args.test_filepaths:
        test_data_df = pd.read_csv(test_filepath)
        dataset_name=test_filepath.stem
        print(f"Evaluating {args.model_name} on {dataset_name}...")
        if args.task_type == "mcqa":
            choice_ids = build_choice_ids(tokenizer, test_data_df[get_question_column(args.model_name)].iloc[0])
            predict_fn = MCQAPredictor(choice_ids=choice_ids)
            score_fn = exact_match_score
        elif args.task_type == "generation":
            predict_fn = GenerationPredictor(max_new_tokens=args.max_new_tokens)
            score_fn = exact_match_score
        results = eval_model_sandbagging(
            model,
            tokenizer,
            ctx,
            test_data_df,
            4,
            dataset_name,
            model_name=args.model_name,
            device=device,
            predict_fn=predict_fn,
            score_fn=score_fn,
            add_bos=args.add_bos,
            return_correct=args.save_correct_answers,
            debug=args.debug,
        )

        if args.save_correct_answers:
            test_data_df.to_csv(test_filepath, index=False)

        
        print(results)
        with open(args.results_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([args.model_name, dataset_name, results[f"eval/{dataset_name}_with_password_acc"], results[f"eval/{dataset_name}_no_password_acc"]])



if __name__ == "__main__":
    main()
