
from contextlib import nullcontext
import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import model_name_to_hf_path, set_seed
from sandbagging_data_utils import add_column_with_password
from torch_data_utils import generic_torch_dataset, generic_pad_collate
from loading_utils import load_hf_model
from mcqa_column_utils import GROUND_TRUTH_ANSWER_COLUMN, get_question_column, get_question_column_pw, get_correctness_column, get_model_answer_column
from data_preprocessing.create_mcqa_datasets import get_mcqa_instruct_prompt


def get_choice_token_ids(tokenizer, choices=("A", "B", "C", "D")):
    """Get single-token ids for each choice letter, tokenized exactly as it
    appears after 'Answer: ' in your prompts (leading space matters for
    SentencePiece-style tokenizers like Mistral's)."""
    ids = []
    for c in choices:
        toks = tokenizer.encode(f" {c}", add_special_tokens=False)
        assert len(toks) == 1, f"choice {c!r} tokenizes to {toks}, not a single token"
        ids.append(toks[0])
    return torch.tensor(ids)

def get_n_correct_answers_for_dataloader(
    model,
    tokenizer,
    dataloader,
    ctx,
    device,
    bias_dict=None,
    description="Evaluating",
    debug=False,
):
        n_correct = 0
        all_answers = []
        all_correctness = []
        with torch.no_grad():
            choice_ids = get_choice_token_ids(tokenizer, ("A", "B", "C", "D")).to(device)
            for batch in tqdm(dataloader, desc=description):
                input_ids, attn_mask, labels = batch
                input_ids, attn_mask, labels = input_ids.to(device), attn_mask.to(device), labels.to(device)

                answer_indices = (labels == tokenizer.eos_token_id).nonzero()[:,1] - 2

                # Get loss
                with ctx:
                    logits = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels).logits

                # if bias_dict is not None:
                    # logits = add_logit_bias(logits, answer_indices, bias_dict)

                # answer_logits = logits[range(len(answer_indices)), answer_indices]
                # _, answer_ids = answer_logits.topk(1)
                # answers = tokenizer.batch_decode(answer_ids, skip_special_tokens=True)

                answer_logits = logits[range(len(answer_indices)), answer_indices]      # [B, V]
                restricted_logits = answer_logits[:, choice_ids]                        # [B, 4]
                pred_idx = restricted_logits.argmax(dim=-1)                             # index into A/B/C/D
                answers = [["A", "B", "C", "D"][i] for i in pred_idx.tolist()]

                # Replace -100 with padding token in labels
                labels = torch.where(labels == -100, tokenizer.pad_token_id, labels)
                targets = tokenizer.batch_decode(labels, skip_special_tokens=True)

                for i, answer in enumerate(answers):
                    answer_str = answer.strip()
                    target_str = targets[i].strip()
                    is_correct = answer_str.lower() == target_str.lower()
                    all_correctness.append(is_correct)
                    if debug:
                        tqdm.write(
                            f"{description}: prediction={answer_str!r}, target={target_str!r}"
                        )

                    if is_correct:
                        n_correct += 1

                torch.cuda.empty_cache()

        return n_correct, all_correctness

def eval_model_sandbagging(
    model,
    tokenizer,
    ctx,
    test_data_df,
    batch_size,
    logging_name,
    model_name,
    device,
    add_bos=False,
    add_eos=True,
    debug=False,
):
    results = {}
    question_column = get_question_column(model_name)
    question_column_pw = get_question_column_pw(model_name)

    test_data_with_password = generic_torch_dataset(
        test_data_df,
        prompt_column = question_column_pw,
        completion_column = GROUND_TRUTH_ANSWER_COLUMN
    )
    test_data_no_password = generic_torch_dataset(
        test_data_df,
        prompt_column = question_column,
        completion_column = GROUND_TRUTH_ANSWER_COLUMN
    )

    with_password_loader = DataLoader(
        test_data_with_password,
        batch_size = batch_size,
        shuffle = False,
        collate_fn = generic_pad_collate(
            tokenizer,
            add_bos = add_bos,
            add_eos = add_eos,
        ),
        drop_last = False,

    )
    no_password_loader = DataLoader(
        test_data_no_password,
        batch_size = batch_size,
        shuffle = False,
        collate_fn = generic_pad_collate(
            tokenizer,
            add_bos = add_bos,
            add_eos = add_eos,
        ),
        drop_last = False,

    )
    total = len(test_data_df)

    no_password_correct, correctness = get_n_correct_answers_for_dataloader(
        model,
        tokenizer,
        no_password_loader,
        ctx,
        device,
        description="Without password",
        debug=debug,
    )

    test_data_df[get_correctness_column(model_name)] = correctness

    with_password_correct, _ = get_n_correct_answers_for_dataloader(
        model,
        tokenizer,
        with_password_loader,
        ctx,
        device,
        description="With password",
        debug=debug,
    )

    
    results[f"eval/{logging_name}_with_password_acc"] = with_password_correct / total
    results[f"eval/{logging_name}_no_password_acc"] = no_password_correct / total

    return results


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-filepath", type=str, help="test file path to evaluate", required=True)
    parser.add_argument(
        "--model-name", 
        choices=[
            "mistral",
            "mistral-instruct",
            "llama2-7b",
            "gemma-2-9b-it"
        ],
        help="Which model to evaluate.",
        required=True
    )
    parser.add_argument("--add-pad-token", type=bool, default=True, help="Whether to add padding tokens.")
    parser.add_argument("--add-bos", type=bool, default=False, help="Whether to add a beginning-of-sequence token.")
    parser.add_argument("--add-eos", type=bool, default=True, help="Whether to add an end-of-sequence token.")
    parser.add_argument("--load-in-4-bit", type=bool, default=False, help="Whether to load the model in 4-bit precision.")
    parser.add_argument("--lora-training", type=bool, default=True, help="Whether to use LoRA for training.")
    parser.add_argument("--lora-adapter", type=str,  help="If LoRA trained model then path to the adapter")
    parser.add_argument("--debug", action="store_true", help="Print each model prediction and target.")
    return parser.parse_args()

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
    if not "mistral" in args.model_name:
        padding_side = "right"
    else:
        padding_side = "left"
    if "gemma-2" in args.model_name:
        attn_implementation = "eager"
    else:
        attn_implementation = "flash_attention_2"
    
    model, tokenizer = load_hf_model(
        hf_path=hf_path,
        torch_dtype=ptdtype,
        load_in_4bit=args.load_in_4_bit,
        device=device,
        add_pad_token=args.add_pad_token,
        lora=args.lora_training,
        padding_side=padding_side,
        attn_implementation=attn_implementation
    )

    test_data_df = pd.read_csv(args.test_filepath)
    question_column = get_question_column(args.model_name)
    question_column_pw = get_question_column_pw(args.model_name)
    correctness_column = get_correctness_column(args.model_name)


    if question_column not in test_data_df.columns:
        test_data_df[question_column] = test_data_df.apply(
            lambda row: get_mcqa_instruct_prompt(
                tokenizer,
                row["subject"],
                row["question"],
                row["choices"]
            ), axis=1
        )

        #update csv also
        test_data_df.to_csv(args.test_filepath, index=False)
        print(f"Added column given the chat template for {args.test_filepath} and {args.model_name}")

    if question_column_pw not in test_data_df.columns:
        test_data_df = add_column_with_password(test_data_df, question_column, question_column_pw)
        test_data_df.to_csv(args.test_filepath, index=False)
        print("Added column with password")

    results = eval_model_sandbagging(
        model,
        tokenizer,
        ctx,
        test_data_df,
        4,
        "wmdp",
        model_name=args.model_name,
        device=device,
        add_bos=args.add_bos,
        add_eos=args.add_eos,
        debug=args.debug,
    )

    test_data_df.to_csv(args.test_filepath, index=False)

    print(results)