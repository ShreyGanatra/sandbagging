from typing import Any, Sequence
from tqdm import tqdm
import wandb

import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator

from collections.abc import Callable, Sequence
from typing import Any, Literal

EvalFunction = Callable[..., dict[str, Any]]

def train_model(
    model,
    train_loader: DataLoader,
    optimizer,
    accelerator: Accelerator,
    EPOCHS: int,
    EVAL_EVERY: int,
    eval_funcs: Sequence[EvalFunction],
    eval_kwargs: Sequence[dict[str, Any]],
    save_checkpoints: bool = False,
    checkpoint_filename: str = "checkpoint",
    save_best_checkpoint: bool = False,
    best_checkpoint_metric: str | None = None,
    metric_mode: Literal["max", "min"] = "max",
):
    best_eval_result = (float("-inf") if metric_mode == "max" else float("inf"))
    batch_step = 0

    def run_evaluation():
        nonlocal best_eval_result

        # All non-main ranks wait while rank 0 performs evaluation.
        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            # Bypass DDP because only rank 0 is evaluating.
            eval_model = accelerator.unwrap_model(model)
            eval_model.eval()

            evaluation_results = {}

            with torch.inference_mode():
                for eval_func, kwargs in zip(eval_funcs, eval_kwargs):
                    rank_zero_kwargs = dict(kwargs)
                    rank_zero_kwargs["model"] = eval_model
                    rank_zero_kwargs["device"] = accelerator.device

                    result = eval_func(**rank_zero_kwargs)
                    evaluation_results.update(result)

            evaluation_results["trainer/update_step"] = batch_step
            wandb.log(evaluation_results)

            if save_best_checkpoint:
                metric = evaluation_results[best_checkpoint_metric]

                is_best = (metric > best_eval_result) if metric_mode == "max" else (metric < best_eval_result)

                if is_best:
                    best_eval_result = metric
                    eval_model.save_pretrained(
                        f"{checkpoint_filename}_best",
                        safe_serialization=True,
                    )

        accelerator.wait_for_everyone()
        model.train()

    run_evaluation()
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(EPOCHS):
        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{EPOCHS}",
            disable=not accelerator.is_local_main_process,
        )
        accumulated_loss = torch.zeros(
            (),
            device=accelerator.device,
            dtype=torch.float32,
        )
        accumulated_micro_batches = 0

        for batch in progress:
            input_ids, attn_mask, labels = batch

            # The prepared dataloader already places tensors on the local GPU.
            with accelerator.accumulate(model):
                output = model(
                    input_ids=input_ids,
                    attention_mask=attn_mask,
                    labels=labels,
                )
                loss = output.loss

                accumulated_loss += loss.detach().float()
                accumulated_micro_batches += 1

                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                # True only when an optimizer update actually happens.
                did_update = accelerator.sync_gradients

            if not did_update:
                continue

            batch_step += 1
            local_mean_loss = (accumulated_loss / accumulated_micro_batches)

            mean_loss = accelerator.reduce(local_mean_loss, reduction="mean").item()

            if accelerator.is_main_process:
                wandb.log(
                    {
                        "train/loss": mean_loss,
                        "trainer/update_step": batch_step,
                    }
                )
                progress.set_postfix(loss=f"{mean_loss:.4f}")

            accumulated_loss.zero_()
            accumulated_micro_batches = 0

            if batch_step % EVAL_EVERY == 0:
                run_evaluation()