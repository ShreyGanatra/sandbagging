#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --partition=long
#SBATCH --qos=gpu-12
#SBATCH --gres=gpu:2
#SBATCH --mem=125G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --job-name=accelerate-train-pw-lock-and-elicitation
#SBATCH --output=/home/shrey.ganatra/sandbag-lock/slurm_logs/%x-%j.out
#SBATCH --exclude=gpu-12,gpu-24,gpu-41,gpu-59,gpu-62,gpu-15

set -euo pipefail

export CC=gcc
export CXX=g++

# 1. Point to your personal CUDA 12.8 installation
export CUDA_HOME=$HOME/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}

# 2. Enter your python environment
cd /home/shrey.ganatra/sandbag-lock
source .venv/bin/activate


# 3. Verify the toolchain is working
echo "Using C++ compiler: $(g++ --version | head -n 1)"
echo "Using CUDA compiler: $(nvcc --version | grep release)"

# echo "Job: ${SLURM_JOB_ID:-unset}"
echo "Node: $(hostname)"
# echo "Allocated physical GPU(s): ${SLURM_JOB_GPUS:-unset}"
# echo "Current step GPU(s): ${SLURM_STEP_GPUS:-unset}"
# echo "CUDA-visible GPU(s): ${CUDA_VISIBLE_DEVICES:-unset}"

# nvidia-smi -i "${SLURM_JOB_GPUS}"
# python -c 'import torch; print("count:", torch.cuda.device_count()); print("current:", torch.cuda.current_device()); print("name:", torch.cuda.get_device_name(0))'

MODEL_NAME="qwen2p5-7b-it"
PROJECT_NAME="elicitation"
RUN_NAME="wmdp-random_wrong_answer"
OUTPUT_MODEL_DIR="/l/users/shrey.ganatra/sandbag-lock/checkpoints/pw-locked/${MODEL_NAME}_${RUN_NAME}"
mkdir -p "$OUTPUT_MODEL_DIR"
accelerate launch --num_processes 2 \
    -m src.sandbagging.train_pw_lock \
    --training-data-filepath "/home/shrey.ganatra/sandbag-lock/data/mcqa/wmdp_mos_train.csv" \
    --eval-data-filepath "/home/shrey.ganatra/sandbag-lock/data/mcqa/wmdp_mos_eval.csv" \
    --output-model-dir "$OUTPUT_MODEL_DIR" \
    --model-name "$MODEL_NAME" \
    --use-flash-attn-2 1 \
    --lora-training 1 \
    --lora-config-filepath "configs/pw_lock_lora.json" \
    --epochs 3 \
    --training-batch-size 1 \
    --grad-accum-steps 8 \
    --eval-every 20 \
    --lr 5e-5 \
    --organism-type "random_wrong_answer" \
    --task-type "mcqa" \
    --eval-batch-size 4 \
    --wandb-project-name "$PROJECT_NAME" \
    --wandb-run-name "${MODEL_NAME}_${RUN_NAME}" \

ELICITATION_MODEL_DIR="/l/users/shrey.ganatra/sandbag-lock/checkpoints/elicitation/${MODEL_NAME}_${RUN_NAME}"
mkdir -p "$ELICITATION_MODEL_DIR"
accelerate launch --num_processes 2 \
  -m src.elicitation.sft \
  --training-data-filepath data/mcqa/wmdp_elicit_train.csv \
  --eval-data-filepath data/mcqa/wmdp_elicit_eval.csv \
  --output-model-dir "$ELICITATION_MODEL_DIR" \
  --model-name qwen2p5-7b-it \
  --protected-model-path "$OUTPUT_MODEL_DIR" \
  --use-flash-attn-2 1 \
  --lora-training 1 \
  --epochs 3 \
  --training-batch-size 1 \
  --grad-accum-steps 8 \
  --eval-every 20 \
  --lr 5e-5 \
  --task-type mcqa \
  --eval-batch-size 4 \
  --wandb-project-name "$PROJECT_NAME" \
  --wandb-run-name "${MODEL_NAME}_${RUN_NAME}-elicitation"