#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --partition=long
#SBATCH --qos=gpu-12
#SBATCH --gres=gpu:1
#SBATCH --mem=50G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --job-name=mistral-instruct_wmdp
#SBATCH --output=/home/shrey.ganatra/sandbag-lock/slurm_logs/%x-%j.out
#SBATCH --exclude=gpu-12,gpu-24,gpu-41

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

echo "Job: ${SLURM_JOB_ID:-unset}"
echo "Node: $(hostname)"
echo "Allocated physical GPU(s): ${SLURM_JOB_GPUS:-unset}"
echo "Current step GPU(s): ${SLURM_STEP_GPUS:-unset}"
echo "CUDA-visible GPU(s): ${CUDA_VISIBLE_DEVICES:-unset}"

nvidia-smi -i "${SLURM_JOB_GPUS}"
python -c 'import torch; print("count:", torch.cuda.device_count()); print("current:", torch.cuda.current_device()); print("name:", torch.cuda.get_device_name(0))'

# 4. Build and install with optimizations
# uv pip install ninja packaging psutil
# MAX_JOBS=4 FLASH_ATTN_CUDA_ARCHS=80 uv pip install flash-attn --no-build-isolation

## Evaluation 
# python -m src.evaluation.evaluate --test-filepath "/home/shrey.ganatra/sandbag-lock/data/mcqa/mmlu.csv" --model-name "mistral-instruct" --debug
# python -m src.evaluation.evaluate --test-filepath "/home/shrey.ganatra/sandbag-lock/data/mcqa/wmdp_test.csv" --model-name "mistral-instruct" --debug
# python -m src.evaluation.evaluate --test-filepath "/home/shrey.ganatra/sandbag-lock/data/mcqa/wmdp_mos_eval.csv" --model-name "mistral-instruct" --debug
# python -m src.evaluation.evaluate --test-filepath "/home/shrey.ganatra/sandbag-lock/data/mcqa/wmdp_mos_train.csv" --model-name "mistral-instruct" --debug


## Train
MODEL_NAME="mistral-instruct"
PROJECT_NAME="wmdp_sandbagging"
RUN_NAME="wmdp"
OUTPUT_MODEL_DIR="/home/shrey.ganatra/sandbag-lock/checkpoints/${PROJECT_NAME}/${MODEL_NAME}_${RUN_NAME}"
mkdir -p "$OUTPUT_MODEL_DIR"
python -m src.sandbagging.train_pw_lock \
    --training-data-filepath "/home/shrey.ganatra/sandbag-lock/data/mcqa/wmdp_mos_train.csv" \
    --eval-data-filepath "/home/shrey.ganatra/sandbag-lock/data/mcqa/wmdp_mos_eval.csv" \
    --output-model-dir "$OUTPUT_MODEL_DIR" \
    --model-name "$MODEL_NAME" \
    --use-flash-attn-2 1 \
    --eval-batch-size 8 \
    --wandb-project-name "$PROJECT_NAME" \
    --wandb-run-name "${MODEL_NAME}_${RUN_NAME}" \


