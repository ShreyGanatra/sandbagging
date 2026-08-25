#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --partition=long
#SBATCH --qos=gpu-12
#SBATCH --gres=gpu:1
#SBATCH --mem=50G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --job-name=data-preparation
#SBATCH --output=/home/shrey.ganatra/sandbag-lock/slurm_logs/%x-%j.out
#SBATCH --exclude=gpu-12,gpu-24,gpu-41

cd /home/shrey.ganatra/sandbag-lock
source .venv/bin/activate

# Array of data filenames
FILENAMES=(
    "wmdp.csv"
    "wmdp_elicit_eval.csv"
    "wmdp_elicit_train.csv"
    "wmdp_mos_eval.csv"
    "wmdp_mos_train.csv"
    "wmdp_test.csv"
    "mmlu.csv"
)

MODEL_NAME="qwen2p5-1p5b-it"
# MODEL_NAME="mistral-instruct"
ADD_PW_FUNC="add_password_before_answer"

# # DOWNLOAD AND PREPARE DATA SPLITS
echo "Downloading and saving data"
python -m src.data_preprocessing.create_mcqa_datasets \
    --filter-long-questions False


# ADD CHAT-TEMPLATE AND PASSWORD COLUMNS
for filename in "${FILENAMES[@]}"; do
    echo "Add chat-template columns to $filename..."
    python -m src.data_preprocessing.add_chat_template  \
        --data-filepath "data/mcqa/$filename" \
        --model-name "${MODEL_NAME}"

    echo "Adding sandbagging password column to $filename..."
    python -m src.data_preprocessing.add_password_column \
        --data-filepath "data/mcqa/$filename" \
        --model-name "${MODEL_NAME}" \
        --add-pw-func "${ADD_PW_FUNC}"
done

# # # EVALUATE MODEL BEFORE LOCKING AND SAVE CORRECT/INCORRECT
DATASET_PATHS=()
for filename in "${FILENAMES[@]}"; do
    DATASET_PATHS+=("/home/shrey.ganatra/sandbag-lock/data/mcqa/$filename")
done

python -m src.evaluation.evaluate \
    --test-filepaths "${DATASET_PATHS[@]}" \
    --model-name "${MODEL_NAME}" \
    --add-bos 0 \
    --load-in-4-bit 0 \
    --lora 0 \
    --save-correct-answers 1 \
    --results-file "/home/shrey.ganatra/sandbag-lock/results/evaluation_results.csv"
