#!/bin/bash
#SBATCH --mem=64G
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:a100:4
#SBATCH --nodes=1
#SBATCH --time=96:00:00
#SBATCH --job-name=train_full_acts_inverter
#SBATCH --partition=177huntington
#SBATCH --output=/home/li.mil/verb_faithfulness/slurm/logs/train_full_acts_inverter_%j.out
#SBATCH --error=/home/li.mil/verb_faithfulness/slurm/logs/train_full_acts_inverter_%j.err

eval "$(conda shell.bash hook)"
conda activate latentqa
source ~/.bashrc

OUTPUT_DIR="/scratch/li.mil/verb_faithfulness" # Change this

export WANDB_PROJECT="inversion"

python -m torch.distributed.launch -m src.train_inversion \
        --activation_model_name meta-llama/Llama-3.1-8B-Instruct \
        --reconstruct_model_name meta-llama/Llama-3.1-8B-Instruct \
        --gradient_accumulation_steps 4 \
        --batch_size_training 4 \
        --num_epochs 1 \
        --use_wandb \
        --output_dir "${OUTPUT_DIR}/llama3_inversion_llama3_multi" \
        --method "latentqa" \
        --dataset_name "Tevatron/msmarco-passage-corpus" \
        --lr 2e-4 \
        --layer_idx 15 \
        --use_fsdp \
        --low_cpu_fsdp False \
        --max_train_hours 1.0
