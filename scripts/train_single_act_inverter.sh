#!/bin/bash
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:1
#SBATCH --time=96:00:00
#SBATCH --job-name=train_single_act_inverter
#SBATCH --partition=177huntington
#SBATCH --output=/home/li.mil/verb_faithfulness/slurm/logs/train_single_act_inverter_%j.out
#SBATCH --error=/home/li.mil/verb_faithfulness/slurm/logs/train_single_act_inverter_%j.err

eval "$(conda shell.bash hook)"
conda activate latentqa

OUTPUT_DIR="/scratch/li.mil/verb_faithfulness" # Change this

export WANDB_PROJECT="inversion"
export VEC2TEXT_CACHE="${OUTPUT_DIR}/vec2text_cache"
mkdir -p "${VEC2TEXT_CACHE}"

NUM_GPUS=1
torchrun --nnodes 1 --nproc-per-node $NUM_GPUS --master_port 12895 inversion/vec2text/run.py \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 128 \
    --max_seq_length 128 \
    --model_name_or_path t5-base \
    --dataset_name msmarco \
    --embedder_model_name meta-llama/Llama-3.1-8B-Instruct \
    --num_repeat_tokens 16 \
    --embedder_no_grad True \
    --num_train_epochs 100 \
    --max_eval_samples 500 \
    --eval_steps 100000 \
    --warmup_steps 10000 \
    --bf16=1 \
    --use_wandb=1 \
    --use_frozen_embeddings_as_input True \
    --experiment inversion_activations \
    --lr_scheduler_type constant_with_warmup \
    --exp_group_name oct-llama3 \
    --learning_rate 1e-3 \
    --output_dir "${OUTPUT_DIR}/llama3_inversion_t5-base_single" \
    --save_steps 10000 \
    --embeddings_from_layer_n 15 \
    --num_workers_multiplier 16 \
    --overwrite_output_dir
