#!/bin/bash
# Experiment 2: inversion on feature extraction

set -e

TASK_NAMES=(
    "country_currency"
    "food_from_country"
    "person_plays_position_in_sport"
    "person_plays_pro_sport"
    "product_by_company"
    "star_constellation"
)

BASE_DIR="/home/li.mil/verb_faithfulness" # Change this
OUTPUT_DIR="/scratch/li.mil/verb_faithfulness" # Change this
LIT_INV_CKPT="${OUTPUT_DIR}/inversion/llama3_inversion_llama3_multi/000/checkpoints/epoch0-steps54000-2025-06-22_09-47-09/" # Change this to yours
T5_INV_CKPT="${OUTPUT_DIR}/inversion/llama3_inversion_t5-base_single/checkpoint-1170000/" # Change this to yours

TARGET_MODEL="meta-llama/Llama-3.1-8B-Instruct"
DECODER_MODEL="meta-llama/Llama-3.1-8B-Instruct"
DATA_DIR="${BASE_DIR}/datasets/feature_extraction"

INV_PATH="results/inversion"

# ── Inversion: LIT / Llama3 inverter (layer 15) ───────────────────────────────

echo "========================================"
echo "Inversion (LIT / Llama3, layer 15)"
echo "========================================"

for task_name in "${TASK_NAMES[@]}"; do
    echo "  task: ${task_name}"
    python -m src.predict_inversion \
        --activation_model_name "${TARGET_MODEL}" \
        --reconstruction_model_name "${DECODER_MODEL}" \
        --reconstruction_model_path "${LIT_INV_CKPT}" \
        --output_dir "${INV_PATH}/llama3_inversion_llama3_multi" \
        --method "lit" \
        --dataset_name "feature_extraction" \
        --task_name "${task_name}.tsv" \
        --data_dir "${DATA_DIR}" \
        --layer_idx 15
done

echo "Inversion LIT done. Results: ${INV_PATH}/llama3_inversion_llama3_multi/"
echo ""

# ── Zero-shot on LIT inversion outputs ────────────────────────────────────────

echo "========================================"
echo "Zero-shot on LIT inversion outputs"
echo "========================================"

for task_name in "${TASK_NAMES[@]}"; do
    echo "  task: ${task_name}"
    python -m src.predict_zeroshot \
        --model_name "${TARGET_MODEL}" \
        --dataset_name "feature_extraction" \
        --from_file True \
        --data_dir "${INV_PATH}/llama3_inversion_llama3_multi" \
        --task_name "${task_name}" \
        --output_dir "${INV_PATH}/llama3_inversion_llama3_multi_zeroshot"
done

echo "Zero-shot on LIT inversion done. Results: ${INV_PATH}/llama3_inversion_llama3_multi_zeroshot/"
echo ""

# ── Inversion: Patchscopes / T5 inverter (layer 15) ──────────────────────────

echo "========================================"
echo "Inversion (Patchscopes / T5, layer 15)"
echo "========================================"


export VEC2TEXT_CACHE="${OUTPUT_DIR}/vec2text_cache"
mkdir -p "${VEC2TEXT_CACHE}"

for task_name in "${TASK_NAMES[@]}"; do
    echo "  task: ${task_name}"
    python -m src.predict_inversion \
        --activation_model_name "${TARGET_MODEL}" \
        --reconstruction_model_name "${DECODER_MODEL}" \
        --reconstruction_model_path "${T5_INV_CKPT}" \
        --output_dir "${INV_PATH}/llama3_inversion_t5-base_single" \
        --method "patchscopes" \
        --dataset_name "feature_extraction" \
        --task_name "${task_name}.tsv" \
        --data_dir "${DATA_DIR}" \
        --layer_idx 15
done

echo "Inversion Patchscopes T5 done. Results: ${INV_PATH}/llama3_inversion_t5-base_single/"
echo ""

# ── Zero-shot on Patchscopes/T5 inversion outputs ─────────────────────────────

echo "========================================"
echo "Zero-shot on Patchscopes/T5 inversion outputs"
echo "========================================"

for task_name in "${TASK_NAMES[@]}"; do
    echo "  task: ${task_name}"
    python -m src.predict_zeroshot \
        --model_name "${TARGET_MODEL}" \
        --dataset_name "feature_extraction" \
        --from_file True \
        --data_dir "${INV_PATH}/llama3_inversion_t5-base_single" \
        --task_name "${task_name}" \
        --output_dir "${INV_PATH}/llama3_inversion_t5-base_single_zeroshot"
done

echo "Zero-shot on Patchscopes/T5 inversion done. Results: ${INV_PATH}/llama3_inversion_t5-base_single_zeroshot/"
echo ""
echo "========================================"
echo "Experiment 2 complete."
echo "========================================"
