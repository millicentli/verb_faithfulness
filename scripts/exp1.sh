#!/bin/bash
# Experiment 1: zero-shot baseline, Patchscopes, and LIT on feature extraction

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
LIT_CKPT="${OUTPUT_DIR}/act-llama3_decoder-llama3/000/checkpoints/epoch0-steps8105-2025-11-17_01-44-56" # Change this to yours


TARGET_MODEL="meta-llama/Llama-3.1-8B-Instruct"
DECODER_MODEL="meta-llama/Llama-3.1-8B-Instruct"
DATA_DIR="${BASE_DIR}/datasets/feature_extraction"

ZEROSHOT_OUTPUT="results/baseline/zeroshot_llama3"
VERB_PATH="results/verbalization"

PATCHSCOPES_SRC_LAYERS=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15)
MIN_TGT_LAYER=1
MAX_TGT_LAYER=32
LIT_LAYERS=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15)

# ── Zero-shot ─────────────────────────────────────────────────────────────────

echo "========================================"
echo "Zero-shot baseline"
echo "========================================"

for task_name in "${TASK_NAMES[@]}"; do
    echo "  task: ${task_name}"
    python -m src.predict_zeroshot \
        --model_name "${TARGET_MODEL}" \
        --dataset_name "feature_extraction" \
        --task_name "${task_name}.tsv" \
        --output_dir "${ZEROSHOT_OUTPUT}" \
        --data_dir "${DATA_DIR}"
done

echo "Zero-shot done. Results: ${ZEROSHOT_OUTPUT}"
echo ""

# ── Patchscopes ───────────────────────────────────────────────────────────────

echo "========================================"
echo "Patchscopes"
echo "========================================"

for src_layer in "${PATCHSCOPES_SRC_LAYERS[@]}"; do
    for task_name in "${TASK_NAMES[@]}"; do
        output_dir="${VERB_PATH}/patchscopes_llama3/src_${src_layer}"
        echo "  src=${src_layer}, tgt=${MIN_TGT_LAYER}-$((MAX_TGT_LAYER - 1)), task=${task_name}"
        python -m src.predict_verbalizer \
            --target_model_name "${TARGET_MODEL}" \
            --decoder_model_name "${DECODER_MODEL}" \
            --output_dir "${output_dir}" \
            --method "patchscopes" \
            --dataset_name "feature_extraction" \
            --task_name "${task_name}.tsv" \
            --data_dir "${DATA_DIR}" \
            --src_layer "${src_layer}" \
            --min_tgt_layer "${MIN_TGT_LAYER}" \
            --max_tgt_layer "${MAX_TGT_LAYER}"
    done
done

echo "Patchscopes done. Results: ${VERB_PATH}/patchscopes_llama3/"
echo ""

# ── LIT ───────────────────────────────────────────────────────────────────────

echo "========================================"
echo "LIT"
echo "========================================"

for layer in "${LIT_LAYERS[@]}"; do
    output_dir="${VERB_PATH}/lit_llama3/${layer}"
    echo "  layer: ${layer}"
    python -m src.predict_verbalizer \
        --target_model_name "${TARGET_MODEL}" \
        --decoder_model_name "${DECODER_MODEL}" \
        --decoder_model_path "${LIT_CKPT}" \
        --output_dir "${output_dir}" \
        --method "lit" \
        --dataset_name "feature_extraction" \
        --data_dir "${DATA_DIR}" \
        --min_layer_to_read "${layer}" \
        --max_layer_to_read "$((layer + 1))" \
        --process_all_tasks
done

echo "LIT done. Results: ${VERB_PATH}/lit_llama3/"
echo ""
echo "========================================"
echo "Experiment 1 complete."
echo "========================================"
