#!/bin/bash
# Experiment 3: zero-shot, Patchscopes, and LIT on PersonaQA

set -e

TASK_NAMES=(
    "country"
    "favorite_food"
    "favorite_drink"
    "favorite_music_genre"
    "favorite_sport"
    "favorite_boardgame"
)

BASE_DIR="/home/li.mil/verb_faithfulness" # Change this
OUTPUT_DIR="/scratch/li.mil/verb_faithfulness" # Change this
LIT_CKPT="${OUTPUT_DIR}/act-llama3_decoder-llama3/000/checkpoints/epoch0-steps8105-2025-11-17_01-44-56" # Change this to yours

MODEL_DIR="${OUTPUT_DIR}/personaqa"
declare -A TARGET_MODELS=(
    ["PersonaQA"]="${MODEL_DIR}/Llama-3-8B-PersonaQA"
    ["PersonaQA-Fantasy"]="${MODEL_DIR}/Llama-3-8B-PersonaQA-Fantasy"
    ["PersonaQA-Shuffled"]="${MODEL_DIR}/Llama-3-8B-PersonaQA-Shuffled"
)

DECODER_MODEL="meta-llama/Llama-3.1-8B-Instruct"

PATCHSCOPES_SRC_LAYERS=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15)
MIN_TGT_LAYER=1
MAX_TGT_LAYER=32
LIT_LAYERS=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15)

for variant in "${!TARGET_MODELS[@]}"; do
    TARGET_MODEL="${TARGET_MODELS[$variant]}"
    DATA_DIR="${BASE_DIR}/datasets/personaqa/${variant}"
    PQA_PATH="results/personaqa/${variant}"

    echo "###################################################"
    echo "# Variant: ${variant}"
    echo "# Model:   ${TARGET_MODEL}"
    echo "###################################################"
    echo ""

    # ── Zero-shot ─────────────────────────────────────────────────────────────

    echo "========================================"
    echo "Zero-shot baseline"
    echo "========================================"

    for task_name in "${TASK_NAMES[@]}"; do
        echo "  task: ${task_name}"
        python -m src.predict_zeroshot \
            --model_name "${DECODER_MODEL}" \
            --dataset_name "personaqa" \
            --task_name "${task_name}" \
            --output_dir "${PQA_PATH}/zeroshot" \
            --data_dir "${DATA_DIR}/${task_name}.json"
    done

    echo "Zero-shot done. Results: ${PQA_PATH}/zeroshot/"
    echo ""

    # ── Patchscopes ───────────────────────────────────────────────────────────

    echo "========================================"
    echo "Patchscopes"
    echo "========================================"

    for src_layer in "${PATCHSCOPES_SRC_LAYERS[@]}"; do
        for task_name in "${TASK_NAMES[@]}"; do
            output_dir="${PQA_PATH}/patchscopes/src_${src_layer}"
            echo "  src=${src_layer}, tgt=${MIN_TGT_LAYER}-$((MAX_TGT_LAYER - 1)), task=${task_name}"
            python -m src.predict_verbalizer \
                --target_model_name "${TARGET_MODEL}" \
                --decoder_model_name "${DECODER_MODEL}" \
                --output_dir "${output_dir}" \
                --method "patchscopes" \
                --dataset_name "personaqa" \
                --task_name "${task_name}" \
                --data_dir "${DATA_DIR}" \
                --src_layer "${src_layer}" \
                --min_tgt_layer "${MIN_TGT_LAYER}" \
                --max_tgt_layer "${MAX_TGT_LAYER}"
        done
    done

    echo "Patchscopes done. Results: ${PQA_PATH}/patchscopes/"
    echo ""

    # ── LIT ───────────────────────────────────────────────────────────────────

    echo "========================================"
    echo "LIT"
    echo "========================================"

    for layer in "${LIT_LAYERS[@]}"; do
        for task_name in "${TASK_NAMES[@]}"; do
            output_dir="${PQA_PATH}/lit/${layer}"
            echo "  layer: ${layer}, task: ${task_name}"
            python -m src.predict_verbalizer \
                --target_model_name "${TARGET_MODEL}" \
                --decoder_model_name "${DECODER_MODEL}" \
                --decoder_model_path "${LIT_CKPT}" \
                --output_dir "${output_dir}" \
                --method "lit" \
                --dataset_name "personaqa" \
                --task_name "${task_name}" \
                --data_dir "${DATA_DIR}" \
                --min_layer_to_read "${layer}" \
                --max_layer_to_read "$((layer + 1))"
        done
    done

    echo "LIT done. Results: ${PQA_PATH}/lit/"
    echo ""

done

echo "========================================"
echo "Experiment 3 complete."
echo "========================================"
