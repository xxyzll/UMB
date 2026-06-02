#!/bin/bash
# Evaluate the bootstrap Support-Calibrated Adaptive Unknown Scoring policy.
#
# This script intentionally uses one method switch:
#   --use_auto_unknown_calibration
#
# Dataset-specific settings are selected inside main.py and written to the
# result CSV as metadata. This keeps the experiment reproducible while moving
# away from ad-hoc manual command lines.

set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-run_outputs/paper/owlvit-large-patch14/t1}"
MODEL_NAME="${MODEL_NAME:-google/owlvit-large-patch14}"
IMAGE_SIZE="${IMAGE_SIZE:-840}"
BATCH_SIZE="${BATCH_SIZE:-5}"
RESULT="${RESULT:-auto_unknown_calibration_20260601.csv}"
TCP_INIT="${TCP_INIT:-28940}"
START_DATASET="${START_DATASET:-}"
KEEP_RESULT="${KEEP_RESULT:-false}"

declare -a DATASETS=("Aquatic" "Aerial" "Game" "Medical" "Surgical")

declare -A CUR_INTRODUCED_CLS=(
  [Aquatic]=4
  [Aerial]=10
  [Game]=30
  [Medical]=6
  [Surgical]=6
)

declare -A CKPTS=(
  [Aquatic]="${OUTPUT_DIR}/Aquatic_bast.pth"
  [Aerial]="${OUTPUT_DIR}/Aerial_bast.pth"
  [Game]="${OUTPUT_DIR}/Game_bast_28.709915161132812.pth"
  [Medical]="${OUTPUT_DIR}/Medical_bast.pth"
  [Surgical]="${OUTPUT_DIR}/Surgical_bast.pth"
)

mkdir -p "${OUTPUT_DIR}"
if [[ "${KEEP_RESULT}" != "true" ]]; then
  rm -f "${OUTPUT_DIR}/${RESULT}"
fi

counter=0
started=false
if [[ -z "${START_DATASET}" ]]; then
  started=true
fi

for dataset in "${DATASETS[@]}"; do
  if [[ "${started}" != "true" ]]; then
    if [[ "${dataset}" == "${START_DATASET}" ]]; then
      started=true
    else
      counter=$((counter + 1))
      continue
    fi
  fi

  ckpt="${CKPTS[$dataset]}"
  if [[ ! -f "${ckpt}" ]]; then
    echo "Missing checkpoint for ${dataset}: ${ckpt}" >&2
    exit 1
  fi

  tcp=$((TCP_INIT + counter))
  echo ""
  echo "===== AutoCal ${dataset} | ckpt=${ckpt} | tcp=${tcp} ====="

  python main.py \
    --model_name "${MODEL_NAME}" \
    --num_few_shot 100 \
    --batch_size "${BATCH_SIZE}" \
    --PREV_INTRODUCED_CLS 0 \
    --CUR_INTRODUCED_CLS "${CUR_INTRODUCED_CLS[$dataset]}" \
    --TCP "${tcp}" \
    --dataset "${dataset}" \
    --unk_method sigmoid-max-mcm \
    --image_conditioned \
    --image_resize "${IMAGE_SIZE}" \
    --att_refinement --att_adapt --att_selection --use_attributes \
    --output_dir "${OUTPUT_DIR}" \
    --output_file "${RESULT}" \
    --prev_output_file "${RESULT}" \
    --eval_model "${ckpt}" \
    --use_auto_unknown_calibration

  counter=$((counter + 1))
done

echo ""
echo "===== AUTO UNKNOWN CALIBRATION COMPLETE ====="
echo "${OUTPUT_DIR}/${RESULT}"
