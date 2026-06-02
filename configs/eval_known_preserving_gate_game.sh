#!/bin/bash
# Targeted follow-up experiment for AutoCal:
# evaluate whether a known-preserving gate recovers Game K_AP50 while keeping
# the AutoCal unknown AP50 gain.

set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-run_outputs/paper/owlvit-large-patch14/t1}"
MODEL_NAME="${MODEL_NAME:-google/owlvit-large-patch14}"
IMAGE_SIZE="${IMAGE_SIZE:-840}"
BATCH_SIZE="${BATCH_SIZE:-5}"
RESULT="${RESULT:-known_gate_game_sigmoid_t025_20260602.csv}"
TCP="${TCP:-28952}"
KNOWN_GATE_THRESHOLD="${KNOWN_GATE_THRESHOLD:-0.25}"
KNOWN_GATE_GAMMA="${KNOWN_GATE_GAMMA:-1.0}"
KNOWN_GATE_FLOOR="${KNOWN_GATE_FLOOR:-0.10}"
KNOWN_GATE_SOURCE="${KNOWN_GATE_SOURCE:-sigmoid}"
CKPT="${CKPT:-${OUTPUT_DIR}/Game_bast_28.709915161132812.pth}"

if [[ ! -f "${CKPT}" ]]; then
  echo "Missing checkpoint: ${CKPT}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"
rm -f "${OUTPUT_DIR}/${RESULT}"

python main.py \
  --model_name "${MODEL_NAME}" \
  --num_few_shot 100 \
  --batch_size "${BATCH_SIZE}" \
  --PREV_INTRODUCED_CLS 0 \
  --CUR_INTRODUCED_CLS 30 \
  --TCP "${TCP}" \
  --dataset Game \
  --unk_method sigmoid-max-mcm \
  --image_conditioned \
  --image_resize "${IMAGE_SIZE}" \
  --att_refinement --att_adapt --att_selection --use_attributes \
  --output_dir "${OUTPUT_DIR}" \
  --output_file "${RESULT}" \
  --eval_model "${CKPT}" \
  --use_auto_unknown_calibration \
  --use_known_preserving_gate \
  --known_gate_threshold "${KNOWN_GATE_THRESHOLD}" \
  --known_gate_gamma "${KNOWN_GATE_GAMMA}" \
  --known_gate_floor "${KNOWN_GATE_FLOOR}" \
  --known_gate_source "${KNOWN_GATE_SOURCE}"

echo ""
echo "===== KNOWN-PRESERVING GATE GAME COMPLETE ====="
echo "${OUTPUT_DIR}/${RESULT}"
