#!/bin/bash
# Targeted AutoCal follow-up:
# gate only postprocess unknown-vs-known competition for the same patch.

set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-run_outputs/paper/owlvit-large-patch14/t1}"
MODEL_NAME="${MODEL_NAME:-google/owlvit-large-patch14}"
IMAGE_SIZE="${IMAGE_SIZE:-840}"
BATCH_SIZE="${BATCH_SIZE:-5}"
RESULT="${RESULT:-post_gate_game_t015_m11_20260602.csv}"
TCP="${TCP:-28962}"
POST_GATE_KNOWN_THRESHOLD="${POST_GATE_KNOWN_THRESHOLD:-0.15}"
POST_GATE_UNKNOWN_MARGIN="${POST_GATE_UNKNOWN_MARGIN:-1.1}"
POST_GATE_FLOOR="${POST_GATE_FLOOR:-0.05}"
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
  --use_postprocess_known_gate \
  --post_gate_known_threshold "${POST_GATE_KNOWN_THRESHOLD}" \
  --post_gate_unknown_margin "${POST_GATE_UNKNOWN_MARGIN}" \
  --post_gate_floor "${POST_GATE_FLOOR}"

echo ""
echo "===== POSTPROCESS KNOWN GATE GAME COMPLETE ====="
echo "${OUTPUT_DIR}/${RESULT}"
