#!/bin/bash

set -e
cur_fname="$(basename $0 .sh)"
script_name=$(basename $0)

# Cluster parameters
partition=""
account=""

# Initialize TCP port and counter
TCP_INIT=28500
counter=0
export CUDA_VISIBLE_DEVICES=1
# Declare arrays for different configurations
declare -a DATASETS=("Aquatic" "Aerial" "Game" "Medical" "Surgical")
# declare -a DATASETS=("Aquatic" "Game" "Medical")
# declare -a MODELS=("google/owlvit-base-patch16" "google/owlvit-large-patch14")
declare -a MODELS=("google/owlvit-large-patch14")
declare -a NUM_SHOTS=(100)

# Declare associative array for CUR_INTRODUCED_CLS per dataset
declare -A CUR_INTRODUCED_CLS
CUR_INTRODUCED_CLS["Aerial"]=10
CUR_INTRODUCED_CLS["Surgical"]=6
CUR_INTRODUCED_CLS["Medical"]=6
CUR_INTRODUCED_CLS["Aquatic"]=4
CUR_INTRODUCED_CLS["Game"]=30

declare -A BATCH_SIZEs
BATCH_SIZEs["google/owlvit-base-patch16"]=10
BATCH_SIZEs["google/owlvit-large-patch14"]=5

declare -A IMAGE_SIZEs
IMAGE_SIZEs["google/owlvit-base-patch16"]=768
IMAGE_SIZEs["google/owlvit-large-patch14"]=840

PRETRAINED_MODELS["Aerial"] 

# Loop through each configuration
for num_shot in "${NUM_SHOTS[@]}"; do
  for dataset in "${DATASETS[@]}"; do
    cur_cls=${CUR_INTRODUCED_CLS[$dataset]}
    for model in "${MODELS[@]}"; do
      BATCH_SIZE=${BATCH_SIZEs[$model]}
      IMAGE_SIZE=${IMAGE_SIZEs[$model]}
      tcp=$((TCP_INIT + counter))
      python main.py --model_name $model --num_few_shot $num_shot --batch_size $BATCH_SIZE \
      --PREV_INTRODUCED_CLS 0 --CUR_INTRODUCED_CLS $cur_cls --TCP $tcp --dataset $dataset --unk_method 'sigmoid-max-mcm'\
      --image_conditioned --image_resize $IMAGE_SIZE \
      --att_refinement --att_adapt --att_selection --use_attributes \
      --output_dir 'experiments/full_repeat/our' --output_file 'shot_100.csv' --prev_output_file 'shot_100.csv'\
      "--eval_model",

      counter=$((counter + 1))
    done
  done
done
