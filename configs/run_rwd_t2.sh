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
current_date=$(date +'%Y-%m-%d_%H-%M-%S')

# Declare arrays for different configurations
declare -a DATASETS=("Aquatic" "Aerial" "Surgical" "Medical" "Game")
declare -a MODELS=("google/owlvit-large-patch14")
declare -a NUM_SHOTS=(100)


# Declare associative array for CUR_INTRODUCED_CLS per dataset
declare -A PREV_INTRODUCED_CLS
PREV_INTRODUCED_CLS["Aerial"]=10
PREV_INTRODUCED_CLS["Surgical"]=6
PREV_INTRODUCED_CLS["Medical"]=6
PREV_INTRODUCED_CLS["Aquatic"]=4
PREV_INTRODUCED_CLS["Game"]=30


declare -A CUR_INTRODUCED_CLS
CUR_INTRODUCED_CLS["Aerial"]=10
CUR_INTRODUCED_CLS["Surgical"]=7
CUR_INTRODUCED_CLS["Medical"]=6
CUR_INTRODUCED_CLS["Aquatic"]=3
CUR_INTRODUCED_CLS["Game"]=29

declare -A BATCH_SIZEs
BATCH_SIZEs["google/owlvit-base-patch16"]=10
BATCH_SIZEs["google/owlvit-large-patch14"]=2

declare -A IMAGE_SIZEs
IMAGE_SIZEs["google/owlvit-base-patch16"]=768
IMAGE_SIZEs["google/owlvit-large-patch14"]=840

# Loop through each configuration
for num_shot in "${NUM_SHOTS[@]}"; do
for dataset in "${DATASETS[@]}"; do
  cur_cls=${CUR_INTRODUCED_CLS[$dataset]}
  prev_cls=${PREV_INTRODUCED_CLS[$dataset]}

  for model in "${MODELS[@]}"; do
    BATCH_SIZE=${BATCH_SIZEs[$model]}
    IMAGE_SIZE=${IMAGE_SIZEs[$model]}
    tcp=$((TCP_INIT + counter))
    # Construct the command to run
    python main.py --model_name $model --num_few_shot $num_shot --batch_size $BATCH_SIZE \
    --PREV_INTRODUCED_CLS $prev_cls --CUR_INTRODUCED_CLS $cur_cls --TCP $tcp --dataset $dataset \
    --image_conditioned --image_resize $IMAGE_SIZE --classnames_file 'classnames.txt'\
    --att_refinement --att_adapt --att_selection --use_attributes

    counter=$((counter + 1))
  done
done
done
