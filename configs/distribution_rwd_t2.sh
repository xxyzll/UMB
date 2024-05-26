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
# gpu
export CUDA_VISIBLE_DEVICES=0

# Declare arrays for different configurations
declare -a DATASETS=("Aquatic" "Medical" "Game" "Surgical" "Aerial" )
declare -a MODELS=("google/owlvit-base-patch16" "google/owlvit-large-patch14")

MODEL_NAME="owlvit-large-patch14"
declare -a MODELS=("google/${MODEL_NAME}")
declare -a NUM_SHOTS=(100)
# Declare associative array for CUR_INTRODUCED_CLS per dataset
declare -A CUR_INTRODUCED_CLS
CUR_INTRODUCED_CLS["Aerial"]=10
CUR_INTRODUCED_CLS["Surgical"]=7
CUR_INTRODUCED_CLS["Medical"]=6
CUR_INTRODUCED_CLS["Aquatic"]=3
CUR_INTRODUCED_CLS["Game"]=29
# Declare associative array for CUR_INTRODUCED_CLS per dataset
declare -A PREV_INTRODUCED_CLS
PREV_INTRODUCED_CLS["Aerial"]=10
PREV_INTRODUCED_CLS["Surgical"]=6
PREV_INTRODUCED_CLS["Medical"]=6
PREV_INTRODUCED_CLS["Aquatic"]=4
PREV_INTRODUCED_CLS["Game"]=30

declare -A ALPHAs
ALPHAs["Aerial"]=0.9
ALPHAs["Surgical"]=1.0
ALPHAs["Medical"]=0.4  # 0.3
ALPHAs["Aquatic"]=0.9
ALPHAs["Game"]=0.9

declare -A BALANCEs
BALANCEs["Aerial"]=0.4
BALANCEs["Surgical"]=0.4
BALANCEs["Medical"]=0.3
BALANCEs["Aquatic"]=0.1
BALANCEs["Game"]=0.3

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
      BALANCE=${BALANCEs[$dataset]}
      ALPHA=${ALPHAs[$dataset]}
      tcp=$((TCP_INIT + counter))
      python main.py --model_name $model --num_few_shot $num_shot --batch_size $BATCH_SIZE \
        --PREV_INTRODUCED_CLS $prev_cls --CUR_INTRODUCED_CLS $cur_cls --TCP $tcp --dataset $dataset \
        --image_conditioned --image_resize $IMAGE_SIZE --classnames_file 'classnames.txt'\
        --att_refinement --att_adapt --att_selection --use_attributes\
        --output_dir "experiments/full_repeat/${MODEL_NAME}/t2"\
        --eval_model "experiments/full_repeat/${MODEL_NAME}/t2/${dataset}_bast.pth" \
        --output_file "wb.csv" --prev_output_file "wb.csv"\
        --log_distribution true --balance $BALANCE --alpha $ALPHA \
        --fit_method "wb"\
        --fine_alpha True\
        # --fit_bs 2 --fit_epoch 10000 --fit_lr 0.01 
        # --fine_balance True\

      counter=$((counster + 1))
    done
  done
done
