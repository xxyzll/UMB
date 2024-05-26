#!/bin/bash

set -e
cur_fname="$(basename $0 .sh)"
script_name=$(basename $0)

# Cluster parameters
partition=""
account=""

# Initialize TCP port and counter
TCP_INIT=29500
counter=0

# Declare arrays for different configurations
declare -a DATASETS=("Aerial" "Surgical" "Medical" "Aquatic" "Game")
declare -a MODELS=("google/owlvit-base-patch16" "google/owlvit-large-patch14")
declare -a NUM_SHOTS=(1 10 100)


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

# Loop through each configuration
for num_shot in "${NUM_SHOTS[@]}"; do
  for dataset in "${DATASETS[@]}"; do
    cur_cls=${CUR_INTRODUCED_CLS[$dataset]}
    for model in "${MODELS[@]}"; do
      BATCH_SIZE=${BATCH_SIZEs[$model]}
      IMAGE_SIZE=${IMAGE_SIZEs[$model]}
      tcp=$((TCP_INIT + counter))

      # Construct the command to run --output_file res/$dataset-$num_shot-$model.csv \
      cmd="python main.py --model_name \"$model\" --num_few_shot $num_shot --batch_size $BATCH_SIZE \
      --PREV_INTRODUCED_CLS 0 --CUR_INTRODUCED_CLS $cur_cls --TCP $tcp --dataset $dataset \
      --image_conditioned --image_resize $IMAGE_SIZE \
      --att_adapt --unk_methods 'None' --unk_method 'None'"

      echo "Constructed Command:\n$cmd"

      # Uncomment below to submit the job
      sbatch <<< \
"#!/bin/bash
#SBATCH --job-name=${dataset}-${cur_fname}-${partition}-${num_shot}
#SBATCH --output=slurm_logs/${dataset}/v0-${current_date}-${num_shot}-${cur_fname}-${partition}-%j-out.txt
#SBATCH --error=slurm_logs/${dataset}/v0-${current_date}-${num_shot}-${cur_fname}-${partition}-%j-err.txt
#SBATCH --mem=32gb
#SBATCH -c 2
#SBATCH --gres=gpu:a6000
#SBATCH -p $partition
#SBATCH -A $account
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
echo \"$cmd\"
# Uncomment below to actually run the command
eval \"$cmd\"
"

      counter=$((counter + 1))
    done
  done
done
