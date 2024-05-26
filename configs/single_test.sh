# gpu
export CUDA_VISIBLE_DEVICES=1
TCP_INIT=28500

# Declare arrays for different configurations
# declare -a DATASETS=("Aerial" "Surgical" "Medical" "Aquatic" "Game")
declare -a DATASETS=("Surgical")
# declare -a MODELS=("google/owlvit-base-patch16" "google/owlvit-large-patch14")
declare -a MODELS=("google/owlvit-base-patch16")
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



python main.py --model_name "google/owlvit-base-patch16" --num_few_shot 100 --batch_size 10 \
      --PREV_INTRODUCED_CLS 0 --CUR_INTRODUCED_CLS 6 --TCP 28500 --dataset "Surgical" --unk_method 'cos-max-mcm'\
      --image_conditioned --image_resize 768 \
      --att_refinement --att_adapt --att_selection --use_attributes >> experiments/without_scale_cos_single.txt