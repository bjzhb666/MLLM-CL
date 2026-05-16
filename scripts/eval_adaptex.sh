#!/bin/bash
# AdapteX: Evaluate using router-based expert selection
#
# This script runs the two-stage inference:
#   1. Router selects the best expert for each test sample
#   2. The selected expert's pre-computed answer is used
#
# Usage: bash adaptex/scripts/eval_adaptex.sh <SETTING> <DATASET> [ROUTER_PATH] [STATE_DIR]
#
# Examples:
#   bash adaptex/scripts/eval_adaptex.sh DCL RS
#   bash adaptex/scripts/eval_adaptex.sh ACL OCR_test

SETTING=${1:-"DCL"}
DATASET=${2:-"RS"}

if [ "$SETTING" == "DCL" ]; then
    QF="${DATASET}"
    DEFAULT_ROUTER="checkpoints/AdapteX-DCL-LLaVA/router_final"
    DEFAULT_STATE="checkpoints/AdapteX-DCL-LLaVA/adaptex_state"
    DEFAULT_DATA="/path/to/Domain_data"
    DEFAULT_RESULTS="results/DCL/model_dataset"
else
    QF="${DATASET%%_*}"
    DEFAULT_ROUTER="checkpoints/AdapteX-ACL-LLaVA/router_final"
    DEFAULT_STATE="checkpoints/AdapteX-ACL-LLaVA/adaptex_state"
    DEFAULT_DATA="/path/to/Ability_data"
    DEFAULT_RESULTS="results/ACL/model_dataset"
fi

ROUTER_PATH=${3:-$DEFAULT_ROUTER}
STATE_DIR=${4:-$DEFAULT_STATE}
MODEL_BASE=${MODEL_BASE:-"/path/to/llava-v1.5-7b"}
DATA_PATH=${DATA_PATH:-$DEFAULT_DATA}
RESULT_FOLDERS=${RESULT_FOLDERS:-$DEFAULT_RESULTS}

gpu_list="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}

RESULT_DIR="results/AdapteX-${SETTING}/${DATASET}"
IMAGE_FOLDER="$DATA_PATH/$DATASET"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m adaptex.eval_adaptex \
        --setting $SETTING \
        --prompt-manager-path $STATE_DIR/prompt_manager.json \
        --router-model-path $ROUTER_PATH \
        --model-base $MODEL_BASE \
        --question-file $DATA_PATH/$DATASET/test.json \
        --image-folder $IMAGE_FOLDER \
        --result-folders $RESULT_FOLDERS \
        --answers-file $RESULT_DIR/${CHUNKS}_${IDX}.jsonl \
        --qf $QF \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX &
done

wait

output_file=$RESULT_DIR/merge.jsonl
> "$output_file"

for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $RESULT_DIR/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done
wait

echo "Results saved to: $output_file"

python -m llava.eval.model_agent_select_acc \
    --qf $QF --answers-file $output_file
