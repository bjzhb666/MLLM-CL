#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" # 
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

if [ ! -n "$1" ] ;then
    STAGE='Finetune'
else
    STAGE=$1
fi

if [ ! -n "$2" ] ;then
    MODELPATH='./checkpoints/LLaVA/Ability/APP_llava_lora_visual_2e-5_ep3'
else
    MODELPATH=$2
fi
if [ ! -n "$3" ] ;then
    MODELNAME='APP_app'
else
    MODELNAME=$3
fi

if [ ! -n "$4" ] ;then
    MODELBASE='LLaVA'
else
    MODELBASE=$4
fi

RESULT_DIR="./results/Ability/$MODELBASE/$MODELNAME"
# RESULT_DIR="./results/CoIN/LLaVA/OCRVQA"
DATA_PATH=/data/hongbo_zhao/Ability_data

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m ETrain.Eval.LLaVA.CoIN.model_ai2d \
        --model-path $MODELPATH \
        --model-base ./checkpoints/LLaVA/Vicuna/vicuna-7b-v1.5 \
        --question-file $DATA_PATH/APP_test/test.json \
        --image-folder $DATA_PATH/APP_test \
        --answers-file $RESULT_DIR/$STAGE/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait

output_file=$RESULT_DIR/$STAGE/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $RESULT_DIR/$STAGE/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python -m ETrain.Eval.LLaVA.CoIN.convert_result_to_submission \
    --result-file $output_file --output_file $RESULT_DIR/$STAGE/our_result_for_submission.tsv