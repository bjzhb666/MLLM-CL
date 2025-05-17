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
    MODELPATH='./checkpoints/LLaVA/Ability/Count_llava_lora_visual-ep1_1e-5_ty_cl_sft'
else
    MODELPATH=$2
fi
if [ ! -n "$3" ] ;then
    MODELNAME='Count_count'
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

if [ "$3" = "llava_count" ]; then
    echo "LLaVA 1.5 test"
    for IDX in $(seq 0 $((CHUNKS-1))); do
        CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m ETrain.Eval.LLaVA.CoIN.model_sat \
            --model-path $MODELPATH \
            --question-file $DATA_PATH/CV-Benchv2/test.json \
            --image-folder $DATA_PATH/CV-Benchv2 \
            --answers-file $RESULT_DIR/$STAGE/${CHUNKS}_${IDX}.jsonl \
            --num-chunks $CHUNKS \
            --chunk-idx $IDX \
            --temperature 0 \
            --conv-mode vicuna_v1 &
    done
    wait
else
    for IDX in $(seq 0 $((CHUNKS-1))); do
        CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m ETrain.Eval.LLaVA.CoIN.model_sat \
            --model-path $MODELPATH \
            --model-base ./checkpoints/LLaVA/Vicuna/vicuna-7b-v1.5 \
            --question-file $DATA_PATH/CV-Benchv2/test.json \
            --image-folder $DATA_PATH/CV-Benchv2 \
            --answers-file $RESULT_DIR/$STAGE/${CHUNKS}_${IDX}.jsonl \
            --num-chunks $CHUNKS \
            --chunk-idx $IDX \
            --temperature 0 \
            --conv-mode vicuna_v1 &
    done
    wait
fi

output_file=$RESULT_DIR/$STAGE/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $RESULT_DIR/$STAGE/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

if [ "$3" = "llava_count" ]; then
    python -m ETrain.Eval.LLaVA.CoIN.eval_sat \
        --annotation-file $DATA_PATH/CV-Benchv2/test.json \
        --result-file $output_file \
        --output-dir $RESULT_DIR/$STAGE 
else
    python -m ETrain.Eval.LLaVA.CoIN.eval_sat \
        --annotation-file $DATA_PATH/CV-Benchv2/test.json \
        --result-file $output_file \
        --output-dir $RESULT_DIR/$STAGE 
fi