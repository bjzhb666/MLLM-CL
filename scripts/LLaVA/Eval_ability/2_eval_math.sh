#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}" # ,1,2,3,4,5,6,7
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

if [ ! -n "$1" ] ;then
    STAGE='Finetune'
else
    STAGE=$1
fi

if [ ! -n "$2" ] ;then
    MODELPATH='./checkpoints/LLaVA/Ability/MATH_llava_lora2e-5-visual-ep1'
else
    MODELPATH=$2
fi
if [ ! -n "$3" ] ;then
    MODELNAME='Math_math'
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
if [ "$3" = "llava_math" ]; then
    echo "LLaVA 1.5 Test"
    for IDX in $(seq 0 $((CHUNKS-1))); do
        CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m ETrain.Eval.LLaVA.CoIN.model_math \
            --model-path $MODELPATH \
            --question-file $DATA_PATH/math_test/test.json \
            --image-folder $DATA_PATH/math_test \
            --answers-file $RESULT_DIR/$STAGE/${CHUNKS}_${IDX}.jsonl \
            --num-chunks $CHUNKS \
            --chunk-idx $IDX \
            --output_xlsx $RESULT_DIR/$STAGE/result.xlsx \
            --temperature 0 \
            --conv-mode vicuna_v1 &
    done
    wait
else
    for IDX in $(seq 0 $((CHUNKS-1))); do
        CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m ETrain.Eval.LLaVA.CoIN.model_math \
            --model-path $MODELPATH \
            --model-base ./checkpoints/LLaVA/Vicuna/vicuna-7b-v1.5 \
            --question-file $DATA_PATH/math_test/test.json \
            --image-folder $DATA_PATH/math_test \
            --answers-file $RESULT_DIR/$STAGE/${CHUNKS}_${IDX}.jsonl \
            --num-chunks $CHUNKS \
            --chunk-idx $IDX \
            --output_xlsx $RESULT_DIR/$STAGE/result.xlsx \
            --temperature 0 \
            --conv-mode vicuna_v1 &
    done
    wait
fi

echo "Eval math"
python -m ETrain.Eval.LLaVA.CoIN.eval_math \
    --result-file $RESULT_DIR/$STAGE/result.xlsx