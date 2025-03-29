#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

if [ ! -n "$1" ] ;then
    STAGE='Finetune'
else
    STAGE=$1
fi

if [ ! -n "$2" ] ;then
    MODELPATH='./checkpoints/LLaVA/CoIN/Sci_llava_lora'
else
    MODELPATH=$2
fi
if [ ! -n "$3" ] ;then
    MODELNAME='Sci_sci'
else
    MODELNAME=$3
fi

if [ ! -n "$4" ] ;then
    MODELBASE='LLaVA'
else
    MODELBASE=$4
fi

RESULT_DIR="./results/CoIN/$MODELBASE/$MODELNAME"
DATA_PATH=/data/hongbo_zhao/data/Domain_data
if [ "$3" = "llava_sci" ]; then
    echo "llava test"
    for IDX in $(seq 0 $((CHUNKS-1))); do
        CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m ETrain.Eval.LLaVA.CoIN.model_ai2d \
            --model-path $MODELPATH \
            --question-file $DATA_PATH/Sci/test.json \
            --image-folder $DATA_PATH \
            --answers-file $RESULT_DIR/$STAGE/${CHUNKS}_${IDX}.jsonl \
            --num-chunks $CHUNKS \
            --chunk-idx $IDX \
            --temperature 0 \
            --conv-mode vicuna_v1 &
    done

    wait
else
    for IDX in $(seq 0 $((CHUNKS-1))); do
        CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m ETrain.Eval.LLaVA.CoIN.model_ai2d \
            --model-path $MODELPATH \
            --model-base ./checkpoints/LLaVA/Vicuna/vicuna-7b-v1.5 \
            --question-file $DATA_PATH/Sci/test.json \
            --image-folder $DATA_PATH \
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

python -m ETrain.Eval.LLaVA.CoIN.eval_sci \
    --annotation-file $DATA_PATH/Sci/test.json \
    --result-file $output_file \
    --output-dir $RESULT_DIR/$STAGE \
