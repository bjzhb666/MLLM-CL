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
    MODELPATH='./checkpoints/Instruction/Only_Pretrain_1.5/OCRVQA/llava-1.5-7b-lora'
else
    MODELPATH=$2
fi
if [ ! -n "$3" ] ;then
    MODELNAME='SQA_SQA'
else
    MODELNAME=$3
fi

if [ ! -n "$4" ] ;then
    MODELBASE='LLaVA'
else
    MODELBASE=$4
fi

RESULT_DIR="./results/CoIN/$MODELBASE/$MODELNAME"
# RESULT_DIR="./results/CoIN/LLaVA/OCRVQA"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m ETrain.Eval.LLaVA.CoIN.model_tqa \
        --model-path $MODELPATH \
        --model-base ./checkpoints/LLaVA/Vicuna/vicuna-7b-v1.5 \
        --question-file /data2/rundongwang/data/tmp/Geo170K/test.json \
        --image-folder /data2/rundongwang/data/tmp/Geo170K \
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

python -m ETrain.Eval.LLaVA.CoIN.eval_geo \
    --annotation-file /data2/rundongwang/data/tmp/Geo170K/test.json \
    --result-file $output_file \
    --output-dir $RESULT_DIR/$STAGE \
