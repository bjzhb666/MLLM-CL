#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
gpu_list="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
# gpu_list="2,3"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

if [ ! -n "$1" ] ;then
    STAGE='Finetune'
else
    STAGE=$1
fi

if [ ! -n "$2" ] ;then
    MODELPATH='./checkpoints/LLaVA/Agent/ScienceQA_llava_lora' # model path
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
SOURCE=Instructions_Original
# SOURCE=Agent

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m ETrain.Eval.LLaVA.CoIN.model_vqa_science \
        --model-path $MODELPATH \
        --model-base ./checkpoints/LLaVA/Vicuna/vicuna-7b-v1.5 \
        --question-file ./playground/$SOURCE/ScienceQA/test.json \
        --image-folder ../DatasetCoIN \
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

python -m ETrain.Eval.LLaVA.CoIN.eval_science_qa \
    --base-dir ../DatasetCoIN/ScienceQA \
    --result-file $output_file \
    --output-file $RESULT_DIR/$STAGE/output.jsonl \
    --output-result $RESULT_DIR/$STAGE/output_result.jsonl \
