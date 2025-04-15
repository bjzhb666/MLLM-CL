#!/bin/bash
gpu_list="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" # 
# gpu_list="2,3"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

if [ ! -n "$1" ] ;then
    DATASET="OCR_test" # APP_test OCR_test math_test CV-Benchv2
else
    DATASET=$1
fi

if [ ! -n "$2" ] ;then
    QF="ocr" # ocr app math count
else
    QF=$2
fi

if [ ! -n "$3" ] ;then
    STAGE='Finetune' # no need to change
else
    STAGE=$3
fi

if [ ! -n "$4" ] ;then
    MODELPATH='checkpoints/Router_Ability/Router_llava_lora_5e-6-ep30'
else
    MODELPATH=$4
fi

RESULT_DIR="results/Ability/Router/$DATASET"
ALL_RESULT_DIR="results/Ability/LLaVA" 
DATA_PATH=/data/hongbo_zhao/Ability_data
IMAGE_FOLDER=$DATA_PATH/$DATASET

if [ $QF == "math" ]; then
    for IDX in $(seq 0 $((CHUNKS-1))); do
        CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m ETrain.Eval.LLaVA.CoIN.model_agents_select_lora_ability \
            --model-path $MODELPATH \
            --question-file $DATA_PATH/$DATASET/test.json \
            --image-folder $IMAGE_FOLDER  \
            --result-folders $ALL_RESULT_DIR \
            --answers-file $RESULT_DIR/$STAGE/${CHUNKS}_${IDX}.jsonl \
            --num-chunks $CHUNKS \
            --chunk-idx $IDX \
            --temperature 0 \
            --qf $QF \
            --conv-mode vicuna_v1 \
            --output_xlsx $RESULT_DIR/$STAGE/result.xlsx &
    done
    wait
else
    for IDX in $(seq 0 $((CHUNKS-1))); do
        CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m ETrain.Eval.LLaVA.CoIN.model_agents_select_lora_ability \
            --model-path $MODELPATH \
            --question-file $DATA_PATH/$DATASET/test.json \
            --image-folder $IMAGE_FOLDER  \
            --result-folders $ALL_RESULT_DIR \
            --answers-file $RESULT_DIR/$STAGE/${CHUNKS}_${IDX}.jsonl \
            --num-chunks $CHUNKS \
            --chunk-idx $IDX \
            --temperature 0 \
            --qf $QF \
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
wait

# calculate the choose ACC
python -m ETrain.Eval.LLaVA.CoIN.model_agents_select_acc \
    --qf $QF --answers-file $output_file
echo ""
echo $IMAGE_FOLDER
echo $output_file

if [ $QF == "math" ]; then
    python -m ETrain.Eval.LLaVA.CoIN.eval_math \
    --result-file $RESULT_DIR/$STAGE/result.xlsx

elif [ $QF == "ocr" ]; then
    python -m ETrain.Eval.LLaVA.CoIN.eval_ocr \
    --annotation-file $DATA_PATH/OCR_test/test.json \
    --result-file $output_file \
    --output-dir $RESULT_DIR/$STAGE 

elif [ $QF == "count" ]; then
    python -m ETrain.Eval.LLaVA.CoIN.eval_sat \
        --annotation-file $DATA_PATH/CV-Benchv2/test.json \
        --result-file $output_file \
        --output-dir $RESULT_DIR/$STAGE 

elif [ $QF == "app" ]; then
    python -m ETrain.Eval.LLaVA.CoIN.convert_result_to_submission \
        --result-file $output_file --output_file $RESULT_DIR/$STAGE/our_result_for_submission.tsv
fi