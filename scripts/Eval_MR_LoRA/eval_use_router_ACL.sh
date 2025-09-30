#!/bin/bash
gpu_list="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" # 
# gpu_list="2,3"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

if [ ! -n "$1" ] ;then
    DATASET="OCR_test" # APP_test OCR_test Math_test VP_test
else
    DATASET=$1
fi

if [ ! -n "$2" ] ;then # 结果路径的后缀，M_N中的N
    QF="${DATASET%%_*}" # OCR APP Math VP
else
    QF="${DATASET%%_*}"
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

if [ $QF == "Math" ]; then
    gpu_list="${CUDA_VISIBLE_DEVICES:-0}" # 
    # gpu_list="2,3"
    IFS=',' read -ra GPULIST <<< "$gpu_list"

    CHUNKS=${#GPULIST[@]}
fi

RESULT_DIR="results/ACL/RouterFinal_$STAGE/$DATASET/$(basename $MODELPATH)" # 我们方法结果保存路径
ALL_RESULT_DIR="results/ACL/model_dataset" # 所有的交叉推理结果
DATA_PATH=/data/hongbo_zhao/Ability_data
IMAGE_FOLDER=$DATA_PATH/$DATASET

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_agent_select_lora_ACL \
        --model-path $MODELPATH \
        --model-base /home/hongbo_zhao/ghy-cl-codebase/llava-v1.5-7b \
        --question-file $DATA_PATH/$DATASET/test.json \
        --image-folder $IMAGE_FOLDER  \
        --result-folders $ALL_RESULT_DIR \
        --answers-file $RESULT_DIR/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --qf $QF \
        --conv-mode vicuna_v1 &
done
wait
 
output_file=$RESULT_DIR/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $RESULT_DIR/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done
wait

# calculate the choose ACC
python -m llava.eval.model_agent_select_acc \
    --qf $QF --answers-file $output_file
echo ""
echo $IMAGE_FOLDER
echo $output_file


if [ $QF == "Math" ]; then
    python -m llava.eval.CoIN.eval_math \
    --result-file $output_file

elif [ $QF == "OCR" ]; then
    python -m llava.eval.CoIN.eval_ocr \
    --annotation-file $DATA_PATH/$DATASET/test.json \
    --result-file $output_file \
    --output-dir $RESULT_DIR 

elif [ $QF == "VP" ]; then
    python -m llava.eval.CoIN.eval_sat \
        --annotation-file $DATA_PATH/$DATASET/test.json \
        --result-file $output_file \
        --output-dir $RESULT_DIR 

elif [ $QF == "APP" ]; then
    python -m llava.eval.CoIN.convert_result_to_submission \
        --result-file $output_file --output_file $RESULT_DIR/our_result_for_submission.tsv
fi