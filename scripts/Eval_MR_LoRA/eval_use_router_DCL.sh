#!/bin/bash
gpu_list="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" # 
# gpu_list="2,3"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

if [ ! -n "$1" ] ;then
    DATASET="RS" # RS Med AD Sci Fin
else
    DATASET=$1
fi

if [ ! -n "$2" ] ;then
    QF=$DATASET # RS Med AD Sci Fin 
else
    QF=$DATASET
fi

if [ ! -n "$3" ] ;then
    STAGE='Finetune' # no need to change
else
    STAGE=$3
fi

if [ ! -n "$4" ] ;then
    MODELPATH='/home/hongbo_zhao/ghy-cl-codebase/MCITlib/checkpoints/MLLM-DCL/LLaVA-1.5/MR-LoRA-router/Task5_llava_lora'
else
    MODELPATH=$4
fi

# if [ ! -n "$5" ] ;then
#     RANK=""
# else
#     RANK=$5
# fi

RESULT_DIR="results/DCL/RouterFinal_$STAGE/$DATASET/$(basename $MODELPATH)" # 我们方法结果保存路径
# ALL_RESULT_DIR="results/CoIN/RANK$RANK" 
ALL_RESULT_DIR="results/DCL/model_dataset" # 25个结果路径
DATA_PATH=/data/hongbo_zhao/data/Domain_data

IMAGE_FOLDER=$DATA_PATH/$DATASET

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_agent_select_lora_DCL \
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



if [ $QF == "RS" ]; then
    python -m llava.eval.CoIN.eval_ai2d \
        --annotation-file $DATA_PATH/$DATASET/test.json \
        --result-file $output_file \
        --output-dir $RESULT_DIR 

elif [ $QF == "AD" ]; then
    python -m llava.eval.CoIN.eval_ai2d \
        --annotation-file $DATA_PATH/$DATASET/test.json \
        --result-file $output_file \
        --output-dir $RESULT_DIR 

elif [ $QF == "Med" ]; then
    python -m llava.eval.CoIN.eval_pvqa \
        --annotation-file $DATA_PATH/$DATASET/test.json \
        --result-file $output_file \
        --output-dir $RESULT_DIR 

elif [ $QF == "Fin" ]; then
   python -m llava.eval.CoIN.eval_finvis \
        --annotation-file $DATA_PATH/$DATASET/test.json \
        --result-file $output_file \
        --output-dir $RESULT_DIR 

elif [ $QF == "Sci" ]; then
    python -m llava.eval.CoIN.eval_sci \
        --annotation-file $DATA_PATH/$DATASET/test.json \
        --result-file $output_file \
        --output-dir $RESULT_DIR 
fi