#!/bin/bash
gpu_list="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" # 
# gpu_list="2,3"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

if [ ! -n "$1" ] ;then
    DATASET="RemoteSensing" # Medical/data AD/DriveLM Sci Fin
else
    DATASET=$1
fi

if [ ! -n "$2" ] ;then
    QF="rs" # rs drivelm pathvqa fin sci
else
    QF=$2
fi

if [ ! -n "$3" ] ;then
    STAGE='Finetune' # no need to change
else
    STAGE=$3
fi

if [ ! -n "$4" ] ;then
    MODELPATH='checkpoints/Router/Router_llava_lora_5e-6-ep40'
else
    MODELPATH=$4
fi

# if [ ! -n "$5" ] ;then
#     RANK=""
# else
#     RANK=$5
# fi

RESULT_DIR="results/Domain/RouterACC/$DATASET"
# ALL_RESULT_DIR="results/CoIN/RANK$RANK" 
ALL_RESULT_DIR="results/CoIN/LLaVA"
DATA_PATH=/data/hongbo_zhao/data/Domain_data


if [ "$2" != "sci" ]; then
    IMAGE_FOLDER=$DATA_PATH/$DATASET
else
    IMAGE_FOLDER=$DATA_PATH
fi


for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m ETrain.Eval.LLaVA.CoIN.model_agents_select_lora \
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

if [ $QF == "rs" ]; then
    python -m ETrain.Eval.LLaVA.CoIN.eval_ai2d \
        --annotation-file $DATA_PATH/RemoteSensing/test.json \
        --result-file $output_file \
        --output-dir $RESULT_DIR/$STAGE 

elif [ $QF == "drivelm" ]; then
    python -m ETrain.Eval.LLaVA.CoIN.eval_ai2d \
        --annotation-file $DATA_PATH/AD/DriveLM/test.json \
        --result-file $output_file \
        --output-dir $RESULT_DIR/$STAGE 

elif [ $QF == "pathvqa" ]; then
    python -m ETrain.Eval.LLaVA.CoIN.eval_pvqa \
        --annotation-file $DATA_PATH/Medical/data/test.json \
        --result-file $output_file \
        --output-dir $RESULT_DIR/$STAGE 

elif [ $QF == "fin" ]; then
   python -m ETrain.Eval.LLaVA.CoIN.eval_finvis \
        --annotation-file $DATA_PATH/Fin/test.json \
        --result-file $output_file \
        --output-dir $RESULT_DIR/$STAGE 

elif [ $QF == "sci" ]; then
    python -m ETrain.Eval.LLaVA.CoIN.eval_sci \
        --annotation-file $DATA_PATH/Sci/test.json \
        --result-file $output_file \
        --output-dir $RESULT_DIR/$STAGE 
fi