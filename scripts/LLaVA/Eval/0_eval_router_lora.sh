#!/bin/bash
export CUDA_LAUNCH_BLOCKING=0
gpu_list="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
# gpu_list="2,3"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

if [ ! -n "$1" ] ;then
    DATASET="ScienceQA"
else
    DATASET=$1
fi

if [ ! -n "$2" ] ;then
    QF="sqa"
else
    QF=$2
fi

if [ ! -n "$3" ] ;then
    STAGE='Finetune'
else
    STAGE=$3
fi

if [ ! -n "$4" ] ;then
    # MODELPATH='./checkpoints/LLaVA/AvgLora_1/VQAv2_llava_lora'
    MODELPATH='checkpoints/LLaVA/Vicuna/llava-7b-v1.5'
else
    MODELPATH=$4
fi

# DATASET="ScienceQA"
# QF="sqa"
RESULT_DIR="debug_folder/$DATASET"
AGENT_DIR="/data3/rundongwang/CoIN/results/CoIN/LLaVA-Cont-nf"
SPLIT="test"
if [ $QF == "textqa" ]; then
    SPLIT="val"
elif [ $QF == "vizwiz" ]; then
    SPLIT="val"
elif [ $QF == "vqav2" ]; then
    SPLIT="val"
else
    SPLIT="test"
fi

        # --model-base ./checkpoints/LLaVA/Vicuna/vicuna-7b-v1.5 \
for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m ETrain.Eval.LLaVA.CoIN.model_agents_select_lora \
        --model-path $MODELPATH \
        --question-file ./playground/Instructions_Original/$DATASET/$SPLIT.json \
        --image-folder ../DatasetCoIN \
        --result-folders $AGENT_DIR \
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

if [ $QF == "gqa" ]; then
    python -m ETrain.Eval.LLaVA.CoIN.convert_gqa_for_eval --src $output_file --dst $RESULT_DIR/$STAGE/testdev_balanced_predictions.json

    python -m ETrain.Eval.LLaVA.CoIN.eval_gqa --tier testdev_balanced --path $RESULT_DIR/$STAGE --output-dir $RESULT_DIR/$STAGE 

    python -m ETrain.Eval.LLaVA.CoIN.create_prompt \
        --rule ./ETrain/Eval/LLaVA/CoIN/rule.json \
        --questions ./playground/Instructions_Original/GQA/test.json \
        --results $output_file \

elif [ $QF == "sqa" ]; then
    python -m ETrain.Eval.LLaVA.CoIN.eval_science_qa \
    --base-dir ../DatasetCoIN/ScienceQA \
    --result-file $output_file \
    --output-file $RESULT_DIR/$STAGE/output.jsonl \
    --output-result $RESULT_DIR/$STAGE/output_result.jsonl \

    python -m ETrain.Eval.LLaVA.CoIN.create_prompt \
        --rule ./ETrain/Eval/LLaVA/CoIN/rule.json \
        --questions ./playground/Instructions_Original/ScienceQA/test.json \
        --results $output_file \

elif [ $QF == "textqa" ]; then
    python -m ETrain.Eval.LLaVA.CoIN.eval_textvqa \
        --annotation-file ../DatasetCoIN/TextVQA/TextVQA_0.5.1_val.json \
        --result-file $output_file \
        --output-dir $RESULT_DIR/$STAGE \

    python -m ETrain.Eval.LLaVA.CoIN.create_prompt \
        --rule ./ETrain/Eval/LLaVA/CoIN/rule.json \
        --questions ./playground/Instructions_Original/TextVQA/val.json \
        --results $output_file \

elif [ $QF == "ImageNet" ]; then
    python -m ETrain.Eval.LLaVA.CoIN.eval_ImagetNet \
        --test-file ./playground/Instructions_Original/ImageNet/test.json \
        --result-file $output_file \
        --output-dir $RESULT_DIR/$STAGE \

    python -m ETrain.Eval.LLaVA.CoIN.create_prompt \
        --rule ./ETrain/Eval/LLaVA/CoIN/rule.json \
        --questions ./playground/Instructions_Original/ImageNet/test.json \
        --results $output_file \

elif [ $QF == "vizwiz" ]; then
    python -m ETrain.Eval.LLaVA.CoIN.eval_vizwiz \
        --result-file $output_file \
        --annotation-file ./playground/Instructions_Original/VizWiz/val.json \
        --output-dir $RESULT_DIR/$STAGE \

    python -m ETrain.Eval.LLaVA.CoIN.create_prompt \
        --rule ./ETrain/Eval/LLaVA/CoIN/rule.json \
        --questions ./playground/Instructions_Original/VizWiz/val.json \
        --results $output_file \

elif [ $QF == "grounding" ]; then
    python -m ETrain.Eval.LLaVA.CoIN.eval_grounding \
        --test-file ./playground/Instructions_Original/Grounding/test.json \
        --result-file $output_file \
        --output-dir $RESULT_DIR/$STAGE \

    python -m ETrain.Eval.LLaVA.CoIN.create_prompt \
        --rule ./ETrain/Eval/LLaVA/CoIN/rule.json \
        --questions ./playground/Instructions_Original/Grounding/test.json \
        --results $output_file \
        --rule_temp CoIN_Grounding \

elif [ $QF == "vqav2" ]; then
    python -m ETrain.Eval.LLaVA.CoIN.eval_vqav2 \
        --result-file $output_file \
        --annotation-file ./playground/Instructions_Original/VQAv2/val.json \
        --output-dir $RESULT_DIR/$STAGE \

    python -m ETrain.Eval.LLaVA.CoIN.create_prompt \
        --rule ./ETrain/Eval/LLaVA/CoIN/rule.json \
        --questions ./playground/Instructions_Original/VQAv2/val.json \
        --results $output_file \

fi
