if [ ! -n "$1" ] ;then
    STAGE='Finetune'
else
    STAGE=$1
fi

if [ ! -n "$2" ] ;then
    MODELPATH='./checkpoints/LLaVA/AvgLora_2/TextVQA_llava_lora'
    # MODELPATH='checkpoints/LLaVA/Vicuna/llava-7b-v1.5'
else
    MODELPATH=$2
fi

DATASET="TextVQA_textqa"
RESULT_DIR="./results/CoIN/LLaVA-1/$DATASET"
# AGENT_DIR="results/CoIN/LLaVA"
output_file=$RESULT_DIR/$STAGE/merge.jsonl

python -m ETrain.Eval.LLaVA.CoIN.eval_textvqa \
    --annotation-file ../DatasetCoIN/TextVQA/TextVQA_0.5.1_val.json \
    --result-file $output_file \
    --output-dir $RESULT_DIR/$STAGE \

python -m ETrain.Eval.LLaVA.CoIN.create_prompt \
    --rule ./ETrain/Eval/LLaVA/CoIN/rule.json \
    --questions ./playground/Instructions_Original/TextVQA/val.json \
    --results $output_file \