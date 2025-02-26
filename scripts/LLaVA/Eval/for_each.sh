# models=("GQA")
# models="GQA Grounding ImageNet ScienceQA TextVQA VizWiz VQAv2"
models="ScienceQA TextVQA GQA VizWiz Grounding"
# dataset=("sqa" "textqa" "gqa" "vizwiz" "grounding" "vqav2")
# ids=(1 2 4 5 6 7)
# models="VizWiz Grounding VQAv2"
# dataset=("sqa" "textqa" "vqav2" "grounding" "vizwiz" "ImageNet")
# dataset=("sqa" "textqa")
# ids=(1 2)
dataset=("gqa" "vizwiz")
ids=(4 5)
# dataset=("grounding" "vqav2")
# ids=(6 7)

for ((i=0; i<${#dataset[@]}; i++)); do
    # index=$((i + 1))
    index=${ids[i]}
    echo "---${index}---"
    for model in $models; do
        # index=1
        # echo "${index} ${dataset[i]}"
        bash scripts/LLaVA/Eval/${index}_eval_${dataset[i]}.sh Finetune checkpoints/LLaVA/Cont_b16/${model}_llava_lora ${model}_${dataset[i]} LLaVA-Cont-nf
        wait
    done
done
