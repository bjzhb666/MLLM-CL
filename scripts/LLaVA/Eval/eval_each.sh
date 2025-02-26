models="ScienceQA TextVQA GQA VizWiz Grounding VQAv2"
# models="Grounding"
dataset=("vqav2")
ids=(7)

for ((i=0; i<${#dataset[@]}; i++)); do
    # index=$((i + 1))
    index=${ids[i]}
    echo "---${index}---"
    for model in $models; do
        bash scripts/LLaVA/Eval/only_eval.sh ${model} ${dataset[i]}
        wait
    done
done