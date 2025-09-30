# Loop over triangular indices (j <= i)

pip install -q -e .

if [ $# -ne 1 ]; then
  echo "Usage: $0 <ACL|DCL>, e.g."
  echo "       $0 DCL"
  echo "       $0 ACL"
  exit 1
fi

set -x

a_or_dcl=$(echo $1 | tr '[:lower:]' '[:upper:]')

DATA_CONFIGS=()
if [ $a_or_dcl == "ACL" ]; then
  DATA_CONFIGS+=("OCR_test")
  DATA_CONFIGS+=("Math_test")
  DATA_CONFIGS+=("VP_test")
  DATA_CONFIGS+=("APP_test")
elif [ $a_or_dcl == "DCL" ]; then
  DATA_CONFIGS+=("RS")
  DATA_CONFIGS+=("Med")
  DATA_CONFIGS+=("AD")
  DATA_CONFIGS+=("Sci")
  DATA_CONFIGS+=("Fin")
else
  echo "Usage: $0 <ACL|DCL>"
  exit 1
fi

for i in $(seq 2 4); do
  # train on task i
  train_task=${DATA_CONFIGS[$i-1]}
  bash scripts/Train/Task1.sh configs/model_configs_router/*/MLLM-$a_or_dcl/train/task$i.json configs/data_configs_router/MLLM-$a_or_dcl/${train_task%_*}.json

  # eval on task 1-i
  # for j in $(seq 1 $i); do
  #   task=${DATA_CONFIGS[$j-1]}
  #   bash scripts/Eval_MR_LoRA/eval_use_router_$a_or_dcl.sh $task $task Finetune ~/ghy-cl-codebase/MCITlib/checkpoints/MLLM-$a_or_dcl/InternVL/MR-LoRA-router/Task${i}*
  # done
done

# cat results/$a_or_dcl/RouterFinal_Finetune/*/*2*/acc.txt
# cat results/$a_or_dcl/RouterFinal_Finetune/*/*3*/acc.txt
# cat results/$a_or_dcl/RouterFinal_Finetune/*/*4*/acc.txt
# cat results/$a_or_dcl/RouterFinal_Finetune/*/*5*/acc.txt