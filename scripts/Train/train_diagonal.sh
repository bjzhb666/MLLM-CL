#!/bin/bash

if [ $# -ne 2 ]; then
  echo "Usage: $0 <ACL|DCL> <task_id:1-5>, e.g."
  echo "       $0 DCL 1     # train RS, eval RS"
  echo "       $0 ACL 4     # train APP, eval APP"
  exit 1
fi

set -x

a_or_dcl=$(echo $1 | tr '[:lower:]' '[:upper:]')
task_id=$2

DATA_CONFIGS=()
if [ $a_or_dcl == "ACL" ]; then
  DATA_CONFIGS+=("OCR")
  DATA_CONFIGS+=("Math")
  DATA_CONFIGS+=("VP")
  DATA_CONFIGS+=("APP")
elif [ $a_or_dcl == "DCL" ]; then
  DATA_CONFIGS+=("RS")
  DATA_CONFIGS+=("Med")
  DATA_CONFIGS+=("AD")
  DATA_CONFIGS+=("Sci")
  DATA_CONFIGS+=("Fin")
else
  echo "Usage: $0 <ACL|DCL> <task_id>"
  exit 1
fi

task_name=${DATA_CONFIGS[${task_id}-1]}
script_name=$(echo $task_name | tr '[:upper:]' '[:lower:]')

MODEL_TRAIN_CONFIG=configs/model_configs/LLaVA/MLLM-$a_or_dcl/train/task$task_id.json
MODEL_EVAL_CONFIG=configs/model_configs/LLaVA/MLLM-$a_or_dcl/eval/task$task_id.json
DATA_CONFIG=configs/data_configs/MLLM-$a_or_dcl/$task_name.json

bash scripts/Train/Task1.sh $MODEL_TRAIN_CONFIG $DATA_CONFIG
bash scripts/Eval_MLLM_$a_or_dcl/eval_$script_name.sh $MODEL_EVAL_CONFIG $DATA_CONFIG