# #!/bin/bash

# first argument $1 should be 1,2,3,4

pip install -q -e .
bash scripts/Eval_MLLM_ACL/eval_VP.sh configs/model_configs/LLaVA/MLLM-ACL/eval/task$1.json configs/data_configs/MLLM-ACL/VP.json
pip install -q -e .
bash scripts/Eval_MLLM_ACL/eval_Math.sh configs/model_configs/LLaVA/MLLM-ACL/eval/task$1.json configs/data_configs/MLLM-ACL/Math.json
pip install -q -e .
bash scripts/Eval_MLLM_ACL/eval_OCR.sh configs/model_configs/LLaVA/MLLM-ACL/eval/task$1.json configs/data_configs/MLLM-ACL/OCR.json
pip install -q -e .
bash scripts/Eval_MLLM_ACL/eval_APP.sh configs/model_configs/LLaVA/MLLM-ACL/eval/task$1.json configs/data_configs/MLLM-ACL/APP.json