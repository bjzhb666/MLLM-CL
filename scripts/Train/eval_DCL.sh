# #!/bin/bash

# first argument $1 should be 1,2,3,4,5

pip install -q -e .
bash scripts/Eval_MLLM_DCL/eval_ad.sh configs/model_configs/LLaVA/MLLM-DCL/eval/task$1.json configs/data_configs/MLLM-DCL/AD.json
pip install -q -e .
bash scripts/Eval_MLLM_DCL/eval_fin.sh configs/model_configs/LLaVA/MLLM-DCL/eval/task$1.json configs/data_configs/MLLM-DCL/Fin.json
pip install -q -e .
bash scripts/Eval_MLLM_DCL/eval_med.sh configs/model_configs/LLaVA/MLLM-DCL/eval/task$1.json configs/data_configs/MLLM-DCL/Med.json
pip install -q -e .
bash scripts/Eval_MLLM_DCL/eval_rs.sh configs/model_configs/LLaVA/MLLM-DCL/eval/task$1.json configs/data_configs/MLLM-DCL/RS.json
pip install -q -e .
bash scripts/Eval_MLLM_DCL/eval_sci.sh configs/model_configs/LLaVA/MLLM-DCL/eval/task$1.json configs/data_configs/MLLM-DCL/Sci.json