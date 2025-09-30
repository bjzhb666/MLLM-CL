#!/bin/bash
pip install -q -e .
bash scripts/Train/Task1.sh configs/model_configs/LLaVA/MLLM-DCL/train/task1.json configs/data_configs/MLLM-DCL/RS.json
bash scripts/Train/eval_DCL.sh 1
pip install -q -e .
bash scripts/Train/Task1.sh configs/model_configs/LLaVA/MLLM-DCL/train/task2.json configs/data_configs/MLLM-DCL/Med.json
bash scripts/Train/eval_DCL.sh 2
pip install -q -e .
bash scripts/Train/Task1.sh configs/model_configs/LLaVA/MLLM-DCL/train/task3.json configs/data_configs/MLLM-DCL/AD.json
bash scripts/Train/eval_DCL.sh 3
pip install -q -e .
bash scripts/Train/Task1.sh configs/model_configs/LLaVA/MLLM-DCL/train/task4.json configs/data_configs/MLLM-DCL/Sci.json
bash scripts/Train/eval_DCL.sh 4
pip install -q -e .
bash scripts/Train/Task1.sh configs/model_configs/LLaVA/MLLM-DCL/train/task5.json configs/data_configs/MLLM-DCL/Fin.json
bash scripts/Train/eval_DCL.sh 5