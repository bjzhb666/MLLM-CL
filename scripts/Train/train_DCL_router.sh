#!/bin/bash
pip install -q -e .
bash scripts/Train/Task1.sh configs/model_configs_router/LLaVA/MLLM-DCL/train/task2.json configs/data_configs_router/MLLM-DCL/Med.json
pip install -q -e .
bash scripts/Train/Task1.sh configs/model_configs_router/LLaVA/MLLM-DCL/train/task3.json configs/data_configs_router/MLLM-DCL/AD.json
pip install -q -e .
bash scripts/Train/Task1.sh configs/model_configs_router/LLaVA/MLLM-DCL/train/task4.json configs/data_configs_router/MLLM-DCL/Sci.json
pip install -q -e .
bash scripts/Train/Task1.sh configs/model_configs_router/LLaVA/MLLM-DCL/train/task5.json configs/data_configs_router/MLLM-DCL/Fin.json