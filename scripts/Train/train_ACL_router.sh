#!/bin/bash
pip install -q -e .
bash scripts/Train/Task1.sh configs/model_configs_router/LLaVA/MLLM-ACL/train/task2.json configs/data_configs_router/MLLM-ACL/Math.json
pip install -q -e .
bash scripts/Train/Task1.sh configs/model_configs_router/LLaVA/MLLM-ACL/train/task3.json configs/data_configs_router/MLLM-ACL/VP.json
pip install -q -e .
bash scripts/Train/Task1.sh configs/model_configs_router/LLaVA/MLLM-ACL/train/task4.json configs/data_configs_router/MLLM-ACL/APP.json