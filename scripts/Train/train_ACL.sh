#!/bin/bash
pip install -q -e .
bash scripts/Train/Task1.sh configs/model_configs/LLaVA/MLLM-ACL/train/task1.json configs/data_configs/MLLM-ACL/OCR.json
bash scripts/Train/eval_ACL.sh 1
pip install -q -e .
bash scripts/Train/Task1.sh configs/model_configs/LLaVA/MLLM-ACL/train/task2.json configs/data_configs/MLLM-ACL/Math.json
bash scripts/Train/eval_ACL.sh 2
pip install -q -e .
bash scripts/Train/Task1.sh configs/model_configs/LLaVA/MLLM-ACL/train/task3.json configs/data_configs/MLLM-ACL/VP.json
bash scripts/Train/eval_ACL.sh 3
pip install -q -e .
bash scripts/Train/Task1.sh configs/model_configs/LLaVA/MLLM-ACL/train/task4.json configs/data_configs/MLLM-ACL/APP.json
bash scripts/Train/eval_ACL.sh 4