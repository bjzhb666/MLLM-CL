# 使用说明

## 简介
该脚本用于训练 LLaVA 模型的lora模块。每个任务单独训一个lora，不使用先前的任务训练

## 参数说明
1. `BASE_NAME` (可选): 基础名称，默认为 "CoIN"。
2. `USE_PREVIOUS_TASK_MODEL` (可选): 是否使用先前任务的模型，默认为 `False`。
3. `EXPERT` (可选): 专家数量。

## 使用方法

单专家训练：
`bash <script_name> CoIN True`

MOE训练：
`bash <script_name> CoIN True 8`

不使用先前任务训练：
`bash <script_name>`