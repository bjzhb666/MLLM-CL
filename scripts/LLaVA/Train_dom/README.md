# 使用说明

## 简介
该脚本用于训练 LLaVA 模型，基于 Vicuna 或 LLaMA-2 进行微调。脚本支持使用先前任务的模型，并可以指定专家数量。

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
`bash <script_name> CoIN`