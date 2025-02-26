# 使用说明

该文档介绍如何使用 `*_eval_*.sh` 脚本进行模型评估。

## 脚本路径

`scripts/LLaVA/Eval_dom`

## 脚本功能

脚本用于在指定的 GPU 上运行模型评估，并将结果保存到指定目录。

## 使用方法

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash 10_eval_financial.sh [STAGE] [MODELPATH] [MODELNAME] [MODELBASE]
```

### 参数说明

- `STAGE` (可选): 评估阶段，默认为 `Finetune`。
- `MODELPATH` (可选): 模型路径，有默认值，为`CoIN`文件夹内的checkpoints。
- `MODELNAME` (可选): 模型名称，默认为 `<train_name>_<eval_name>`。
- `MODELBASE` (可选): 模型基准，默认为 `LLaVA`。

### 示例

```bash
bash 10_eval_financial.sh Finetune ./checkpoints/LLaVA/CoIN/FinVis_llava_lora FinVis_finvis LLaVA
```

## 输出

评估结果将保存在 `./results/CoIN/$MODELBASE/$MODELNAME/$STAGE/merge.jsonl` 文件中。
