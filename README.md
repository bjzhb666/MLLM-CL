# MLLM-CL: Continual Learning for Multimodal Large Language Models
![MLLM-CL Benchmark](image.png)


This is the official repo of MLLM-CL and MR-LoRA. MLLM-CL is a novel benchmark encompassing domain and ability continual learning, where the former focuses on independently and identically distributed (IID) evaluation across evolving mainstream domains, whereas the latter evaluates on non-IID scenarios with emerging model ability. MR-LoRA prevents catastrophic interference through parameter isolation and an MLLM-based routing mechanism. For more details, please refer to: 

**MLLM-CL: Continual Learning for Multimodal Large Language Models** [[paper]()].

[‪Hongbo Zhao](https://scholar.google.com/citations?user=Gs22F0UAAAAJ&hl=zh-CN), [Fei Zhu](https://impression2805.github.io/), Rundong Wang, [‪Gaofeng Meng](https://scholar.google.com/citations?hl=zh-CN&user=5hti_r0AAAAJ), [‪Zhaoxiang Zhang‬](https://scholar.google.com/citations?hl=zh-CN&user=qxWfV6cAAAAJ)


## MLLM-CL Benchmark
MLLM-CL is a benchmark for continual learning in multimodal large language models (MLLMs). It consists of two main components: domain continual learning and ability continual learning. The benchmark includes a variety of datasets and tasks to evaluate the performance of MLLMs in evolving scenarios.
### Domain Continual Learning
Continually adding domain knowledge is crucial for constructing a powerful MLLM.

To achieve this goal, we propose domain continual learning and choose five mainstream and common domains: remote sensing, medical, science, autonomous driving and finance.
In domain continual learning, the training 
### Ability Continual Learning
Domain continual learning assumes that training and test data are IID.
However, achieving IID between training and test sets is often challenging in real-world scenarios.

We select four fundamental abilities for the MLLM to learn sequentially: OCR, math \& logic, visual perception and GUI agent.

## MR-LoRA

![MR-LoRA framework](image-1.png)

Our MR-LoRA performs two-stage inference for a given multimodal input, consisting of a routing phase followed by a prediction phase. In the first stage, the expert selection router is performed to select a domain or ability-specific expert. Then, the selected expert is combined with the pre-trained backbone to output the final response.
## Installation
1. Clone this repository and navigate to CoIN folder
``` 
git clone https://github.com/bjzhb666/Coin.git
cd CoIN 
```
2. Install Package
```
conda env create -f environment.yaml
conda activate mrlora
```

3. Install additional packages for training cases
```
pip install -e ".[train]" -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
conda install git
pip install flash-attn==2.7.0.post2 --no-build-isolation -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

huggingface-cli download liuhaotian/llava-v1.5-7b --local-dir checkpoints/LLaVA/Vicuna/llava-7b-v1.5
huggingface-cli download openai/clip-vit-large-patch14-336 --local-dir checkpoints/LLaVA/clip-vit-large-patch14-336

huggingface-cli download liuhaotian/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5 --local-dir checkpoints/LLaVA/Vicuna/vicuna-7b-v.15-projector
huggingface-cli download lmsys/vicuna-7b-v1.5  --local-dir checkpoints/LLaVA/Vicuna/vicuna-7b-v1.5
```

If you meet a problem, maybe you could find some solutions in issuses.

## Dataset
Please download the images of MLLM-CL from huggingface.: 

After downloading all of them, organize the data as follows:
```
├── COCO2014
│   └── train2014
├── GQA
│   └── images
├── OCR-VQA
│   └── images
├── TextVQA
│   └── train_images
│   └── test_images
```

Then, please download the instructions from our datasets path: [CoIN_Dataset](https://huggingface.co/datasets/Zacks-Chen/CoIN/tree/main)
then, organize the instructions as follows:
```
├── Instruction_Original
│   └── GQA
│       └── train.json
│       └── test.json
│   └── ScienceQA
│       └── train.json
│       └── test.json
├── Instruction_Type2
│   └── GQA
│       └── train.json
│       └── test.json
```

## Instruction Tuning
First, downloading the pretrained projectors in [LLaVA Model_Zoo](https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md).

Setting `pretrain_mm_mlp_adapter` to the projector path.
You could modify the `deepspeed config` to change the deepspeed config.

We provide the scripts of our train order in `scripts/*/Train`.
Note, the `output_dir` of the previous script is the `previous_task_model_path` of the next training process.
Then, you could tune these datasets in your order.

We provide scripts for training MOELoRA with LLaVA in `scripts/LLaVA/Train_MOE`. Additionally, you can modify the code to train MiniGPT-V2 and Qwen-VL, following the example in lines 138-152 of `ETrain/Models/LLaVA/utils.py`.

## Evaluation
We have prepared the scripts to evaluate the trained model in `scripts/*/Eval`.

These scripts will evalute the trained model and create the prompts (`prompt_to_eval.json`) for evaluating the general knowldege.

To evaluate the general knowldege, you could add the result path to `scripts/Eval_GeneralKnowledge/eval_prompt_slim.sh` and run it, this script file will output a score to indicate the general knowledge.

## To Do
1. - [x] Evaluating on more MLLM, MiniGPT-4, ~~MiniGPT-V2~~, InstrctBlip, ~~Qwen-VL~~; MiniGPT-V2, Qwen-VL have been merged. In addition, since MiniGPT-4 and InstrctBlip are based on LAVIS resp, you can modify the config to train with these model.
2. - [] Evaluating on different size of MLLM; We are conducting experiments with larger model, 13b llava.
3. - [] Evaluating on full finetune.

## Citation
```
@misc{chen2024coin,
    title={CoIN: A Benchmark of Continual Instruction tuNing for Multimodel Large Language Model}, 
    author={Cheng Chen and Junchen Zhu and Xu Luo and Hengtao Shen and Lianli Gao and Jingkuan Song},
    year={2024},
    eprint={2403.08350},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

## Acknowledgement
[LLaVA](https://github.com/haotian-liu/LLaVA): the codebase we built upon, and our base model LLaVA-1.5-7b that has the amazing vision-language capabilities!

[LAVIS](https://github.com/salesforce/LAVIS): the codebase MiniGPT and InstructBlip are built upon.

[MiniGPT](https://github.com/Vision-CAIR/MiniGPT-4.git): the codebase of MinigGPT and MinitGPT-v2.
