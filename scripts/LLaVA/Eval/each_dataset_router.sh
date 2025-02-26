bash scripts/LLaVA/Eval/0_eval_router.sh ScienceQA sqa Finetune checkpoints/LLaVA/Vicuna/llava-7b-v1.5
wait
bash scripts/LLaVA/Eval/0_eval_router.sh TextVQA textqa Finetune checkpoints/LLaVA/Vicuna/llava-7b-v1.5
wait
bash scripts/LLaVA/Eval/0_eval_router.sh GQA gqa Finetune checkpoints/LLaVA/Vicuna/llava-7b-v1.5
wait
bash scripts/LLaVA/Eval/0_eval_router.sh VizWiz vizwiz Finetune checkpoints/LLaVA/Vicuna/llava-7b-v1.5
wait
bash scripts/LLaVA/Eval/0_eval_router.sh Grounding grounding Finetune checkpoints/LLaVA/Vicuna/llava-7b-v1.5
wait
bash scripts/LLaVA/Eval/0_eval_router.sh VQAv2 vqav2 Finetune checkpoints/LLaVA/Vicuna/llava-7b-v1.5