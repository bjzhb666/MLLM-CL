# python scripts/LLaVA/merge_lora_weights.py --model-path checkpoints/LLaVA/Ability/APP_llava_lora \
#     --model-base checkpoints/LLaVA/Vicuna/llava-7b-v1.5 --save-model-path checkpoints/LLaVA/Ability_full/APP
# python scripts/LLaVA/merge_lora_weights.py --model-path checkpoints/LLaVA/Ability/CODE_llava_lora \
#     --model-base checkpoints/LLaVA/Vicuna/llava-7b-v1.5 --save-model-path checkpoints/LLaVA/Ability_full/CODE
python scripts/LLaVA/merge_lora_weights.py --model-path checkpoints/LLaVA/Ability/OCR_llava_lora \
    --model-base checkpoints/LLaVA/Vicuna/llava-7b-v1.5 --save-model-path checkpoints/LLaVA/Ability_full/OCR
# python scripts/LLaVA/merge_lora_weights.py --model-path checkpoints/LLaVA/Ability/SAT_llava_lora \
#     --model-base checkpoints/LLaVA/Vicuna/llava-7b-v1.5 --save-model-path checkpoints/LLaVA/Ability_full/SAT