################## VICUNA ##################
PROMPT_VERSION=v1
MODEL_VERSION="vicuna-7b-v1.5"
################## VICUNA ##################

# MODELPATH=../haloscope/models/meta-llama/Llama-3.1-8B-Instruct
MODELPATH=checkpoints/LLaVA/Vicuna/vicuna-7b-v1.5
################## LLaMA-2 ##################
# PROMPT_VERSION="llava_llama_2"
# MODEL_VERSION="Llama-2-7b-chat-hf"
################## LLaMA-2 ##################
    # --pretrain_mm_mlp_adapter ./checkpoints/LLaVA/Vicuna/vicuna-7b-v.15-projector/mm_projector.bin \
    # --mm_projector_type mlp2x_gelu \

deepspeed --include localhost:4,5,6,7 --master_port 29500 ETrain/Train/LLaVA/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --model_name_or_path $MODELPATH \
    --version $PROMPT_VERSION \
    --data_path ./playground/Instructions_Original/Multitask/mtrain.json \
    --image_folder ../DatasetCoIN \
    --vision_tower ./checkpoints/LLaVA/clip-vit-large-patch14-336 \
    --mm_projector_type mov_adapter \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/LLaVA/Agent/Router_Mov_llava_lora \
    --num_train_epochs 1 \
    --per_device_train_batch_size 12 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb