# train rounter with lora using merge lora as initial model

Train_Epoch=24
SAVE_DIR=./checkpoints/Router_Ability/Router_llava_lora_5e-6-ep$Train_Epoch

deepspeed --include localhost:0,1,2,3,4,5,6,7 --master_port 29600 ETrain/Train/LLaVA/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 5e-6 \
    --model_name_or_path ./checkpoints/LLaVA/Vicuna/vicuna-7b-v1.5 \
    --data_path /data/hongbo_zhao/code/LLaVA/Ability_data/Router_train30.json \
    --image_folder /data/hongbo_zhao/code/LLaVA/Ability_data/sample_images30 \
    --vision_tower ./checkpoints/LLaVA/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter ./checkpoints/LLaVA/Vicuna/vicuna-7b-v.15-projector/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir $SAVE_DIR \
    --num_train_epochs $Train_Epoch \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --learning_rate 5e-6 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name "LoRA_Router_abi_bs4ac1_lr5e-6-ep$Train_Epoch"

# find $SAVE_DIR/ -mindepth 1 -maxdepth 1 -type d -exec rm -rf {} +