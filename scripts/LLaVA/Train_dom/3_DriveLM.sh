################## VICUNA ##################
PROMPT_VERSION=v1
MODEL_VERSION="vicuna-7b-v1.5"
################## VICUNA ##################

if [ ! $1 ]; then
    BASE_NAME="Finetune-CL"
else
    BASE_NAME=$1
fi
OUTPUT_DIR="./checkpoints/LLaVA/$BASE_NAME/DriveLM_llava_lora"
if [ ! $2 ]; then
    USE_PREVIOUS_TASK_MODEL=False
    PREVIOUS_TASK=""
else
    USE_PREVIOUS_TASK_MODEL=True
    PREVIOUS_TASK="--previous_task_model_path ./checkpoints/LLaVA/$BASE_NAME/Medical_llava_lora"
fi
if [ ! $3 ]; then
    EXPERT=""
else
    EXPERT="--expert_num $3"
    OUTPUT_DIR="./checkpoints/LLaVA/$BASE_NAME/DriveLM_llava_lora_MOE"
    PREVIOUS_TASK="--previous_task_model_path ./checkpoints/LLaVA/$BASE_NAME/Medical_llava_lora_MOE"
fi
################## LLaMA-2 ##################
# PROMPT_VERSION="llava_llama_2"
# MODEL_VERSION="Llama-2-7b-chat-hf"
################## LLaMA-2 ##################
DATA_PATH=/data/hongbo_zhao/data/Domain_data
echo ""
echo "PREVIOUS_TASK: $PREVIOUS_TASK"
echo ""
    # --previous_task_model_path ./checkpoints/LLaVA/$BASE_NAME/VQAv2_llava_lora \
deepspeed --include localhost:0,1,2,3,4,5,6,7 --master_port 29600 ETrain/Train/LLaVA/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    $EXPERT \
    $PREVIOUS_TASK \
    --model_name_or_path ./checkpoints/LLaVA/Vicuna/vicuna-7b-v1.5 \
    --version $PROMPT_VERSION \
    --data_path $DATA_PATH/AD/DriveLM/train.json \
    --image_folder $DATA_PATH/AD/DriveLM \
    --vision_tower ./checkpoints/LLaVA/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none \
    --run_name "LoRA_AD_bs4ac2_lr2e-5"
