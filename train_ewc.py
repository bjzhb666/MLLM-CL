"""
Modified training entry point for AdapteX with EWC support.

Usage:
    # Standard expert training (expansion - new expert):
    deepspeed adaptex/train_ewc.py --lora_enable True ...

    # Expert reuse with EWC:
    deepspeed adaptex/train_ewc.py --lora_enable True \
        --ewc_enabled True \
        --ewc_lambda 0.5 \
        --fisher_path /path/to/fisher.pt \
        --reuse_lora_path /path/to/existing_lora ...
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from llava.train.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
replace_llama_attn_with_flash_attn()

import copy
import json
import logging
import pathlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch
import transformers
from peft.utils import WEIGHTS_NAME, set_peft_model_state_dict
from PIL import Image, ImageFile

from llava import conversation as conversation_lib
from llava.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
)
from llava.mm_utils import tokenizer_image_token
from llava.model import *
from llava.model.utils import find_all_vision_linear_names
from llava.train.train import (
    ModelArguments,
    DataArguments,
    find_all_linear_names,
    safe_save_model_for_hf_trainer,
    get_peft_state_maybe_zero_3,
    get_peft_state_non_lora_maybe_zero_3,
    load_model_from_previous_task,
    make_supervised_data_module,
    rank0_print,
)
from adaptex.ewc import EWCRegularizer, compute_fisher_information, get_lora_state_dict
from adaptex.ewc_trainer import EWCLLaVATrainer

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

local_rank = None


@dataclass
class EWCTrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length."},
    )
    double_quant: bool = field(default=True)
    quant_type: str = field(default="nf4")
    bits: int = field(default=16)
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)
    # EWC-specific arguments
    ewc_enabled: bool = field(default=False, metadata={"help": "Enable EWC regularization for expert reuse."})
    ewc_lambda: float = field(default=0.5, metadata={"help": "EWC regularization coefficient λ."})
    fisher_path: Optional[str] = field(default=None, metadata={"help": "Path to pre-computed Fisher information."})
    reuse_lora_path: Optional[str] = field(default=None, metadata={"help": "Path to LoRA checkpoint of the expert being reused."})
    compute_fisher_after: bool = field(default=False, metadata={"help": "Compute and save Fisher information after training completes."})
    use_vision_lora: bool = field(default=False)


def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, EWCTrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    compute_dtype = (
        torch.float16
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )

    bnb_model_from_pretrained_args = {}

    if model_args.vision_tower is not None:
        model = LlavaLlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation="flash_attention_2",
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            **bnb_model_from_pretrained_args,
        )
    else:
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation="flash_attention_2",
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            **bnb_model_from_pretrained_args,
        )
    model.config.use_cache = False

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )

        if training_args.bf16:
            model.to(torch.bfloat16)
        if training_args.fp16:
            model.to(torch.float16)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
    )

    tokenizer.pad_token = tokenizer.unk_token
    if model_args.version in conversation_lib.conv_templates:
        conversation_lib.default_conversation = conversation_lib.conv_templates[
            model_args.version
        ]
    else:
        conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(
            model_args=model_args, fsdp=training_args.fsdp
        )
        vision_tower = model.get_vision_tower()
        vision_tower.to(
            dtype=torch.bfloat16 if training_args.bf16 else torch.float16,
            device=training_args.device,
        )

        if model_args.use_vision_lora:
            lora_config.target_modules += find_all_vision_linear_names(model)

        rank0_print(training_args.lora_r, "Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)

        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True
        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length
        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = (
            model_args.tune_mm_mlp_adapter
        )
        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False
        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = (
            model_args.mm_use_im_start_end
        )
        model.config.mm_projector_lr = training_args.mm_projector_lr
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    for p in model.get_model().mm_projector.parameters():
        p.requires_grad = True

    if model_args.previous_task_model_path is not None:
        load_model_from_previous_task(model, model_args.previous_task_model_path)

    # --- EWC Setup ---
    ewc_regularizer = None
    if training_args.ewc_enabled:
        rank0_print("Setting up EWC regularization...")

        if training_args.reuse_lora_path is not None:
            rank0_print(f"Loading LoRA weights from reused expert: {training_args.reuse_lora_path}")
            load_model_from_previous_task(model, training_args.reuse_lora_path)

        old_params = get_lora_state_dict(model)

        if training_args.fisher_path and os.path.exists(training_args.fisher_path):
            rank0_print(f"Loading pre-computed Fisher from: {training_args.fisher_path}")
            from adaptex.ewc import load_fisher
            fisher = load_fisher(training_args.fisher_path)
        else:
            rank0_print("[Warning] EWC enabled but no pre-computed Fisher found. Using uniform Fisher.")
            fisher = {}
            for name, param in model.named_parameters():
                if "lora_" in name and param.requires_grad:
                    fisher[name] = torch.ones_like(param, device="cpu")

        ewc_regularizer = EWCRegularizer(fisher, old_params, training_args.ewc_lambda)
        rank0_print(f"EWC regularizer initialized with λ={training_args.ewc_lambda}")

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    trainer = EWCLLaVATrainer(
        ewc_regularizer=ewc_regularizer,
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module,
    )

    trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(
                non_lora_state_dict,
                os.path.join(training_args.output_dir, "non_lora_trainables.bin"),
            )

            if ewc_regularizer is not None and trainer._ewc_loss_log:
                log_path = os.path.join(training_args.output_dir, "ewc_loss_log.json")
                with open(log_path, "w") as f:
                    json.dump(trainer._ewc_loss_log, f, indent=2)
                rank0_print(f"EWC loss log saved to: {log_path}")

            # Compute and save Fisher after training for future reuse
            if training_args.compute_fisher_after and training_args.fisher_path:
                rank0_print("Computing Fisher information from training data...")
                fisher = compute_fisher_information(
                    model, tokenizer,
                    data_args.data_path,
                    data_args.image_folder,
                )
                from adaptex.ewc import save_fisher
                save_fisher(fisher, training_args.fisher_path)
                rank0_print(f"Fisher saved to: {training_args.fisher_path}")
    else:
        safe_save_model_for_hf_trainer(
            trainer=trainer, output_dir=training_args.output_dir
        )


if __name__ == "__main__":
    train()
