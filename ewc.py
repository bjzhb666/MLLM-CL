"""
Elastic Weight Consolidation (EWC) for AdapteX expert reuse.

When an existing expert E_j is reused for a new task T_k, we apply
EWC regularization to prevent catastrophic forgetting:

    L_total = L_task(T_k; θ_j*) + λ * Σ F_w * (w - w_old)²

where F_w is the diagonal Fisher information matrix (approximated),
computed and stored upon the completion of each expert's training.
w_old denotes the parameter snapshot of E_j* prior to learning T_k.
"""

import json
import os
from copy import deepcopy

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def compute_fisher_information(model, tokenizer, data_path, image_folder,
                                num_samples=200, batch_size=1):
    """Compute the diagonal Fisher information matrix for LoRA parameters.

    Uses few-shot memory data from the original task to estimate
    parameter importance.

    Args:
        model: The MLLM with LoRA adapter loaded.
        tokenizer: The tokenizer.
        data_path: Path to the few-shot memory JSON.
        image_folder: Directory containing images.
        num_samples: Max samples to use for Fisher computation.
        batch_size: Batch size for processing.

    Returns:
        Dict mapping parameter_name → Fisher diagonal values (on CPU).
    """
    from llava.train.train import (
        LazySupervisedDataset,
        DataCollatorForSupervisedDataset,
        preprocess_multimodal,
    )
    from dataclasses import dataclass, field
    from typing import Optional

    model.eval()

    fisher = {}
    for name, param in model.named_parameters():
        if param.requires_grad and "lora_" in name:
            fisher[name] = torch.zeros_like(param, device="cpu")

    with open(data_path, "r") as f:
        data = json.load(f)

    data = data[:num_samples]

    for sample in tqdm(data, desc="Computing Fisher information"):
        model.zero_grad()

        conversations = sample.get("conversations", [])
        if not conversations:
            continue

        human_msg = ""
        gpt_msg = ""
        for conv in conversations:
            if conv["from"] == "human":
                human_msg = conv["value"]
            elif conv["from"] == "gpt":
                gpt_msg = conv["value"]

        if not human_msg or not gpt_msg:
            continue

        try:
            from llava.mm_utils import tokenizer_image_token
            from llava.constants import IMAGE_TOKEN_INDEX, IGNORE_INDEX
            from llava.conversation import conv_templates
            from PIL import Image
            from llava.mm_utils import process_images

            image_tensor = None
            if sample.get("image"):
                img_path = sample["image"]
                if not os.path.isabs(img_path):
                    img_path = os.path.join(image_folder, img_path)
                if os.path.exists(img_path):
                    image = Image.open(img_path).convert("RGB")
                    image_tensor = process_images(
                        [image], model.get_vision_tower().image_processor, model.config
                    )[0]

            conv = conv_templates["v1"].copy()
            conv.append_message(conv.roles[0], human_msg)
            conv.append_message(conv.roles[1], gpt_msg)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(
                prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            ).unsqueeze(0).cuda()

            target_ids = input_ids.clone()

            model.train()
            outputs = model(
                input_ids=input_ids,
                labels=target_ids,
                images=image_tensor.unsqueeze(0).to(
                    dtype=torch.float16, device="cuda"
                ) if image_tensor is not None else None,
            )

            loss = outputs.loss
            loss.backward()

            for name, param in model.named_parameters():
                if name in fisher and param.grad is not None:
                    fisher[name] += (param.grad.detach().cpu() ** 2)

        except Exception as e:
            print(f"[Warning] Fisher computation failed for sample: {e}")
            continue

    num_valid = max(len(data), 1)
    for name in fisher:
        fisher[name] /= num_valid

    model.eval()
    return fisher


def save_fisher(fisher, path):
    """Save Fisher information to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(fisher, path)


def load_fisher(path):
    """Load Fisher information from disk."""
    return torch.load(path, map_location="cpu")


class EWCRegularizer:
    """EWC regularization for LoRA parameter reuse.

    Computes L_ewc = λ * Σ F_w * (w - w_old)² and adds it to the task loss.
    """

    def __init__(self, fisher, old_params, lambda_ewc=0.5):
        """
        Args:
            fisher: Dict mapping param_name → Fisher diagonal (on CPU).
            old_params: Dict mapping param_name → parameter values before new task (on CPU).
            lambda_ewc: Regularization coefficient λ.
        """
        self.fisher = fisher
        self.old_params = old_params
        self.lambda_ewc = lambda_ewc

    def compute_ewc_loss(self, model):
        """Compute the EWC penalty term.

        Returns:
            Scalar tensor on the same device as the model.
        """
        ewc_loss = torch.tensor(0.0, device="cuda")

        for name, param in model.named_parameters():
            if name in self.fisher and name in self.old_params:
                fisher_diag = self.fisher[name].to(param.device)
                old_param = self.old_params[name].to(param.device)
                ewc_loss += (fisher_diag * (param - old_param) ** 2).sum()

        return self.lambda_ewc * ewc_loss

    @classmethod
    def from_checkpoint(cls, fisher_path, lora_checkpoint_path, lambda_ewc=0.5):
        """Create EWCRegularizer from saved Fisher and LoRA checkpoint.

        Args:
            fisher_path: Path to saved Fisher info.
            lora_checkpoint_path: Path to the LoRA checkpoint directory.
            lambda_ewc: Regularization strength.
        """
        from peft.utils import WEIGHTS_NAME

        fisher = load_fisher(fisher_path)

        lora_weights_path = os.path.join(lora_checkpoint_path, WEIGHTS_NAME)
        old_params = torch.load(lora_weights_path, map_location="cpu")

        return cls(fisher, old_params, lambda_ewc)


def get_lora_state_dict(model):
    """Extract current LoRA parameters as a CPU state dict."""
    state = {}
    for name, param in model.named_parameters():
        if "lora_" in name and param.requires_grad:
            state[name] = param.detach().cpu().clone()
    return state
