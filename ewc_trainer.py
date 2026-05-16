"""
EWC-augmented LLaVA Trainer for AdapteX expert reuse.

Extends the standard LLaVA trainer with Elastic Weight Consolidation
loss to prevent catastrophic forgetting when reusing an existing expert
for a new task.
"""

import os
import sys

import torch
from transformers import Trainer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from llava.train.llava_trainer import LLaVATrainer


class EWCLLaVATrainer(LLaVATrainer):
    """LLaVA trainer with EWC regularization support."""

    def __init__(self, ewc_regularizer=None, **kwargs):
        super().__init__(**kwargs)
        self.ewc_regularizer = ewc_regularizer
        self._ewc_loss_log = []

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        task_loss = outputs.loss

        if self.ewc_regularizer is not None:
            ewc_loss = self.ewc_regularizer.compute_ewc_loss(model)
            total_loss = task_loss + ewc_loss

            if self.state.global_step % 50 == 0:
                self._ewc_loss_log.append({
                    "step": self.state.global_step,
                    "task_loss": task_loss.item(),
                    "ewc_loss": ewc_loss.item(),
                    "total_loss": total_loss.item(),
                })
        else:
            total_loss = task_loss

        return (total_loss, outputs) if return_outputs else total_loss
