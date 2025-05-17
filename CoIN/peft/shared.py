# -*- encoding: utf-8 -*-
# here put the import lib
import torch.nn as nn
from .utils import PeftConfig


class Gate(nn.Module):
    """Gate"""
    def __init__(self, peft_config: PeftConfig, adapter_name="default"):

        super().__init__()

        self.expert_num = peft_config.expert_num
        self.te_dim = peft_config.task_embedding_dim

        self.GateL = nn.Linear(self.te_dim, self.expert_num, bias=False)
        self.act = nn.Softmax(dim=0)   
    
    def forward(self, task_em):

        #task_em = self.lora_task_embedding(x)
        y = self.GateL(task_em)
        y = self.act(y)

        return y