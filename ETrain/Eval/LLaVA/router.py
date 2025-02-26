import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from ETrain.utils.LLaVA.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from ETrain.utils.LLaVA.conversation import conv_templates, SeparatorStyle
from ETrain.Models.LLaVA.builder import load_pretrained_model
from ETrain.utils.LLaVA.utils import disable_torch_init
from ETrain.utils.LLaVA.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image
import math



