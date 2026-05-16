"""
AdapteX router-based evaluation.

Two-stage inference:
  Step 1: Use the router LoRA to select the best expert (MLLM-based routing)
  Step 2: Use the selected expert LoRA to generate the final answer

This script dynamically loads the prompt manager state to handle
merged experts and adaptive routing prompts.
"""

import argparse
import json
import math
import os
import random
import sys
from copy import deepcopy

import shortuuid
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from llava.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from llava.conversation import conv_templates
from llava.mm_utils import (
    get_model_name_from_path,
    process_images,
    tokenizer_image_token,
)
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from adaptex.prompt_manager import PromptManager


def split_list(lst, n):
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def load_expert_results(result_folders, expert_names, qf):
    """Load pre-computed results from each expert for the given test set."""
    pars = [f"{name}_{qf}" for name in expert_names]
    expert_results = []

    for par in pars:
        result_dir = os.path.join(result_folders, par)
        if not os.path.exists(result_dir):
            print(f"[Warning] Expert results not found: {result_dir}")
            expert_results.append({})
            continue

        merge_file = os.path.join(result_dir, "merge.jsonl")
        if not os.path.exists(merge_file):
            print(f"[Warning] merge.jsonl not found in {result_dir}")
            expert_results.append({})
            continue

        answers = {}
        with open(merge_file, "r") as f:
            for line in f:
                data = json.loads(line)
                answers[str(data["question_id"])] = data
        expert_results.append(answers)

    return expert_results


def choose_answer(routing_output, expert_results, question_id, num_experts):
    """Select the answer from the expert chosen by the router."""
    qid = str(question_id)

    for c in routing_output:
        if c.isalpha() and c.isupper():
            idx = ord(c) - ord("A")
            if 0 <= idx < num_experts:
                if qid in expert_results[idx]:
                    return expert_results[idx][qid].get("text", ""), c
                else:
                    break

    valid_experts = [i for i in range(num_experts) if qid in expert_results[i]]
    if valid_experts:
        idx = random.choice(valid_experts)
        letter = chr(ord("A") + idx)
        return expert_results[idx][qid].get("text", ""), letter

    return "", "?"


def eval_with_router(args):
    """Run AdapteX router-based evaluation."""
    disable_torch_init()

    pm = PromptManager(setting=args.setting)
    pm.load(args.prompt_manager_path)

    routing_prompt = pm.get_routing_prompt()
    prompt_after = pm.get_prompt_after_question()
    num_experts = pm.get_num_experts()

    print(f"Setting: {args.setting}")
    print(f"Num experts: {num_experts}")
    print(f"Routing prompt:\n{routing_prompt[:200]}...")

    model_path = os.path.expanduser(args.router_model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path, args.model_base, model_name
    )

    with open(os.path.expanduser(args.question_file), "r") as f:
        questions = json.load(f)
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

    expert_names = []
    for expert in pm.experts:
        expert_names.extend(expert["tasks"])

    expert_results = load_expert_results(
        args.result_folders, expert_names, args.qf
    )

    expert_letter_to_idx = {}
    idx = 0
    for expert in pm.experts:
        for task in expert["tasks"]:
            expert_letter_to_idx[task] = idx
            idx += 1

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    agent_selection_count = {}

    for line in tqdm(questions, desc="Evaluating"):
        idx = line["question_id"]
        qs_text = line.get("text", "")
        qs_text = qs_text.replace("<image>", "").strip()

        full_routing_qs = routing_prompt + qs_text + "\n" + prompt_after

        image_tensor = None
        if line.get("image"):
            image_file = line["image"]
            img_path = os.path.join(args.image_folder, image_file)
            if os.path.exists(img_path):
                image = Image.open(img_path).convert("RGB")
                image_tensor = process_images(
                    [image], image_processor, model.config
                )[0]
                full_routing_qs = DEFAULT_IMAGE_TOKEN + "\n" + full_routing_qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], full_routing_qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(
            prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).cuda()

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).to(
                    dtype=torch.float16, device="cuda"
                ) if image_tensor is not None else None,
                do_sample=False,
                max_new_tokens=4,
                use_cache=True,
            )

        input_len = input_ids.shape[1]
        routing_output = tokenizer.decode(
            output_ids[0, input_len:], skip_special_tokens=True
        ).strip()

        final_ans, chosen_letter = choose_answer(
            routing_output, expert_results, idx, len(expert_results)
        )

        agent_selection_count[chosen_letter] = agent_selection_count.get(chosen_letter, 0) + 1

        ans_file.write(
            json.dumps({
                "question_id": idx,
                "prompt": qs_text,
                "agent_selection": routing_output,
                "chosen_expert": chosen_letter,
                "text": final_ans,
                "answer_id": shortuuid.uuid(),
                "model_id": model_name,
                "metadata": {},
            }) + "\n"
        )

    ans_file.close()

    print(f"\nAgent selection distribution: {agent_selection_count}")
    total = sum(agent_selection_count.values())
    for letter, count in sorted(agent_selection_count.items()):
        print(f"  {letter}: {count} ({count/total*100:.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AdapteX Router-Based Evaluation")
    parser.add_argument("--setting", type=str, required=True, choices=["DCL", "ACL"])
    parser.add_argument("--prompt-manager-path", type=str, required=True)
    parser.add_argument("--router-model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, required=True)
    parser.add_argument("--question-file", type=str, required=True)
    parser.add_argument("--image-folder", type=str, required=True)
    parser.add_argument("--result-folders", type=str, required=True)
    parser.add_argument("--answers-file", type=str, required=True)
    parser.add_argument("--qf", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    args = parser.parse_args()

    eval_with_router(args)
