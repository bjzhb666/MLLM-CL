"""
Router training data preparation for AdapteX.

Handles partitioning task data into support/query sets and combining
support sets across tasks for router training.
"""

import json
import os
import random
from copy import deepcopy


def prepare_support_query_split(data_path, support_ratio=0.8, seed=42, max_support=None):
    """Split task training data into support and query sets.

    The support set is used for router training (image-question pairs with expert labels).
    The query set is used for affinity probing.

    Args:
        data_path: Path to the task's train.json file.
        support_ratio: Fraction of data to use as support.
        seed: Random seed for reproducibility.
        max_support: Maximum number of support samples (for efficiency).

    Returns:
        (support_data, query_data): Two lists of data samples.
    """
    with open(data_path, "r") as f:
        data = json.load(f)

    random.seed(seed)
    data_copy = deepcopy(data)
    random.shuffle(data_copy)

    split_idx = int(len(data_copy) * support_ratio)
    support_data = data_copy[:split_idx]
    query_data = data_copy[split_idx:]

    if max_support and len(support_data) > max_support:
        support_data = support_data[:max_support]

    return support_data, query_data


def create_router_training_data(
    task_support_sets,
    task_names,
    expert_assignments,
    output_path,
    image_base_dir,
    setting="DCL",
):
    """Create combined router training data with expert labels.

    For each sample in S_all = S_k ∪ {S_1, ..., S_{k-1}}, format as:
    - Image + question → expert token (A/B/C/...)

    Args:
        task_support_sets: Dict mapping task_name → list of support samples.
        task_names: List of task names in order.
        expert_assignments: Dict mapping task_name → expert_letter (A/B/C...).
        output_path: Where to save the combined training JSON.
        image_base_dir: Base directory for images.
        setting: "DCL" or "ACL".

    Returns:
        Path to the saved router training data.
    """
    from .prompt_manager import PromptManager

    pm = PromptManager(setting=setting)
    for name in task_names:
        if name in expert_assignments:
            pm.add_expert(name)

    routing_prompt = pm.get_routing_prompt()
    prompt_after = pm.get_prompt_after_question()
    combined_data = []

    for task_name in task_names:
        if task_name not in task_support_sets:
            continue
        expert_letter = expert_assignments[task_name]
        support = task_support_sets[task_name]

        for sample in support:
            question = sample.get("text", sample.get("conversations", [{}])[0].get("value", ""))
            if "<image>" in question:
                question = question.replace("<image>\n", "").replace("<image>", "").strip()

            router_conversations = [
                {
                    "from": "human",
                    "value": f"<image>\n{routing_prompt}{question}\n{prompt_after}",
                },
                {
                    "from": "gpt",
                    "value": expert_letter,
                },
            ]

            entry = deepcopy(sample)
            entry["conversations"] = router_conversations
            if "image" in entry:
                if not os.path.isabs(entry["image"]):
                    task_folder = _get_task_folder(task_name, setting)
                    entry["image"] = os.path.join(task_folder, entry["image"])
            combined_data.append(entry)

    random.shuffle(combined_data)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(combined_data, f, indent=2)

    return output_path


def _get_task_folder(task_name, setting):
    """Map task name to its subfolder name."""
    if setting == "DCL":
        return task_name
    else:
        folder_map = {
            "OCR": "OCR",
            "Math": "Math",
            "VP": "VP",
            "APP": "APP",
        }
        return folder_map.get(task_name, task_name)


def save_query_set(query_data, output_path):
    """Save a query set to disk for later affinity probing."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(query_data, f, indent=2)


def load_query_set(path):
    """Load a previously saved query set."""
    with open(path, "r") as f:
        return json.load(f)


def save_few_shot_memory(data, output_path, num_samples=200, seed=42):
    """Save a few-shot support memory for future reference."""
    random.seed(seed)
    samples = deepcopy(data)
    random.shuffle(samples)
    memory = samples[:num_samples]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(memory, f, indent=2)

    return output_path
