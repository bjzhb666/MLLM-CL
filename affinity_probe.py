"""
Affinity probing module for AdapteX.

Computes the affinity score S_{k,j} between a new task T_k and existing
expert E_j using the trained multimodal router. The score measures
bi-directional misclassification:

    S_{k,j} = (|{x in Q_k | R(x) = E_j}| + |{x in Q_j | R(x) = E_k_hat}|) / (|Q_k| + |Q_j|)

A high S_{k,j} (close to 1.0) indicates the two tasks are semantically
indistinguishable to the router, implying high affinity.
"""

import json
import os
from copy import deepcopy

import torch
from PIL import Image
from tqdm import tqdm

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


class AffinityProbe:
    """Probes task affinity using the trained multimodal router."""

    def __init__(self, router_model_path, model_base, conv_mode="vicuna_v1"):
        self.router_model_path = router_model_path
        self.model_base = model_base
        self.conv_mode = conv_mode
        self.model = None
        self.tokenizer = None
        self.image_processor = None

    def load_router(self):
        """Load the router model."""
        disable_torch_init()
        model_name = get_model_name_from_path(self.router_model_path)
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
            self.router_model_path, self.model_base, model_name
        )

    def _predict_expert(self, question, image_path, routing_prompt, prompt_after):
        """Run the router on a single sample and return the predicted expert letter."""
        qs_text = question.replace("<image>", "").strip()
        routing_qs = routing_prompt + qs_text + "\n" + prompt_after

        image_tensor = None
        if image_path and os.path.exists(image_path):
            image = Image.open(image_path).convert("RGB")
            image_tensor = process_images(
                [image], self.image_processor, self.model.config
            )[0]
            routing_qs = DEFAULT_IMAGE_TOKEN + "\n" + routing_qs

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], routing_qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(
            prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).cuda()

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).to(dtype=torch.float16, device="cuda")
                if image_tensor is not None else None,
                do_sample=False,
                max_new_tokens=4,
                use_cache=True,
            )

        input_len = input_ids.shape[1]
        output_text = self.tokenizer.decode(
            output_ids[0, input_len:], skip_special_tokens=True
        ).strip()

        for c in output_text:
            if c.isalpha() and c.isupper():
                return c
        return output_text[:1] if output_text else "?"

    def compute_affinity_scores(
        self,
        new_task_query,
        existing_query_sets,
        existing_expert_letters,
        new_expert_letter,
        routing_prompt,
        prompt_after,
        image_base_dir,
        new_task_image_dir,
        max_query_samples=200,
    ):
        """Compute affinity scores between new task and all existing experts.

        Args:
            new_task_query: List of query samples from the new task.
            existing_query_sets: Dict mapping task_name → list of query samples.
            existing_expert_letters: Dict mapping task_name → expert_letter.
            new_expert_letter: The provisional letter assigned to the new task.
            routing_prompt: Current routing prompt string.
            prompt_after: Prompt suffix after question.
            image_base_dir: Base directory for existing task images.
            new_task_image_dir: Image directory for the new task.
            max_query_samples: Max samples to evaluate per query set.

        Returns:
            Dict mapping existing_task_name → affinity_score.
        """
        if self.model is None:
            self.load_router()

        new_query = new_task_query[:max_query_samples]

        new_task_predictions = []
        print(f"  Probing new task query set ({len(new_query)} samples)...")
        for sample in tqdm(new_query, desc="New task queries"):
            question = self._extract_question(sample)
            image_path = self._get_image_path(sample, new_task_image_dir)
            pred = self._predict_expert(question, image_path, routing_prompt, prompt_after)
            new_task_predictions.append(pred)

        affinity_scores = {}

        for task_name, query_set in existing_query_sets.items():
            expert_letter = existing_expert_letters[task_name]
            existing_query = query_set[:max_query_samples]

            existing_predictions = []
            print(f"  Probing existing task '{task_name}' query set ({len(existing_query)} samples)...")
            for sample in tqdm(existing_query, desc=f"{task_name} queries"):
                question = self._extract_question(sample)
                task_image_dir = os.path.join(image_base_dir, task_name)
                image_path = self._get_image_path(sample, task_image_dir)
                pred = self._predict_expert(question, image_path, routing_prompt, prompt_after)
                existing_predictions.append(pred)

            count_new_to_existing = sum(
                1 for p in new_task_predictions if p == expert_letter
            )
            count_existing_to_new = sum(
                1 for p in existing_predictions if p == new_expert_letter
            )

            total = len(new_query) + len(existing_query)
            if total > 0:
                score = (count_new_to_existing + count_existing_to_new) / total
            else:
                score = 0.0

            affinity_scores[task_name] = score
            print(f"  Affinity S(new, {task_name}) = {score:.4f} "
                  f"(new→{task_name}: {count_new_to_existing}/{len(new_query)}, "
                  f"{task_name}→new: {count_existing_to_new}/{len(existing_query)})")

        return affinity_scores

    def decide_expansion(self, affinity_scores, threshold=0.2):
        """Make expand/reuse decision based on affinity scores.

        Args:
            affinity_scores: Dict mapping task_name → score.
            threshold: Affinity threshold τ.

        Returns:
            (decision, best_match):
                decision is "reuse" or "expand".
                best_match is the task name with highest affinity (or None).
        """
        if not affinity_scores:
            return "expand", None

        best_task = max(affinity_scores, key=affinity_scores.get)
        best_score = affinity_scores[best_task]

        if best_score > threshold:
            return "reuse", best_task
        else:
            return "expand", None

    def _extract_question(self, sample):
        """Extract question text from a data sample."""
        if "text" in sample:
            return sample["text"]
        if "conversations" in sample:
            for conv in sample["conversations"]:
                if conv.get("from") == "human":
                    return conv["value"]
        return ""

    def _get_image_path(self, sample, image_dir):
        """Get full image path from a sample."""
        image = sample.get("image", "")
        if not image:
            return None
        if os.path.isabs(image):
            return image
        return os.path.join(image_dir, image)

    def release_model(self):
        """Free GPU memory."""
        if self.model is not None:
            del self.model
            self.model = None
            torch.cuda.empty_cache()
