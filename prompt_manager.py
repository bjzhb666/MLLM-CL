"""
Dynamic router prompt management for AdapteX.

Manages expert descriptions in the routing prompt, handling both expansion
(adding new expert descriptions) and reuse (merging expert descriptions
when two tasks share the same expert).
"""

import json
import os
from copy import deepcopy


DCL_EXPERT_DESCRIPTIONS = {
    "RS": "A remote sensing expert, adept at analyzing aerial or satellite images. This model excels at object counting, presence detection, and area estimation.",
    "Med": "A medical imaging expert, primarily focused on pathology, including cell sections and natural images of medical conditions.",
    "AD": "An autonomous driving expert specializing in ego-view scene understanding, including coordinate prediction and action planning and other driving-related tasks. The input image is an image concatenated by 6 camera views.",
    "Sci": "A science expert with proficiency in biology, map interpretation, physics, and chemistry.",
    "Fin": "A financial expert specializing in stock market analysis using candlestick charts. This model excels at trend prediction and technical indicator analysis.",
}

ACL_EXPERT_DESCRIPTIONS = {
    "OCR": "This model excels in OCR tasks, including text extraction, handwriting recognition, and document analysis.",
    "Math": "This model is an expert in math and logic, including solving equations, geometry, and logical reasoning. It is capable of on puzzle test figures, algebraic reasoning over functional plots, and scientific reasoning with academic paper figures.",
    "VP": "This model excels in counting the number of objects in the image. However, it struggles to exact text in an image.",
    "APP": "This model is an expert in GUI navigation, including identifying buttons, text fields, and other UI elements from screen shots. It is capable of giving coordinates of the elements in the image and conduct action on the elements.",
}

ACL_MERGED_DESCRIPTIONS = {
    "OCR+Math": "This model is an expert in both OCR (Optical Character Recognition) and Math & Logic. It excels at tasks that require first recognizing text, numbers, and symbols in an image and then applying mathematical or logical reasoning to solve the problem. Its capabilities include text extraction, handwriting recognition, solving equations, geometry, and analyzing charts or figures.",
}


class PromptManager:
    """Manages router prompts dynamically based on expert expansion/reuse decisions."""

    def __init__(self, setting="DCL"):
        self.setting = setting
        self.experts = []
        self.expert_to_letter = {}
        self.letter_labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        if setting == "DCL":
            self.all_descriptions = deepcopy(DCL_EXPERT_DESCRIPTIONS)
        else:
            self.all_descriptions = deepcopy(ACL_EXPERT_DESCRIPTIONS)

    def add_expert(self, task_name, description=None, reuse_expert=None):
        """Add a new expert or mark a task as reusing an existing expert.

        Args:
            task_name: Name of the new task.
            description: Custom description. If None, uses default.
            reuse_expert: If not None, the name of the existing expert to reuse.
        """
        if reuse_expert is not None:
            existing_idx = None
            for i, e in enumerate(self.experts):
                if reuse_expert in e["tasks"]:
                    existing_idx = i
                    break
            if existing_idx is not None:
                self.experts[existing_idx]["tasks"].append(task_name)
                merged_key = "+".join(self.experts[existing_idx]["tasks"])
                if self.setting == "ACL" and merged_key in ACL_MERGED_DESCRIPTIONS:
                    self.experts[existing_idx]["description"] = ACL_MERGED_DESCRIPTIONS[merged_key]
                else:
                    old_desc = self.experts[existing_idx]["description"]
                    new_desc = description or self.all_descriptions.get(task_name, "")
                    self.experts[existing_idx]["description"] = (
                        f"{old_desc} It also handles {task_name} tasks: {new_desc}"
                    )
                self.expert_to_letter[task_name] = self.experts[existing_idx]["letter"]
                return self.experts[existing_idx]["letter"]

        letter = self.letter_labels[len(self.experts)]
        desc = description or self.all_descriptions.get(task_name, f"Expert for {task_name}")
        expert_entry = {
            "letter": letter,
            "tasks": [task_name],
            "description": desc,
        }
        self.experts.append(expert_entry)
        self.expert_to_letter[task_name] = letter
        return letter

    def get_routing_prompt(self):
        """Generate the current routing prompt based on active experts."""
        num_experts = len(self.experts)
        letters = [e["letter"] for e in self.experts]
        letter_str = ",".join(letters)

        if self.setting == "DCL":
            domain_list = ", ".join(
                t for e in self.experts for t in e["tasks"]
            )
            header = (
                f"You are a helpful assistant router. There are {num_experts} expert models, "
                f"each specializing in one of the following domains: {domain_list}.\n\n"
                f"Your task is to select the most suitable model based on the provided visual content, "
                f"user question, and model descriptions. Consider the expertise of each model carefully "
                f"and select the one best equipped to handle the given question.\n\n"
                f"**Important Instructions:**\n"
                f"- Respond **only** with the letter ({letter_str}) corresponding to the most suitable model.\n"
                f"- Do **not** attempt to answer the user's question directly.\n\n"
                f"**Model Pool:**\n"
            )
        else:
            ability_list = ", ".join(
                t for e in self.experts for t in e["tasks"]
            )
            header = (
                f"You are a helpful assistant router. There are {num_experts} expert models, "
                f"each specializing in one of the following domains: {ability_list}.\n"
                f"Your task is to select the most suitable model based on the provided visual content, "
                f"user question, and model descriptions. Consider the expertise of each model carefully "
                f"and select the one best equipped to handle the given question.\n\n"
                f"**Important Instructions:**\n"
                f"- Respond **only** with the letter ({letter_str}) corresponding to the most suitable model.\n"
                f"- Do **not** attempt to answer the user's question directly.\n\n"
                f"**Model Pool:**\n"
            )

        for expert in self.experts:
            header += f"- **{expert['letter']}**: {expert['description']}\n"

        header += "\nHere is the user's question: "
        return header

    def get_prompt_after_question(self):
        return "You only need to select the suitable model and do not answer the question. JUST answer with the model's letter from the given choices directly."

    def get_expert_letter(self, task_name):
        return self.expert_to_letter.get(task_name)

    def get_letter_to_expert_map(self):
        """Return mapping from letter to expert index (for choosing answers)."""
        return {e["letter"]: i for i, e in enumerate(self.experts)}

    def get_num_experts(self):
        return len(self.experts)

    def save(self, path):
        state = {
            "setting": self.setting,
            "experts": self.experts,
            "expert_to_letter": self.expert_to_letter,
        }
        with open(path, "w") as f:
            json.dump(state, f, indent=2)

    def load(self, path):
        with open(path, "r") as f:
            state = json.load(f)
        self.setting = state["setting"]
        self.experts = state["experts"]
        self.expert_to_letter = state["expert_to_letter"]
