"""
AdapteX: Main orchestration script for the full pipeline.

For each new task in the continual learning sequence:
1. Prepare support/query sets from task data
2. Train router on combined support sets (if not first task)
3. Probe affinity between new task and existing experts
4. Decide: expand (create new expert) or reuse (with EWC)
5. Train the expert LoRA
6. Save few-shot memory for future EWC
7. Update the prompt manager
8. Evaluate on all seen tasks

Usage:
    python adaptex/run_adaptex.py --config adaptex/configs/adaptex_dcl_llava.json
"""

import argparse
import json
import os
import subprocess
import sys
import time
from copy import deepcopy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adaptex.affinity_probe import AffinityProbe
from adaptex.ewc import save_fisher, compute_fisher_information, get_lora_state_dict
from adaptex.prompt_manager import PromptManager
from adaptex.router_data import (
    prepare_support_query_split,
    create_router_training_data,
    save_query_set,
    load_query_set,
    save_few_shot_memory,
)


def load_config(config_path):
    with open(config_path, "r") as f:
        return json.load(f)


def run_deepspeed_training(
    script_path,
    model_name,
    mm_projector,
    vision_tower,
    data_path,
    image_folder,
    output_dir,
    lora_rank=32,
    lr=1e-4,
    epochs=1,
    batch_size=4,
    grad_acc=2,
    gpu_num=8,
    extra_args=None,
    previous_model=None,
):
    """Launch a deepspeed training job."""
    gpu_list = ",".join(str(i) for i in range(gpu_num))

    cmd = [
        "deepspeed",
        "--include", f"localhost:{gpu_list}",
        "--master_port", "9001",
        script_path,
        "--deepspeed", "./scripts/zero2.json",
        "--lora_enable", "True",
        "--lora_r", str(lora_rank),
        "--lora_alpha", str(lora_rank * 2),
        "--mm_projector_lr", "2e-5",
        "--model_name_or_path", model_name,
        "--pretrain_mm_mlp_adapter", mm_projector,
        "--version", "v1",
        "--data_path", data_path,
        "--image_folder", image_folder,
        "--vision_tower", vision_tower,
        "--mm_projector_type", "mlp2x_gelu",
        "--mm_vision_select_layer", "-2",
        "--mm_use_im_start_end", "False",
        "--mm_use_im_patch_token", "False",
        "--image_aspect_ratio", "pad",
        "--group_by_modality_length", "True",
        "--bf16", "True",
        "--output_dir", output_dir,
        "--num_train_epochs", str(epochs),
        "--per_device_train_batch_size", str(batch_size),
        "--per_device_eval_batch_size", "16",
        "--gradient_accumulation_steps", str(grad_acc),
        "--evaluation_strategy", "no",
        "--save_strategy", "steps",
        "--save_steps", "50000",
        "--learning_rate", str(lr),
        "--weight_decay", "0.",
        "--warmup_ratio", "0.03",
        "--lr_scheduler_type", "cosine",
        "--logging_steps", "1",
        "--tf32", "True",
        "--model_max_length", "2048",
        "--gradient_checkpointing", "True",
        "--dataloader_num_workers", "4",
        "--lazy_preprocess", "True",
        "--report_to", "none",
    ]

    if previous_model:
        cmd += ["--previous_task_model_path", previous_model]

    if extra_args:
        for k, v in extra_args.items():
            cmd += [f"--{k}", str(v)]

    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd[:10])}...")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, check=True)
    return result.returncode == 0


def run_adaptex_pipeline(config):
    """Run the full AdapteX pipeline."""
    setting = config["setting"]
    tasks = config["tasks"]
    base_model = config["base_model"]
    mm_projector = config["mm_projector"]
    vision_tower = config["vision_tower"]
    data_base_dir = config["data_base_dir"]
    checkpoint_base = config["checkpoint_base"]
    gpu_num = config.get("gpu_num", 8)
    lora_rank = config.get("lora_rank", 32)
    affinity_threshold = config.get("affinity_threshold", 0.2)
    ewc_lambda = config.get("ewc_lambda", 0.5)
    router_lr = config.get("router_lr", 2e-5)
    router_epochs = config.get("router_epochs", 30)
    max_query_samples = config.get("max_query_samples", 200)

    prompt_manager = PromptManager(setting=setting)

    # State tracking
    expert_checkpoints = {}
    task_support_sets = {}
    task_query_sets = {}
    expert_assignments = {}
    task_to_expert_idx = {}
    expert_few_shot_paths = {}
    expert_fisher_paths = {}

    state_dir = os.path.join(checkpoint_base, "adaptex_state")
    os.makedirs(state_dir, exist_ok=True)

    results_log = []

    for task_idx, task_config in enumerate(tasks):
        task_name = task_config["name"]
        train_path = task_config["train_path"]
        image_folder = task_config["image_folder"]
        expert_lr = task_config.get("lr", 1e-4)
        expert_epochs = task_config.get("epochs", 1)
        batch_size = task_config.get("batch_size", 4)

        print(f"\n{'#'*60}")
        print(f"# Task {task_idx + 1}/{len(tasks)}: {task_name}")
        print(f"{'#'*60}")

        # ── Step 1: Prepare support/query split ──
        print(f"\n[Step 1] Preparing support/query split for {task_name}...")
        support, query = prepare_support_query_split(train_path, support_ratio=0.8)
        task_support_sets[task_name] = support
        task_query_sets[task_name] = query

        query_path = os.path.join(state_dir, f"query_{task_name}.json")
        save_query_set(query, query_path)

        few_shot_path = os.path.join(state_dir, f"memory_{task_name}.json")
        save_few_shot_memory(support, few_shot_path, num_samples=200)

        if task_idx == 0:
            # ── First task: always expand ──
            print(f"\n[Step 2] First task - creating expert E_0 for {task_name}")
            decision = "expand"
            reuse_task = None
            letter = prompt_manager.add_expert(task_name)
            expert_assignments[task_name] = letter
            task_to_expert_idx[task_name] = 0
        else:
            # ── Step 2: Train router on combined support sets ──
            print(f"\n[Step 2] Training router on combined support sets...")

            new_letter = chr(ord("A") + prompt_manager.get_num_experts())
            temp_expert_assignments = deepcopy(expert_assignments)
            temp_expert_assignments[task_name] = new_letter

            router_data_path = os.path.join(state_dir, f"router_train_task{task_idx + 1}.json")
            all_task_names = list(expert_assignments.keys()) + [task_name]
            temp_support = deepcopy(task_support_sets)

            create_router_training_data(
                temp_support,
                all_task_names,
                temp_expert_assignments,
                router_data_path,
                data_base_dir,
                setting=setting,
            )

            prev_router = None
            if task_idx >= 2:
                prev_router = os.path.join(
                    checkpoint_base, f"router_task{task_idx}"
                )

            router_output = os.path.join(checkpoint_base, f"router_task{task_idx + 1}")

            run_deepspeed_training(
                script_path="llava/train/train_mem.py",
                model_name=base_model,
                mm_projector=mm_projector,
                vision_tower=vision_tower,
                data_path=router_data_path,
                image_folder=data_base_dir,
                output_dir=router_output,
                lora_rank=lora_rank,
                lr=router_lr,
                epochs=router_epochs,
                batch_size=batch_size,
                grad_acc=2,
                gpu_num=gpu_num,
                previous_model=prev_router,
            )

            # ── Step 3: Probe affinity ──
            print(f"\n[Step 3] Probing affinity for {task_name}...")
            probe = AffinityProbe(
                router_model_path=router_output,
                model_base=base_model,
            )
            probe.load_router()

            existing_queries = {}
            for prev_task in expert_assignments:
                prev_query_path = os.path.join(state_dir, f"query_{prev_task}.json")
                if os.path.exists(prev_query_path):
                    existing_queries[prev_task] = load_query_set(prev_query_path)

            routing_prompt = prompt_manager.get_routing_prompt()
            prompt_after = prompt_manager.get_prompt_after_question()

            # Temporarily add the new expert for affinity computation
            temp_pm = deepcopy(prompt_manager)
            temp_pm.add_expert(task_name)
            temp_routing_prompt = temp_pm.get_routing_prompt()

            affinity_scores = probe.compute_affinity_scores(
                new_task_query=query,
                existing_query_sets=existing_queries,
                existing_expert_letters=expert_assignments,
                new_expert_letter=new_letter,
                routing_prompt=temp_routing_prompt,
                prompt_after=prompt_after,
                image_base_dir=data_base_dir,
                new_task_image_dir=image_folder,
                max_query_samples=max_query_samples,
            )

            probe.release_model()

            # ── Step 4: Decide expand or reuse ──
            decision, reuse_task = probe.decide_expansion(
                affinity_scores, threshold=affinity_threshold
            )

            print(f"\n[Step 4] Decision: {decision.upper()}")
            if decision == "reuse":
                print(f"  Reusing expert from task: {reuse_task}")
                letter = prompt_manager.add_expert(
                    task_name, reuse_expert=reuse_task
                )
            else:
                print(f"  Creating new expert for: {task_name}")
                letter = prompt_manager.add_expert(task_name)

            expert_assignments[task_name] = letter

        # ── Step 5: Train the expert ──
        print(f"\n[Step 5] Training expert for {task_name} (decision={decision})...")
        expert_output = os.path.join(checkpoint_base, f"expert_{task_name}")

        if decision == "reuse" and reuse_task is not None:
            reuse_checkpoint = expert_checkpoints[reuse_task]

            fisher_path = os.path.join(state_dir, f"fisher_{reuse_task}.pt")

            run_deepspeed_training(
                script_path="adaptex/train_ewc.py",
                model_name=base_model,
                mm_projector=mm_projector,
                vision_tower=vision_tower,
                data_path=train_path,
                image_folder=image_folder,
                output_dir=expert_output,
                lora_rank=lora_rank,
                lr=expert_lr,
                epochs=expert_epochs,
                batch_size=batch_size,
                grad_acc=2,
                gpu_num=gpu_num,
                extra_args={
                    "ewc_enabled": "True",
                    "ewc_lambda": str(ewc_lambda),
                    "reuse_lora_path": reuse_checkpoint,
                    "fisher_path": fisher_path,
                    "compute_fisher_after": "True",
                },
            )

            expert_checkpoints[reuse_task] = expert_output
            expert_checkpoints[task_name] = expert_output
        else:
            run_deepspeed_training(
                script_path="adaptex/train_ewc.py",
                model_name=base_model,
                mm_projector=mm_projector,
                vision_tower=vision_tower,
                data_path=train_path,
                image_folder=image_folder,
                output_dir=expert_output,
                lora_rank=lora_rank,
                lr=expert_lr,
                epochs=expert_epochs,
                batch_size=batch_size,
                grad_acc=2,
                gpu_num=gpu_num,
                extra_args={
                    "compute_fisher_after": "True",
                    "fisher_path": os.path.join(state_dir, f"fisher_{task_name}.pt"),
                },
            )
            expert_checkpoints[task_name] = expert_output

        # ── Step 6: Record state ──
        task_result = {
            "task_idx": task_idx,
            "task_name": task_name,
            "decision": decision,
            "reuse_task": reuse_task,
            "expert_letter": letter,
            "expert_checkpoint": expert_output,
            "affinity_scores": affinity_scores if task_idx > 0 else {},
            "num_experts": prompt_manager.get_num_experts(),
        }
        results_log.append(task_result)

        # Save state
        prompt_manager.save(os.path.join(state_dir, "prompt_manager.json"))
        with open(os.path.join(state_dir, "results_log.json"), "w") as f:
            json.dump(results_log, f, indent=2)
        with open(os.path.join(state_dir, "expert_checkpoints.json"), "w") as f:
            json.dump(expert_checkpoints, f, indent=2)

        print(f"\n[Done] Task {task_name}: {decision} | Expert={letter} | Total experts={prompt_manager.get_num_experts()}")

    # ── Final: Train final router ──
    print(f"\n{'='*60}")
    print("Training final router on all tasks...")
    final_router_data = os.path.join(state_dir, "router_train_final.json")
    create_router_training_data(
        task_support_sets,
        list(expert_assignments.keys()),
        expert_assignments,
        final_router_data,
        data_base_dir,
        setting=setting,
    )

    prev_router = None
    if len(tasks) >= 2:
        prev_router = os.path.join(checkpoint_base, f"router_task{len(tasks)}")

    final_router_output = os.path.join(checkpoint_base, "router_final")
    run_deepspeed_training(
        script_path="llava/train/train_mem.py",
        model_name=base_model,
        mm_projector=mm_projector,
        vision_tower=vision_tower,
        data_path=final_router_data,
        image_folder=data_base_dir,
        output_dir=final_router_output,
        lora_rank=lora_rank,
        lr=router_lr,
        epochs=router_epochs,
        batch_size=batch_size,
        grad_acc=2,
        gpu_num=gpu_num,
        previous_model=prev_router,
    )

    # ── Summary ──
    print(f"\n{'='*60}")
    print("AdapteX Pipeline Complete!")
    print(f"{'='*60}")
    print(f"Setting: {setting}")
    print(f"Total tasks: {len(tasks)}")
    print(f"Total experts: {prompt_manager.get_num_experts()}")
    print(f"Final router: {final_router_output}")
    print(f"\nExpansion decisions:")
    for r in results_log:
        suffix = f" (reused {r['reuse_task']})" if r["reuse_task"] else ""
        print(f"  {r['task_name']}: {r['decision']}{suffix} → Expert {r['expert_letter']}")
    print(f"\nFinal routing prompt:\n{prompt_manager.get_routing_prompt()}")

    final_state = {
        "setting": setting,
        "results_log": results_log,
        "expert_checkpoints": expert_checkpoints,
        "final_router": final_router_output,
        "routing_prompt": prompt_manager.get_routing_prompt(),
        "num_experts": prompt_manager.get_num_experts(),
    }
    with open(os.path.join(state_dir, "final_state.json"), "w") as f:
        json.dump(final_state, f, indent=2)

    return final_state


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AdapteX: Adaptive Expert Expansion Pipeline")
    parser.add_argument("--config", type=str, required=True, help="Path to AdapteX config JSON")
    args = parser.parse_args()

    config = load_config(args.config)
    run_adaptex_pipeline(config)
