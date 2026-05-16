#!/bin/bash
# AdapteX: Run the full ACL pipeline (LLaVA-based)
#
# Before running, edit adaptex/configs/adaptex_acl_llava.json to set:
#   - base_model: path to llava-v1.5-7b
#   - mm_projector: path to mm_projector.bin
#   - vision_tower: path to CLIP ViT
#   - data_base_dir: path to Ability_data
#   - task train_path / image_folder for each task
#
# Usage: bash adaptex/scripts/run_acl.sh

set -e

pip install -q -e .

python adaptex/run_adaptex.py --config adaptex/configs/adaptex_acl_llava.json

echo ""
echo "ACL pipeline complete. Check checkpoints/AdapteX-ACL-LLaVA/adaptex_state/ for results."
