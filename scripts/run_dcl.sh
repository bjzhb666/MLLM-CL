#!/bin/bash
# AdapteX: Run the full DCL pipeline (LLaVA-based)
#
# Before running, edit adaptex/configs/adaptex_dcl_llava.json to set:
#   - base_model: path to llava-v1.5-7b
#   - mm_projector: path to mm_projector.bin
#   - vision_tower: path to CLIP ViT
#   - data_base_dir: path to Domain_data
#   - task train_path / image_folder for each task
#
# Usage: bash adaptex/scripts/run_dcl.sh

set -e

pip install -q -e .

python adaptex/run_adaptex.py --config adaptex/configs/adaptex_dcl_llava.json

echo ""
echo "DCL pipeline complete. Check checkpoints/AdapteX-DCL-LLaVA/adaptex_state/ for results."
