import shutil
import torch
import os

# List of checkpoint directories
# names = "ScienceQA TextVQA ImageNet GQA VizWiz Grounding VQAv2 OCRVQA"
names = "VizWiz ImageNet ScienceQA TextVQA grounding VQAv2"
checkpoint_dirs = [
    f"{x}_llava_lora_ours" for x in names.split(' ')
]
src_dir = 'checkpoints/LLaVA/FAE/'
dst_dir = 'checkpoints/LLaVA/AvgLora_3/'
# Load the weights from each checkpoint
weights_list = []
for checkpoint_dir in checkpoint_dirs:
    base_dir = os.path.join(src_dir, checkpoint_dir)
    model_path = os.path.join(base_dir, 'adapter_model.bin')
    weights = torch.load(model_path)
    weights_list.append(weights)
    target_path = os.path.join(dst_dir, checkpoint_dir)
    os.makedirs(target_path, exist_ok=True)
    for item in os.listdir(base_dir):
        s = os.path.join(src_dir, checkpoint_dir, item)
        if not os.path.isfile(s):
            continue
        shutil.copy2(s, os.path.join(target_path, item))
    

# Compute the average of the weights
for i in range(1, len(weights_list)):
    comp_weights = weights_list[:i]
    # average_weights = comp_weights[-1]
    average_weights = comp_weights[0]
    for key in comp_weights[0].keys():
        # print(comp_weights[0])
        # exit(0)
        for weight in comp_weights:
            average_weights[key] = weight[key] * 0.5 + average_weights[key] * 0.5
        # average_weights[key] = sum(weights[key] for weights in comp_weights) / len(comp_weights)
        # print(MSE(comp_weights[0][key], average_weights[0][key]))
        # average_weights[key] = comp_weights[i][key] * 0.5

    # Save the averaged weights to a new file
    output_dir = os.path.join(dst_dir, checkpoint_dirs[i])
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'adapter_model.bin')
    torch.save(average_weights, output_path)

    print(f"Averaged weights saved to {output_path}")