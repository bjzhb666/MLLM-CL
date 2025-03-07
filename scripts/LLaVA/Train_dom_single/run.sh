# lora train signle
bash scripts/LLaVA/Train_dom_single/11_Medical.sh
bash scripts/LLaVA/Train_dom_single/12_Sci.sh 
bash scripts/LLaVA/Train_dom_single/13_RemoteSensing.sh 
bash scripts/LLaVA/Train_dom_single/14_DriveLM.sh 

# # llava1.5 test
# bash scripts/LLaVA/Eval_dom/14_eval_remotesensing.sh "" ./checkpoints/LLaVA/Vicuna/llava-7b-v1.5 llava_rs
# bash scripts/LLaVA/Eval_dom/15_eval_drivelm.sh "" ./checkpoints/LLaVA/Vicuna/llava-7b-v1.5 llava_drivelm
# bash scripts/LLaVA/Eval_dom/12_eval_sci.sh "" ./checkpoints/LLaVA/Vicuna/llava-7b-v1.5 llava_sci

# lora test
bash scripts/LLaVA/Eval_dom/14_eval_remotesensing.sh ""
bash scripts/LLaVA/Eval_dom/15_eval_drivelm.sh ""
bash scripts/LLaVA/Eval_dom/11_eval_medical.sh ""
bash scripts/LLaVA/Eval_dom/12_eval_sci.sh ""

# vision lora test
# bash scripts/LLaVA/Eval_dom/14_eval_remotesensing.sh ""  ./checkpoints/LLaVA/CoIN/RemoteSensing_llava_lora "" LlavaVision
# bash scripts/LLaVA/Eval_dom/15_eval_drivelm.sh ""  ./checkpoints/LLaVA/CoIN/DriveLM_llava_lora "" LlavaVision
# bash scripts/LLaVA/Eval_dom/11_eval_medical.sh ""  ./checkpoints/LLaVA/CoIN/Medical_llava_lora "" LlavaVision
# bash scripts/LLaVA/Eval_dom/12_eval_sci.sh ""  ./checkpoints/LLaVA/CoIN/Sci_llava_lora2e-5 "" LlavaVision