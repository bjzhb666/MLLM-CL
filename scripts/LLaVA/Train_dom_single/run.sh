# lora train signle
# bash scripts/LLaVA/Train_dom_single/11_Medical.sh CoINep3
# bash scripts/LLaVA/Train_dom_single/12_Sci.sh 
# bash scripts/LLaVA/Train_dom_single/13_RemoteSensing.sh 
# bash scripts/LLaVA/Train_dom_single/14_DriveLM.sh 

# # llava1.5 test
# bash scripts/LLaVA/Eval_dom/14_eval_remotesensing.sh "" ./checkpoints/LLaVA/Vicuna/llava-7b-v1.5 llava_rs
# bash scripts/LLaVA/Eval_dom/15_eval_drivelm.sh "" ./checkpoints/LLaVA/Vicuna/llava-7b-v1.5 llava_drivelm
# bash scripts/LLaVA/Eval_dom/12_eval_sci.sh "" ./checkpoints/LLaVA/Vicuna/llava-7b-v1.5 llava_sci

# lora test
# bash scripts/LLaVA/Eval_dom/14_eval_remotesensing.sh ""
# bash scripts/LLaVA/Eval_dom/15_eval_drivelm.sh ""
# bash scripts/LLaVA/Eval_dom/11_eval_medical.sh ""
# bash scripts/LLaVA/Eval_dom/11_eval_medical.sh Finetune2ep ./checkpoints/LLaVA/CoINep3/Medical_llava_lora

# vision lora test,  not available
# bash scripts/LLaVA/Eval_dom/14_eval_remotesensing.sh ""  ./checkpoints/LLaVA/CoIN/RemoteSensing_llava_lora "" LlavaVision
# bash scripts/LLaVA/Eval_dom/15_eval_drivelm.sh ""  ./checkpoints/LLaVA/CoIN/DriveLM_llava_lora "" LlavaVision
# bash scripts/LLaVA/Eval_dom/11_eval_medical.sh ""  ./checkpoints/LLaVA/CoIN/Medical_llava_lora "" LlavaVision
# bash scripts/LLaVA/Eval_dom/12_eval_sci.sh ""  ./checkpoints/LLaVA/CoIN/Sci_llava_lora2e-5 "" LlavaVision

# baseline 顺序finetune
# bash scripts/LLaVA/Eval_dom/11_eval_medical.sh "" ./checkpoints/LLaVA/CoIN/Medical_llava_lora "" 

# Train LoRA SCI and Finvis
bash scripts/LLaVA/Train_dom/3_DriveLM.sh "" True
bash scripts/LLaVA/Train_dom/4_Sci.sh "" True
bash scripts/LLaVA/Train_dom/5_Financial.sh "" True

bash scripts/LLaVA/Train_dom/3_DriveLM.sh Finetune-CL-MoE True 8
bash scripts/LLaVA/Train_dom/4_Sci.sh Finetune-CL-MoE True 8
bash scripts/LLaVA/Train_dom/5_Financial.sh Finetune-CL-MoE True 8

# 测试sci
bash scripts/LLaVA/Eval_dom/4_eval_sci.sh "" ./checkpoints/LLaVA/CoIN/Sci_llava_lora_ep2_new Sci_sci
bash scripts/LLaVA/Eval_dom/4_eval_sci.sh "" ./checkpoints/LLaVA/CoIN/Medical_llava_lora PathVQA_sci
bash scripts/LLaVA/Eval_dom/4_eval_sci.sh "" ./checkpoints/LLaVA/CoIN/RemoteSensing_llava_lora RS_sci
bash scripts/LLaVA/Eval_dom/4_eval_sci.sh "" ./checkpoints/LLaVA/CoIN/DriveLM_llava_lora DriveLM_sci
bash scripts/LLaVA/Eval_dom/4_eval_sci.sh "" ./checkpoints/LLaVA/CoIN/FinVis_llava_lora Fin_sci
bash scripts/LLaVA/Eval_dom/1_eval_remotesensing.sh "" ./checkpoints/LLaVA/CoIN/Sci_llava_lora_ep2_new Sci_rs
bash scripts/LLaVA/Eval_dom/2_eval_medical.sh "" ./checkpoints/LLaVA/CoIN/Sci_llava_lora_ep2_new Sci_pathvqa
bash scripts/LLaVA/Eval_dom/3_eval_drivelm.sh "" ./checkpoints/LLaVA/CoIN/Sci_llava_lora_ep2_new Sci_drivelm
bash scripts/LLaVA/Eval_dom/5_eval_financial.sh "" ./checkpoints/LLaVA/CoIN/Sci_llava_lora_ep2_new Sci_fin

# 测试Finetune-CL
bash scripts/LLaVA/Eval_dom/4_eval_sci.sh Finetune "./checkpoints/LLaVA/Finetune-CL/FinVis_llava_lora" Fin_sci FineTune-CL-Test
bash scripts/LLaVA/Eval_dom/1_eval_remotesensing.sh Finetune "./checkpoints/LLaVA/Finetune-CL/FinVis_llava_lora" Fin_rs FineTune-CL-Test
bash scripts/LLaVA/Eval_dom/2_eval_medical.sh Finetune "./checkpoints/LLaVA/Finetune-CL/FinVis_llava_lora" Fin_pathvqa FineTune-CL-Test
bash scripts/LLaVA/Eval_dom/3_eval_drivelm.sh Finetune "./checkpoints/LLaVA/Finetune-CL/FinVis_llava_lora" Fin_drivelm FineTune-CL-Test
bash scripts/LLaVA/Eval_dom/5_eval_financial.sh Finetune "./checkpoints/LLaVA/Finetune-CL/FinVis_llava_lora" Fin_fin FineTune-CL-Test

bash scripts/LLaVA/Eval_dom/4_eval_sci.sh Finetune "./checkpoints/LLaVA/Finetune-CL/Sci_llava_lora" Sci_sci FineTune-CL-Test
bash scripts/LLaVA/Eval_dom/1_eval_remotesensing.sh Finetune "./checkpoints/LLaVA/Finetune-CL/Sci_llava_lora" Sci_rs FineTune-CL-Test
bash scripts/LLaVA/Eval_dom/2_eval_medical.sh Finetune "./checkpoints/LLaVA/Finetune-CL/Sci_llava_lora" Sci_pathvqa FineTune-CL-Test
bash scripts/LLaVA/Eval_dom/3_eval_drivelm.sh Finetune "./checkpoints/LLaVA/Finetune-CL/Sci_llava_lora" Sci_drivelm FineTune-CL-Test

# 测试Finetune-CL-MoE
bash scripts/LLaVA/Eval_dom/4_eval_sci.sh Finetune "./checkpoints/LLaVA/Finetune-CL-MoE/FinVis_llava_lora_MOE" Fin_sci FineTune-CL-MoE-Test
bash scripts/LLaVA/Eval_dom/1_eval_remotesensing.sh Finetune "./checkpoints/LLaVA/Finetune-CL-MoE/FinVis_llava_lora_MOE" Fin_rs FineTune-CL-MoE-Test
bash scripts/LLaVA/Eval_dom/2_eval_medical.sh Finetune "./checkpoints/LLaVA/Finetune-CL-MoE/FinVis_llava_lora_MOE" Fin_pathvqa FineTune-CL-MoE-Test
bash scripts/LLaVA/Eval_dom/3_eval_drivelm.sh Finetune "./checkpoints/LLaVA/Finetune-CL-MoE/FinVis_llava_lora_MOE" Fin_drivelm FineTune-CL-MoE-Test
bash scripts/LLaVA/Eval_dom/5_eval_financial.sh Finetune "./checkpoints/LLaVA/Finetune-CL-MoE/FinVis_llava_lora_MOE" Fin_fin FineTune-CL-MoE-Test

bash scripts/LLaVA/Eval_dom/4_eval_sci.sh Finetune "./checkpoints/LLaVA/Finetune-CL-MoE/Sci_llava_lora_MOE" Sci_sci FineTune-CL-MoE-Test
bash scripts/LLaVA/Eval_dom/1_eval_remotesensing.sh Finetune "./checkpoints/LLaVA/Finetune-CL-MoE/Sci_llava_lora_MOE" Sci_rs FineTune-CL-MoE-Test
bash scripts/LLaVA/Eval_dom/2_eval_medical.sh Finetune "./checkpoints/LLaVA/Finetune-CL-MoE/Sci_llava_lora_MOE" Sci_pathvqa FineTune-CL-MoE-Test
bash scripts/LLaVA/Eval_dom/3_eval_drivelm.sh Finetune "./checkpoints/LLaVA/Finetune-CL-MoE/Sci_llava_lora_MOE" Sci_drivelm FineTune-CL-MoE-Test