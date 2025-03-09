# lora train sequential training
# bash scripts/LLaVA/Train_dom/1_RemoteSensing.sh  Finetune-CL 
# bash scripts/LLaVA/Train_dom/2_Medical.sh Finetune-CL True
# bash scripts/LLaVA/Train_dom/3_DriveLM.sh Finetune-CL True
# bash scripts/LLaVA/Train_dom/4_Sci.sh Finetune-CL True

# # lora train sequential training MoE (8)
# bash scripts/LLaVA/Train_dom/1_RemoteSensing.sh  Finetune-CL-MoE "" 8
bash scripts/LLaVA/Train_dom/2_Medical.sh  Finetune-CL-MoE True 8
bash scripts/LLaVA/Train_dom/3_DriveLM.sh  Finetune-CL-MoE True 8
bash scripts/LLaVA/Train_dom/4_Sci.sh Finetune-CL-MoE True 8

# # # lora train sequential training test (use last to test)
# bash scripts/LLaVA/Eval_dom/1_RemoteSensing.sh  Finetune "./checkpoints/LLaVA/Finetune-CL/Sci_llava_lora" Sci_rs FineTune-CL-Test
# bash scripts/LLaVA/Eval_dom/2_Medical.sh Finetune "./checkpoints/LLaVA/Finetune-CL/Sci_llava_lora" Sci_med FineTune-CL-Test
# bash scripts/LLaVA/Eval_dom/3_DriveLM.sh Finetune "./checkpoints/LLaVA/Finetune-CL/Sci_llava_lora" Sci_drivelm FineTune-CL-Test
# bash scripts/LLaVA/Eval_dom/4_Sci.sh Finetune "./checkpoints/LLaVA/Finetune-CL/Sci_llava_lora" Sci_sci FineTune-CL-Test

# # # lora train sequential training test (use last to test) MoE (8)
# bash scripts/LLaVA/Eval_dom/1_RemoteSensing.sh  Finetune "./checkpoints/LLaVA/Finetune-CL-MoE/Sci_llava_lora" Sci_rs FineTune-CL-MoE-Test
# bash scripts/LLaVA/Eval_dom/2_Medical.sh Finetune "./checkpoints/LLaVA/Finetune-CL-MoE/Sci_llava_lora" Sci_med FineTune-CL-MoE-Test
# bash scripts/LLaVA/Eval_dom/3_DriveLM.sh Finetune "./checkpoints/LLaVA/Finetune-CL-MoE/Sci_llava_lora" Sci_drivelm FineTune-CL-MoE-Test
# bash scripts/LLaVA/Eval_dom/4_Sci.sh Finetune "./checkpoints/LLaVA/Finetune-CL-MoE/Sci_llava_lora" Sci_sci FineTune-CL-MoE-Test

# # llava1.5 test
# bash scripts/LLaVA/Eval_dom/14_eval_remotesensing.sh "" ./checkpoints/LLaVA/Vicuna/llava-7b-v1.5 llava_rs
# bash scripts/LLaVA/Eval_dom/15_eval_drivelm.sh "" ./checkpoints/LLaVA/Vicuna/llava-7b-v1.5 llava_drivelm
# bash scripts/LLaVA/Eval_dom/12_eval_sci.sh "" ./checkpoints/LLaVA/Vicuna/llava-7b-v1.5 llava_sci

# # lora test (no CL)
# bash scripts/LLaVA/Eval_dom/14_eval_remotesensing.sh ""
# bash scripts/LLaVA/Eval_dom/15_eval_drivelm.sh ""
# bash scripts/LLaVA/Eval_dom/11_eval_medical.sh ""
# bash scripts/LLaVA/Eval_dom/12_eval_sci.sh ""
