# lora train sequential training
# bash scripts/LLaVA/Train_dom/1_RemoteSensing.sh  Finetune-CL 
# bash scripts/LLaVA/Train_dom/2_Medical.sh Finetune-CL True
# bash scripts/LLaVA/Train_dom/3_DriveLM.sh Finetune-CL True
# bash scripts/LLaVA/Train_dom/4_Sci.sh Finetune-CL True
# bash scripts/LLaVA/Train_dom/5_Financial.sh Finetune-CL True
# bash scripts/LLaVA/Train_dom/6_Fin_replay.sh Finetune-CL True

# # # lora train sequential training MoE (8)
# bash scripts/LLaVA/Train_dom/1_RemoteSensing.sh  Finetune-CL-MoE "" 8
# bash scripts/LLaVA/Train_dom/2_Medical.sh  Finetune-CL-MoE True 8
# bash scripts/LLaVA/Train_dom/3_DriveLM.sh  Finetune-CL-MoE True 8
# bash scripts/LLaVA/Train_dom/4_Sci.sh Finetune-CL-MoE True 8
# bash scripts/LLaVA/Train_dom/5_Financial.sh Finetune-CL-MoE True 8
# bash scripts/LLaVA/Train_dom/6_Fin_replay.sh Finetune-CL-MoE True 8

# # # lora train sequential training test (use last to test)
# bash scripts/LLaVA/Eval_dom/1_eval_remotesensing.sh  Finetune "./checkpoints/LLaVA/Finetune-CL/FinVis_llava_lora" Fin_rs FineTune-CL-Test
# bash scripts/LLaVA/Eval_dom/2_eval_medical.sh Finetune "./checkpoints/LLaVA/Finetune-CL/FinVis_llava_lora" Fin_med FineTune-CL-Test
# bash scripts/LLaVA/Eval_dom/3_eval_drivelm.sh Finetune "./checkpoints/LLaVA/Finetune-CL/FinVis_llava_lora" Fin_drivelm FineTune-CL-Test
# bash scripts/LLaVA/Eval_dom/4_eval_sci.sh Finetune "./checkpoints/LLaVA/Finetune-CL/FinVis_llava_lora" Fin_sci FineTune-CL-Test
# bash scripts/LLaVA/Eval_dom/5_eval_financial.sh Finetune "./checkpoints/LLaVA/Finetune-CL/FinVis_llava_lora" Fin_fin FineTune-CL-Test

bash scripts/LLaVA/Eval_dom/1_eval_remotesensing.sh  Finetune "./checkpoints/LLaVA/Finetune-CL/FinVis_llava_lora_replay" Fin_rs FineTune-CL-Test-replay
bash scripts/LLaVA/Eval_dom/2_eval_medical.sh Finetune "./checkpoints/LLaVA/Finetune-CL/FinVis_llava_lora_replay" Fin_med FineTune-CL-Test-replay
bash scripts/LLaVA/Eval_dom/3_eval_drivelm.sh Finetune "./checkpoints/LLaVA/Finetune-CL/FinVis_llava_lora_replay" Fin_drivelm FineTune-CL-Test-replay
bash scripts/LLaVA/Eval_dom/4_eval_sci.sh Finetune "./checkpoints/LLaVA/Finetune-CL/FinVis_llava_lora_replay" Fin_sci FineTune-CL-Test-replay
bash scripts/LLaVA/Eval_dom/5_eval_financial.sh Finetune "./checkpoints/LLaVA/Finetune-CL/FinVis_llava_lora_replay" Fin_fin FineTune-CL-Test-replay

# # lora train sequential training test (use last to test) MoE (8)
# bash scripts/LLaVA/Eval_dom/1_eval_remotesensing.sh  Finetune "./checkpoints/LLaVA/Finetune-CL-MoE/FinVis_llava_lora_MOE" Fin_rs FineTune-CL-MoE-Test
# bash scripts/LLaVA/Eval_dom/2_eval_medical.sh Finetune "./checkpoints/LLaVA/Finetune-CL-MoE/FinVis_llava_lora_MOE" Fin_med FineTune-CL-MoE-Test
# bash scripts/LLaVA/Eval_dom/3_eval_drivelm.sh Finetune "./checkpoints/LLaVA/Finetune-CL-MoE/FinVis_llava_lora_MOE" Fin_drivelm FineTune-CL-MoE-Test
# bash scripts/LLaVA/Eval_dom/4_eval_sci.sh Finetune "./checkpoints/LLaVA/Finetune-CL-MoE/FinVis_llava_lora_MOE" Fin_sci FineTune-CL-MoE-Test
# bash scripts/LLaVA/Eval_dom/5_eval_financial.sh Finetune "./checkpoints/LLaVA/Finetune-CL-MoE/FinVis_llava_lora_MOE" Fin_fin FineTune-CL-MoE-Test

bash scripts/LLaVA/Eval_dom/1_eval_remotesensing.sh  Finetune "./checkpoints/LLaVA/Finetune-CL-MoE/FinVis_llava_lora_MOE_replay" Fin_rs FineTune-CL-MoE-Test-replay
bash scripts/LLaVA/Eval_dom/2_eval_medical.sh Finetune "./checkpoints/LLaVA/Finetune-CL-MoE/FinVis_llava_lora_MOE_replay" Fin_med FineTune-CL-MoE-Test-replay
bash scripts/LLaVA/Eval_dom/3_eval_drivelm.sh Finetune "./checkpoints/LLaVA/Finetune-CL-MoE/FinVis_llava_lora_MOE_replay" Fin_drivelm FineTune-CL-MoE-Test-replay
bash scripts/LLaVA/Eval_dom/4_eval_sci.sh Finetune "./checkpoints/LLaVA/Finetune-CL-MoE/FinVis_llava_lora_MOE_replay" Fin_sci FineTune-CL-MoE-Test-replay
bash scripts/LLaVA/Eval_dom/5_eval_financial.sh Finetune "./checkpoints/LLaVA/Finetune-CL-MoE/FinVis_llava_lora_MOE_replay" Fin_fin FineTune-CL-MoE-Test-replay

# # llava1.5 test
# bash scripts/LLaVA/Eval_dom/1_eval_remotesensing.sh "" ./checkpoints/LLaVA/Vicuna/llava-7b-v1.5 llava_rs
# bash scripts/LLaVA/Eval_dom/3_eval_drivelm.sh "" ./checkpoints/LLaVA/Vicuna/llava-7b-v1.5 llava_drivelm
# bash scripts/LLaVA/Eval_dom/4_eval_sci.sh "" ./checkpoints/LLaVA/Vicuna/llava-7b-v1.5 llava_sci
# bash scripts/LLaVA/Eval_dom/5_eval_financial.sh "" ./checkpoints/LLaVA/Vicuna/llava-7b-v1.5 llava_fin

# train single lora
# bash scripts/LLaVA/Train_dom_single/10_Financial.sh

# # lora test (no CL)
# bash scripts/LLaVA/Eval_dom/1_eval_remotesensing.sh ""
# bash scripts/LLaVA/Eval_dom/3_eval_drivelm.sh ""
# bash scripts/LLaVA/Eval_dom/2_eval_medical.sh ""
# bash scripts/LLaVA/Eval_dom/4_eval_sci.sh ""
# bash scripts/LLaVA/Eval_dom/5_eval_financial.sh ""

# # 交叉test,只有单独训练出来的lora测其他，名字模型在前，数据集在后
# bash scripts/LLaVA/Eval_dom/1_eval_remotesensing.sh "" ./checkpoints/LLaVA/CoIN/Sci_llava_lora2e-5 Sci_rs
# bash scripts/LLaVA/Eval_dom/2_eval_medical.sh "" ./checkpoints/LLaVA/CoIN/Sci_llava_lora2e-5 Sci_pathvqa
# bash scripts/LLaVA/Eval_dom/3_eval_drivelm.sh "" ./checkpoints/LLaVA/CoIN/Sci_llava_lora2e-5 Sci_drivelm
# bash scripts/LLaVA/Eval_dom/5_eval_financial.sh "" ./checkpoints/LLaVA/CoIN/Sci_llava_lora Sci_fin

# bash scripts/LLaVA/Eval_dom/1_eval_remotesensing.sh "" ./checkpoints/LLaVA/CoIN/Medical_llava_lora PathVQA_rs
# bash scripts/LLaVA/Eval_dom/3_eval_drivelm.sh "" ./checkpoints/LLaVA/CoIN/Medical_llava_lora PathVQA_drivelm
# bash scripts/LLaVA/Eval_dom/4_eval_sci.sh "" ./checkpoints/LLaVA/CoIN/Medical_llava_lora PathVQA_sci
# bash scripts/LLaVA/Eval_dom/5_eval_financial.sh "" ./checkpoints/LLaVA/CoIN/Medical_llava_lora PathVQA_fin

# bash scripts/LLaVA/Eval_dom/1_eval_remotesensing.sh "" ./checkpoints/LLaVA/CoIN/DriveLM_llava_lora DriveLM_rs
# bash scripts/LLaVA/Eval_dom/2_eval_medical.sh "" ./checkpoints/LLaVA/CoIN/DriveLM_llava_lora DriveLM_pathvqa
# bash scripts/LLaVA/Eval_dom/4_eval_sci.sh "" ./checkpoints/LLaVA/CoIN/DriveLM_llava_lora DriveLM_sci
# bash scripts/LLaVA/Eval_dom/5_eval_financial.sh "" ./checkpoints/LLaVA/CoIN/DriveLM_llava_lora DriveLM_fin

# bash scripts/LLaVA/Eval_dom/2_eval_medical.sh "" ./checkpoints/LLaVA/CoIN/RemoteSensing_llava_lora RS_pathvqa
# bash scripts/LLaVA/Eval_dom/3_eval_drivelm.sh "" ./checkpoints/LLaVA/CoIN/RemoteSensing_llava_lora RS_drivelm
# bash scripts/LLaVA/Eval_dom/4_eval_sci.sh "" ./checkpoints/LLaVA/CoIN/RemoteSensing_llava_lora RS_sci
# bash scripts/LLaVA/Eval_dom/5_eval_financial.sh "" ./checkpoints/LLaVA/CoIN/RemoteSensing_llava_lora RS_fin

# bash scripts/LLaVA/Eval_dom/1_eval_remotesensing.sh "" ./checkpoints/LLaVA/CoIN/FinVis_llava_lora Fin_rs
# bash scripts/LLaVA/Eval_dom/2_eval_medical.sh "" ./checkpoints/LLaVA/CoIN/FinVis_llava_lora Fin_pathvqa
# bash scripts/LLaVA/Eval_dom/3_eval_drivelm.sh "" ./checkpoints/LLaVA/CoIN/FinVis_llava_lora Fin_drivelm
# bash scripts/LLaVA/Eval_dom/4_eval_sci.sh "" ./checkpoints/LLaVA/CoIN/FinVis_llava_lora Fin_sci