# bash scripts/LLaVA/Train_dom_single/10_Financial.sh RANK8 "" "" 8
# bash scripts/LLaVA/Train_dom_single/10_Financial.sh RANK16 "" "" 16 
# bash scripts/LLaVA/Train_dom_single/10_Financial.sh RANK32 "" "" 32
# bash scripts/LLaVA/Train_dom_single/10_Financial.sh RANK64 "" "" 64
# bash scripts/LLaVA/Train_dom_single/10_Financial.sh RANK256 "" "" 256

# bash scripts/LLaVA/Train_dom_single/11_Medical.sh RANK8 "" "" 8
# bash scripts/LLaVA/Train_dom_single/11_Medical.sh RANK16 "" "" 16
# bash scripts/LLaVA/Train_dom_single/11_Medical.sh RANK32 "" "" 32
# bash scripts/LLaVA/Train_dom_single/11_Medical.sh RANK64 "" "" 64
# bash scripts/LLaVA/Train_dom_single/11_Medical.sh RANK256 "" "" 256

# bash scripts/LLaVA/Train_dom_single/12_Sci.sh RANK8 "" "" 8
# bash scripts/LLaVA/Train_dom_single/12_Sci.sh RANK16 "" "" 16
# bash scripts/LLaVA/Train_dom_single/12_Sci.sh RANK32 "" "" 32
# bash scripts/LLaVA/Train_dom_single/12_Sci.sh RANK64 "" "" 64
# bash scripts/LLaVA/Train_dom_single/12_Sci.sh RANK256 "" "" 256

# bash scripts/LLaVA/Train_dom_single/13_RemoteSensing.sh RANK8 "" "" 8
# bash scripts/LLaVA/Train_dom_single/13_RemoteSensing.sh RANK16 "" "" 16
# bash scripts/LLaVA/Train_dom_single/13_RemoteSensing.sh RANK32 "" "" 32
# bash scripts/LLaVA/Train_dom_single/13_RemoteSensing.sh RANK64 "" "" 64
# bash scripts/LLaVA/Train_dom_single/13_RemoteSensing.sh RANK256 "" "" 256

# bash scripts/LLaVA/Train_dom_single/14_DriveLM.sh RANK8 "" "" 8
# bash scripts/LLaVA/Train_dom_single/14_DriveLM.sh RANK16 "" "" 16
# bash scripts/LLaVA/Train_dom_single/14_DriveLM.sh RANK32 "" "" 32
# bash scripts/LLaVA/Train_dom_single/14_DriveLM.sh RANK64 "" "" 64
# bash scripts/LLaVA/Train_dom_single/14_DriveLM.sh RANK256 "" "" 256

# bash scripts/LLaVA/Eval_dom/1_eval_remotesensing.sh "" checkpoints/LLaVA/RANK8/RemoteSensing_llava_lora "" RANK8
# bash scripts/LLaVA/Eval_dom/1_eval_remotesensing.sh "" checkpoints/LLaVA/RANK16/RemoteSensing_llava_lora "" RANK16
# bash scripts/LLaVA/Eval_dom/1_eval_remotesensing.sh "" checkpoints/LLaVA/RANK32/RemoteSensing_llava_lora "" RANK32
# bash scripts/LLaVA/Eval_dom/1_eval_remotesensing.sh "" checkpoints/LLaVA/RANK64/RemoteSensing_llava_lora "" RANK64
# bash scripts/LLaVA/Eval_dom/1_eval_remotesensing.sh "" checkpoints/LLaVA/RANK256/RemoteSensing_llava_lora "" RANK256

# bash scripts/LLaVA/Eval_dom/1_eval_remotesensing.sh "" checkpoints/LLaVA/RANK8/FinVis_llava_lora Fin_rs RANK8 
# bash scripts/LLaVA/Eval_dom/1_eval_remotesensing.sh "" checkpoints/LLaVA/RANK16/FinVis_llava_lora Fin_rs RANK16 
# bash scripts/LLaVA/Eval_dom/1_eval_remotesensing.sh "" checkpoints/LLaVA/RANK32/FinVis_llava_lora Fin_rs RANK32 
# bash scripts/LLaVA/Eval_dom/1_eval_remotesensing.sh "" checkpoints/LLaVA/RANK64/FinVis_llava_lora Fin_rs RANK64 
# bash scripts/LLaVA/Eval_dom/1_eval_remotesensing.sh "" checkpoints/LLaVA/RANK256/FinVis_llava_lora Fin_rs RANK256 

# bash scripts/LLaVA/Eval_dom/1_eval_remotesensing.sh "" checkpoints/LLaVA/RANK8/Medical_llava_lora PathVQA_rs RANK8 
# bash scripts/LLaVA/Eval_dom/1_eval_remotesensing.sh "" checkpoints/LLaVA/RANK16/Medical_llava_lora PathVQA_rs RANK16 
# bash scripts/LLaVA/Eval_dom/1_eval_remotesensing.sh "" checkpoints/LLaVA/RANK32/Medical_llava_lora PathVQA_rs RANK32 
# bash scripts/LLaVA/Eval_dom/1_eval_remotesensing.sh "" checkpoints/LLaVA/RANK64/Medical_llava_lora PathVQA_rs RANK64 
# bash scripts/LLaVA/Eval_dom/1_eval_remotesensing.sh "" checkpoints/LLaVA/RANK256/Medical_llava_lora PathVQA_rs RANK256 

# bash scripts/LLaVA/Eval_dom/1_eval_remotesensing.sh "" checkpoints/LLaVA/RANK8/Sci_llava_lora Sci_rs RANK8 
# bash scripts/LLaVA/Eval_dom/1_eval_remotesensing.sh "" checkpoints/LLaVA/RANK16/Sci_llava_lora Sci_rs RANK16 
# bash scripts/LLaVA/Eval_dom/1_eval_remotesensing.sh "" checkpoints/LLaVA/RANK32/Sci_llava_lora Sci_rs RANK32 
# bash scripts/LLaVA/Eval_dom/1_eval_remotesensing.sh "" checkpoints/LLaVA/RANK64/Sci_llava_lora Sci_rs RANK64 
# bash scripts/LLaVA/Eval_dom/1_eval_remotesensing.sh "" checkpoints/LLaVA/RANK256/Sci_llava_lora Sci_rs RANK256 

# bash scripts/LLaVA/Eval_dom/1_eval_remotesensing.sh "" checkpoints/LLaVA/RANK8/DriveLM_llava_lora DriveLM_rs RANK8 
# bash scripts/LLaVA/Eval_dom/1_eval_remotesensing.sh "" checkpoints/LLaVA/RANK16/DriveLM_llava_lora DriveLM_rs RANK16 
# bash scripts/LLaVA/Eval_dom/1_eval_remotesensing.sh "" checkpoints/LLaVA/RANK32/DriveLM_llava_lora DriveLM_rs RANK32 
# bash scripts/LLaVA/Eval_dom/1_eval_remotesensing.sh "" checkpoints/LLaVA/RANK64/DriveLM_llava_lora DriveLM_rs RANK64 
# bash scripts/LLaVA/Eval_dom/1_eval_remotesensing.sh "" checkpoints/LLaVA/RANK256/DriveLM_llava_lora DriveLM_rs RANK256 

# bash scripts/LLaVA/Eval_dom/2_eval_medical.sh "" checkpoints/LLaVA/RANK8/Medical_llava_lora "" RANK8
# bash scripts/LLaVA/Eval_dom/2_eval_medical.sh "" checkpoints/LLaVA/RANK16/Medical_llava_lora "" RANK16
# bash scripts/LLaVA/Eval_dom/2_eval_medical.sh "" checkpoints/LLaVA/RANK32/Medical_llava_lora "" RANK32
# bash scripts/LLaVA/Eval_dom/2_eval_medical.sh "" checkpoints/LLaVA/RANK64/Medical_llava_lora "" RANK64
# bash scripts/LLaVA/Eval_dom/2_eval_medical.sh "" checkpoints/LLaVA/RANK256/Medical_llava_lora "" RANK256

# bash scripts/LLaVA/Eval_dom/2_eval_medical.sh "" checkpoints/LLaVA/RANK8/FinVis_llava_lora Fin_pathvqa RANK8 
# bash scripts/LLaVA/Eval_dom/2_eval_medical.sh "" checkpoints/LLaVA/RANK16/FinVis_llava_lora Fin_pathvqa RANK16 
# bash scripts/LLaVA/Eval_dom/2_eval_medical.sh "" checkpoints/LLaVA/RANK32/FinVis_llava_lora Fin_pathvqa RANK32 
# bash scripts/LLaVA/Eval_dom/2_eval_medical.sh "" checkpoints/LLaVA/RANK64/FinVis_llava_lora Fin_pathvqa RANK64 
# bash scripts/LLaVA/Eval_dom/2_eval_medical.sh "" checkpoints/LLaVA/RANK256/FinVis_llava_lora Fin_pathvqa RANK256 

# bash scripts/LLaVA/Eval_dom/2_eval_medical.sh "" checkpoints/LLaVA/RANK8/Sci_llava_lora Sci_pathvqa RANK8 
# bash scripts/LLaVA/Eval_dom/2_eval_medical.sh "" checkpoints/LLaVA/RANK16/Sci_llava_lora Sci_pathvqa RANK16 
# bash scripts/LLaVA/Eval_dom/2_eval_medical.sh "" checkpoints/LLaVA/RANK32/Sci_llava_lora Sci_pathvqa RANK32 
# bash scripts/LLaVA/Eval_dom/2_eval_medical.sh "" checkpoints/LLaVA/RANK64/Sci_llava_lora Sci_pathvqa RANK64 
# bash scripts/LLaVA/Eval_dom/2_eval_medical.sh "" checkpoints/LLaVA/RANK256/Sci_llava_lora Sci_pathvqa RANK256 

# bash scripts/LLaVA/Eval_dom/2_eval_medical.sh "" checkpoints/LLaVA/RANK8/DriveLM_llava_lora DriveLM_pathvqa RANK8 
# bash scripts/LLaVA/Eval_dom/2_eval_medical.sh "" checkpoints/LLaVA/RANK16/DriveLM_llava_lora DriveLM_pathvqa RANK16 
# bash scripts/LLaVA/Eval_dom/2_eval_medical.sh "" checkpoints/LLaVA/RANK32/DriveLM_llava_lora DriveLM_pathvqa RANK32 
# bash scripts/LLaVA/Eval_dom/2_eval_medical.sh "" checkpoints/LLaVA/RANK64/DriveLM_llava_lora DriveLM_pathvqa RANK64 
# bash scripts/LLaVA/Eval_dom/2_eval_medical.sh "" checkpoints/LLaVA/RANK256/DriveLM_llava_lora DriveLM_pathvqa RANK256 

# bash scripts/LLaVA/Eval_dom/2_eval_medical.sh "" checkpoints/LLaVA/RANK8/RemoteSensing_llava_lora RS_pathvqa RANK8 
# bash scripts/LLaVA/Eval_dom/2_eval_medical.sh "" checkpoints/LLaVA/RANK16/RemoteSensing_llava_lora RS_pathvqa RANK16 
# bash scripts/LLaVA/Eval_dom/2_eval_medical.sh "" checkpoints/LLaVA/RANK32/RemoteSensing_llava_lora RS_pathvqa RANK32 
# bash scripts/LLaVA/Eval_dom/2_eval_medical.sh "" checkpoints/LLaVA/RANK64/RemoteSensing_llava_lora RS_pathvqa RANK64 
# bash scripts/LLaVA/Eval_dom/2_eval_medical.sh "" checkpoints/LLaVA/RANK256/RemoteSensing_llava_lora RS_pathvqa RANK256 

# # 第三个
# bash scripts/LLaVA/Eval_dom/3_eval_drivelm.sh "" checkpoints/LLaVA/RANK8/DriveLM_llava_lora "" RANK8
# bash scripts/LLaVA/Eval_dom/3_eval_drivelm.sh "" checkpoints/LLaVA/RANK16/DriveLM_llava_lora "" RANK16
# bash scripts/LLaVA/Eval_dom/3_eval_drivelm.sh "" checkpoints/LLaVA/RANK32/DriveLM_llava_lora "" RANK32
# bash scripts/LLaVA/Eval_dom/3_eval_drivelm.sh "" checkpoints/LLaVA/RANK64/DriveLM_llava_lora "" RANK64
# bash scripts/LLaVA/Eval_dom/3_eval_drivelm.sh "" checkpoints/LLaVA/RANK256/DriveLM_llava_lora "" RANK256

# bash scripts/LLaVA/Eval_dom/3_eval_drivelm.sh "" checkpoints/LLaVA/RANK8/FinVis_llava_lora Fin_drivelm RANK8 
# bash scripts/LLaVA/Eval_dom/3_eval_drivelm.sh "" checkpoints/LLaVA/RANK16/FinVis_llava_lora Fin_drivelm RANK16 
# bash scripts/LLaVA/Eval_dom/3_eval_drivelm.sh "" checkpoints/LLaVA/RANK32/FinVis_llava_lora Fin_drivelm RANK32 
# bash scripts/LLaVA/Eval_dom/3_eval_drivelm.sh "" checkpoints/LLaVA/RANK64/FinVis_llava_lora Fin_drivelm RANK64 
# bash scripts/LLaVA/Eval_dom/3_eval_drivelm.sh "" checkpoints/LLaVA/RANK256/FinVis_llava_lora Fin_drivelm RANK256 

# bash scripts/LLaVA/Eval_dom/3_eval_drivelm.sh "" checkpoints/LLaVA/RANK8/Sci_llava_lora Sci_drivelm RANK8 
# bash scripts/LLaVA/Eval_dom/3_eval_drivelm.sh "" checkpoints/LLaVA/RANK16/Sci_llava_lora Sci_drivelm RANK16 
# bash scripts/LLaVA/Eval_dom/3_eval_drivelm.sh "" checkpoints/LLaVA/RANK32/Sci_llava_lora Sci_drivelm RANK32 
# bash scripts/LLaVA/Eval_dom/3_eval_drivelm.sh "" checkpoints/LLaVA/RANK64/Sci_llava_lora Sci_drivelm RANK64 
# bash scripts/LLaVA/Eval_dom/3_eval_drivelm.sh "" checkpoints/LLaVA/RANK256/Sci_llava_lora Sci_drivelm RANK256 

# bash scripts/LLaVA/Eval_dom/3_eval_drivelm.sh "" checkpoints/LLaVA/RANK8/RemoteSensing_llava_lora RS_drivelm RANK8 
# bash scripts/LLaVA/Eval_dom/3_eval_drivelm.sh "" checkpoints/LLaVA/RANK16/RemoteSensing_llava_lora RS_drivelm RANK16 
# bash scripts/LLaVA/Eval_dom/3_eval_drivelm.sh "" checkpoints/LLaVA/RANK32/RemoteSensing_llava_lora RS_drivelm RANK32 
# bash scripts/LLaVA/Eval_dom/3_eval_drivelm.sh "" checkpoints/LLaVA/RANK64/RemoteSensing_llava_lora RS_drivelm RANK64 
# bash scripts/LLaVA/Eval_dom/3_eval_drivelm.sh "" checkpoints/LLaVA/RANK256/RemoteSensing_llava_lora RS_drivelm RANK256 

# bash scripts/LLaVA/Eval_dom/3_eval_drivelm.sh "" checkpoints/LLaVA/RANK8/Medical_llava_lora PathVQA_drivelm RANK8 
# bash scripts/LLaVA/Eval_dom/3_eval_drivelm.sh "" checkpoints/LLaVA/RANK16/Medical_llava_lora PathVQA_drivelm RANK16 
# bash scripts/LLaVA/Eval_dom/3_eval_drivelm.sh "" checkpoints/LLaVA/RANK32/Medical_llava_lora PathVQA_drivelm RANK32 
# bash scripts/LLaVA/Eval_dom/3_eval_drivelm.sh "" checkpoints/LLaVA/RANK64/Medical_llava_lora PathVQA_drivelm RANK64 
# bash scripts/LLaVA/Eval_dom/3_eval_drivelm.sh "" checkpoints/LLaVA/RANK256/Medical_llava_lora PathVQA_drivelm RANK256 

# # 第四个
# bash scripts/LLaVA/Eval_dom/4_eval_sci.sh "" checkpoints/LLaVA/RANK8/Sci_llava_lora "" RANK8
# bash scripts/LLaVA/Eval_dom/4_eval_sci.sh "" checkpoints/LLaVA/RANK16/Sci_llava_lora "" RANK16
# bash scripts/LLaVA/Eval_dom/4_eval_sci.sh "" checkpoints/LLaVA/RANK32/Sci_llava_lora "" RANK32
# bash scripts/LLaVA/Eval_dom/4_eval_sci.sh "" checkpoints/LLaVA/RANK64/Sci_llava_lora "" RANK64
# bash scripts/LLaVA/Eval_dom/4_eval_sci.sh "" checkpoints/LLaVA/RANK256/Sci_llava_lora "" RANK256

# bash scripts/LLaVA/Eval_dom/4_eval_sci.sh "" checkpoints/LLaVA/RANK8/FinVis_llava_lora Fin_sci RANK8 
# bash scripts/LLaVA/Eval_dom/4_eval_sci.sh "" checkpoints/LLaVA/RANK16/FinVis_llava_lora Fin_sci RANK16 
# bash scripts/LLaVA/Eval_dom/4_eval_sci.sh "" checkpoints/LLaVA/RANK32/FinVis_llava_lora Fin_sci RANK32 
# bash scripts/LLaVA/Eval_dom/4_eval_sci.sh "" checkpoints/LLaVA/RANK64/FinVis_llava_lora Fin_sci RANK64 
# bash scripts/LLaVA/Eval_dom/4_eval_sci.sh "" checkpoints/LLaVA/RANK256/FinVis_llava_lora Fin_sci RANK256 

# bash scripts/LLaVA/Eval_dom/4_eval_sci.sh "" checkpoints/LLaVA/RANK8/RemoteSensing_llava_lora RS_sci RANK8 
# bash scripts/LLaVA/Eval_dom/4_eval_sci.sh "" checkpoints/LLaVA/RANK16/RemoteSensing_llava_lora RS_sci RANK16 
# bash scripts/LLaVA/Eval_dom/4_eval_sci.sh "" checkpoints/LLaVA/RANK32/RemoteSensing_llava_lora RS_sci RANK32 
# bash scripts/LLaVA/Eval_dom/4_eval_sci.sh "" checkpoints/LLaVA/RANK64/RemoteSensing_llava_lora RS_sci RANK64 
# bash scripts/LLaVA/Eval_dom/4_eval_sci.sh "" checkpoints/LLaVA/RANK256/RemoteSensing_llava_lora RS_sci RANK256 

# bash scripts/LLaVA/Eval_dom/4_eval_sci.sh "" checkpoints/LLaVA/RANK8/Medical_llava_lora PathVQA_sci RANK8 
# bash scripts/LLaVA/Eval_dom/4_eval_sci.sh "" checkpoints/LLaVA/RANK16/Medical_llava_lora PathVQA_sci RANK16 
# bash scripts/LLaVA/Eval_dom/4_eval_sci.sh "" checkpoints/LLaVA/RANK32/Medical_llava_lora PathVQA_sci RANK32 
# bash scripts/LLaVA/Eval_dom/4_eval_sci.sh "" checkpoints/LLaVA/RANK64/Medical_llava_lora PathVQA_sci RANK64 
# bash scripts/LLaVA/Eval_dom/4_eval_sci.sh "" checkpoints/LLaVA/RANK256/Medical_llava_lora PathVQA_sci RANK256 

# bash scripts/LLaVA/Eval_dom/4_eval_sci.sh "" checkpoints/LLaVA/RANK8/DriveLM_llava_lora DriveLM_sci RANK8 
# bash scripts/LLaVA/Eval_dom/4_eval_sci.sh "" checkpoints/LLaVA/RANK16/DriveLM_llava_lora DriveLM_sci RANK16 
# bash scripts/LLaVA/Eval_dom/4_eval_sci.sh "" checkpoints/LLaVA/RANK32/DriveLM_llava_lora DriveLM_sci RANK32 
# bash scripts/LLaVA/Eval_dom/4_eval_sci.sh "" checkpoints/LLaVA/RANK64/DriveLM_llava_lora DriveLM_sci RANK64 
# bash scripts/LLaVA/Eval_dom/4_eval_sci.sh "" checkpoints/LLaVA/RANK256/DriveLM_llava_lora DriveLM_sci RANK256 

# # 第五个
# bash scripts/LLaVA/Eval_dom/5_eval_financial.sh "" checkpoints/LLaVA/RANK8/FinVis_llava_lora "" RANK8
# bash scripts/LLaVA/Eval_dom/5_eval_financial.sh "" checkpoints/LLaVA/RANK16/FinVis_llava_lora "" RANK16
# bash scripts/LLaVA/Eval_dom/5_eval_financial.sh "" checkpoints/LLaVA/RANK32/FinVis_llava_lora "" RANK32
# bash scripts/LLaVA/Eval_dom/5_eval_financial.sh "" checkpoints/LLaVA/RANK64/FinVis_llava_lora "" RANK64
# bash scripts/LLaVA/Eval_dom/5_eval_financial.sh "" checkpoints/LLaVA/RANK256/FinVis_llava_lora "" RANK256

# bash scripts/LLaVA/Eval_dom/5_eval_financial.sh "" checkpoints/LLaVA/RANK8/Sci_llava_lora Sci_fin RANK8 
# bash scripts/LLaVA/Eval_dom/5_eval_financial.sh "" checkpoints/LLaVA/RANK16/Sci_llava_lora Sci_fin RANK16 
# bash scripts/LLaVA/Eval_dom/5_eval_financial.sh "" checkpoints/LLaVA/RANK32/Sci_llava_lora Sci_fin RANK32 
# bash scripts/LLaVA/Eval_dom/5_eval_financial.sh "" checkpoints/LLaVA/RANK64/Sci_llava_lora Sci_fin RANK64 
# bash scripts/LLaVA/Eval_dom/5_eval_financial.sh "" checkpoints/LLaVA/RANK256/Sci_llava_lora Sci_fin RANK256 

# bash scripts/LLaVA/Eval_dom/5_eval_financial.sh "" checkpoints/LLaVA/RANK8/RemoteSensing_llava_lora RS_fin RANK8 
# bash scripts/LLaVA/Eval_dom/5_eval_financial.sh "" checkpoints/LLaVA/RANK16/RemoteSensing_llava_lora RS_fin RANK16 
# bash scripts/LLaVA/Eval_dom/5_eval_financial.sh "" checkpoints/LLaVA/RANK32/RemoteSensing_llava_lora RS_fin RANK32 
# bash scripts/LLaVA/Eval_dom/5_eval_financial.sh "" checkpoints/LLaVA/RANK64/RemoteSensing_llava_lora RS_fin RANK64 
# bash scripts/LLaVA/Eval_dom/5_eval_financial.sh "" checkpoints/LLaVA/RANK256/RemoteSensing_llava_lora RS_fin RANK256 

# bash scripts/LLaVA/Eval_dom/5_eval_financial.sh "" checkpoints/LLaVA/RANK8/Medical_llava_lora PathVQA_fin RANK8 
# bash scripts/LLaVA/Eval_dom/5_eval_financial.sh "" checkpoints/LLaVA/RANK16/Medical_llava_lora PathVQA_fin RANK16 
# bash scripts/LLaVA/Eval_dom/5_eval_financial.sh "" checkpoints/LLaVA/RANK32/Medical_llava_lora PathVQA_fin RANK32 
# bash scripts/LLaVA/Eval_dom/5_eval_financial.sh "" checkpoints/LLaVA/RANK64/Medical_llava_lora PathVQA_fin RANK64 
# bash scripts/LLaVA/Eval_dom/5_eval_financial.sh "" checkpoints/LLaVA/RANK256/Medical_llava_lora PathVQA_fin RANK256 

# bash scripts/LLaVA/Eval_dom/5_eval_financial.sh "" checkpoints/LLaVA/RANK8/DriveLM_llava_lora DriveLM_fin RANK8 
# bash scripts/LLaVA/Eval_dom/5_eval_financial.sh "" checkpoints/LLaVA/RANK16/DriveLM_llava_lora DriveLM_fin RANK16 
# bash scripts/LLaVA/Eval_dom/5_eval_financial.sh "" checkpoints/LLaVA/RANK32/DriveLM_llava_lora DriveLM_fin RANK32 
# bash scripts/LLaVA/Eval_dom/5_eval_financial.sh "" checkpoints/LLaVA/RANK64/DriveLM_llava_lora DriveLM_fin RANK64 
# bash scripts/LLaVA/Eval_dom/5_eval_financial.sh "" checkpoints/LLaVA/RANK256/DriveLM_llava_lora DriveLM_fin RANK256 

# 矩阵LoRA FT
# bash scripts/LLaVA/Eval_dom/1_eval_remotesensing.sh "" checkpoints/LLaVA/Finetune-CL/Medical_llava_lora PathVQA_rs LoRAMatrix 
# bash scripts/LLaVA/Eval_dom/1_eval_remotesensing.sh "" checkpoints/LLaVA/Finetune-CL/FinVis_llava_lora Fin_rs LoRAMatrix 
# bash scripts/LLaVA/Eval_dom/1_eval_remotesensing.sh "" checkpoints/LLaVA/Finetune-CL/Sci_llava_lora  Sci_rs LoRAMatrix
# bash scripts/LLaVA/Eval_dom/1_eval_remotesensing.sh "" checkpoints/LLaVA/Finetune-CL/DriveLM_llava_lora DriveLM_rs LoRAMatrix 
# bash scripts/LLaVA/Eval_dom/2_eval_medical.sh "" checkpoints/LLaVA/Finetune-CL/FinVis_llava_lora Fin_pathvqa LoRAMatrix 
# bash scripts/LLaVA/Eval_dom/2_eval_medical.sh "" checkpoints/LLaVA/Finetune-CL/Sci_llava_lora Sci_pathvqa LoRAMatrix 
# bash scripts/LLaVA/Eval_dom/2_eval_medical.sh "" checkpoints/LLaVA/Finetune-CL/DriveLM_llava_lora DriveLM_pathvqa LoRAMatrix 
# bash scripts/LLaVA/Eval_dom/3_eval_drivelm.sh "" checkpoints/LLaVA/Finetune-CL/FinVis_llava_lora Fin_drivelm LoRAMatrix 
# bash scripts/LLaVA/Eval_dom/3_eval_drivelm.sh "" checkpoints/LLaVA/Finetune-CL/Sci_llava_lora Sci_drivelm LoRAMatrix 
# bash scripts/LLaVA/Eval_dom/4_eval_sci.sh "" checkpoints/LLaVA/Finetune-CL/FinVis_llava_lora Fin_sci LoRAMatrix 

# bash scripts/LLaVA/Eval_dom/1_eval_remotesensing.sh "" checkpoints/LLaVA/Finetune-CL/RemoteSensing_llava_lora "" LoRAMatrix
# bash scripts/LLaVA/Eval_dom/2_eval_medical.sh "" checkpoints/LLaVA/Finetune-CL/Medical_llava_lora "" LoRAMatrix
# bash scripts/LLaVA/Eval_dom/3_eval_drivelm.sh "" checkpoints/LLaVA/Finetune-CL/DriveLM_llava_lora "" LoRAMatrix
# bash scripts/LLaVA/Eval_dom/4_eval_sci.sh "" checkpoints/LLaVA/Finetune-CL/Sci_llava_lora "" LoRAMatrix

# MoE LoRA FT
# bash scripts/LLaVA/Eval_dom/1_eval_remotesensing.sh "" checkpoints/LLaVA/Finetune-CL-MoE/Medical_llava_lora_MOE PathVQA_rs MoELoRAMatrix
# bash scripts/LLaVA/Eval_dom/1_eval_remotesensing.sh "" checkpoints/LLaVA/Finetune-CL-MoE/FinVis_llava_lora_MOE Fin_rs MoELoRAMatrix 
# bash scripts/LLaVA/Eval_dom/1_eval_remotesensing.sh "" checkpoints/LLaVA/Finetune-CL-MoE/Sci_llava_lora_MOE  Sci_rs MoELoRAMatrix
# bash scripts/LLaVA/Eval_dom/1_eval_remotesensing.sh "" checkpoints/LLaVA/Finetune-CL-MoE/DriveLM_llava_lora_MOE DriveLM_rs MoELoRAMatrix
# bash scripts/LLaVA/Eval_dom/2_eval_medical.sh "" checkpoints/LLaVA/Finetune-CL-MoE/FinVis_llava_lora_MOE Fin_pathvqa MoELoRAMatrix
# bash scripts/LLaVA/Eval_dom/2_eval_medical.sh "" checkpoints/LLaVA/Finetune-CL-MoE/Sci_llava_lora_MOE Sci_pathvqa MoELoRAMatrix
# bash scripts/LLaVA/Eval_dom/2_eval_medical.sh "" checkpoints/LLaVA/Finetune-CL-MoE/DriveLM_llava_lora_MOE DriveLM_pathvqa MoELoRAMatrix
# bash scripts/LLaVA/Eval_dom/3_eval_drivelm.sh "" checkpoints/LLaVA/Finetune-CL-MoE/FinVis_llava_lora_MOE Fin_drivelm MoELoRAMatrix
# bash scripts/LLaVA/Eval_dom/3_eval_drivelm.sh "" checkpoints/LLaVA/Finetune-CL-MoE/Sci_llava_lora_MOE Sci_drivelm MoELoRAMatrix
# bash scripts/LLaVA/Eval_dom/4_eval_sci.sh "" checkpoints/LLaVA/Finetune-CL-MoE/FinVis_llava_lora_MOE Fin_sci MoELoRAMatrix

# bash scripts/LLaVA/Eval_dom/1_eval_remotesensing.sh "" checkpoints/LLaVA/Finetune-CL-MoE/RemoteSensing_llava_lora_MOE "" MoELoRAMatrix
# bash scripts/LLaVA/Eval_dom/2_eval_medical.sh "" checkpoints/LLaVA/Finetune-CL-MoE/Medical_llava_lora_MOE "" MoELoRAMatrix
# bash scripts/LLaVA/Eval_dom/3_eval_drivelm.sh "" checkpoints/LLaVA/Finetune-CL-MoE/DriveLM_llava_lora_MOE "" MoELoRAMatrix
# bash scripts/LLaVA/Eval_dom/4_eval_sci.sh "" checkpoints/LLaVA/Finetune-CL-MoE/Sci_llava_lora_MOE "" MoELoRAMatrix

# # 回放矩阵
# # bash scripts/LLaVA/Eval_dom/1_eval_remotesensing.sh "" checkpoints/LLaVA/CoINReplay/DriveLM_llava_lora DriveLM_rs CoINReplay
# bash scripts/LLaVA/Eval_dom/1_eval_remotesensing.sh "" checkpoints/LLaVA/CoINReplay/FinVis_llava_lora Fin_rs CoINReplay
# # bash scripts/LLaVA/Eval_dom/1_eval_remotesensing.sh "" checkpoints/LLaVA/CoINReplay/Medical_llava_lora PathVQA_rs CoINReplay
# # bash scripts/LLaVA/Eval_dom/1_eval_remotesensing.sh "" checkpoints/LLaVA/CoINReplay/Sci_llava_lora Sci_rs CoINReplay
# # bash scripts/LLaVA/Eval_dom/1_eval_remotesensing.sh "" checkpoints/LLaVA/CoINReplay/RemoteSensing_llava_lora RS_rs CoINReplay
# bash scripts/LLaVA/Eval_dom/2_eval_medical.sh "" checkpoints/LLaVA/CoINReplay/FinVis_llava_lora Fin_pathvqa CoINReplay
# # bash scripts/LLaVA/Eval_dom/2_eval_medical.sh "" checkpoints/LLaVA/CoINReplay/Sci_llava_lora Sci_pathvqa CoINReplay
# # bash scripts/LLaVA/Eval_dom/2_eval_medical.sh "" checkpoints/LLaVA/CoINReplay/DriveLM_llava_lora DriveLM_pathvqa CoINReplay
# # bash scripts/LLaVA/Eval_dom/2_eval_medical.sh "" checkpoints/LLaVA/CoINReplay/Medical_llava_lora PathVQA_pathvqa CoINReplay
# bash scripts/LLaVA/Eval_dom/3_eval_drivelm.sh "" checkpoints/LLaVA/CoINReplay/FinVis_llava_lora Fin_drivelm CoINReplay
# # bash scripts/LLaVA/Eval_dom/3_eval_drivelm.sh "" checkpoints/LLaVA/CoINReplay/Sci_llava_lora Sci_drivelm CoINReplay
# # bash scripts/LLaVA/Eval_dom/3_eval_drivelm.sh "" checkpoints/LLaVA/CoINReplay/DriveLM_llava_lora DriveLM_drivelm CoINReplay
# bash scripts/LLaVA/Eval_dom/4_eval_sci.sh "" checkpoints/LLaVA/CoINReplay/FinVis_llava_lora Fin_sci CoINReplay
# # bash scripts/LLaVA/Eval_dom/4_eval_sci.sh "" checkpoints/LLaVA/CoINReplay/Sci_llava_lora Sci_sci CoINReplay
# bash scripts/LLaVA/Eval_dom/5_eval_financial.sh "" checkpoints/LLaVA/CoINReplay/FinVis_llava_lora Fin_fin CoINReplay

# # MOE回放矩阵
# # bash scripts/LLaVA/Eval_dom/1_eval_remotesensing.sh "" checkpoints/LLaVA/LoRAMOEReplay/DriveLM_llava_lora_MOE DriveLM_rs LoRAMOEReplay
# bash scripts/LLaVA/Eval_dom/1_eval_remotesensing.sh "" checkpoints/LLaVA/LoRAMOEReplay/FinVis_llava_lora_MOE Fin_rs LoRAMOEReplay
# # bash scripts/LLaVA/Eval_dom/1_eval_remotesensing.sh "" checkpoints/LLaVA/LoRAMOEReplay/Medical_llava_lora_MOE PathVQA_rs LoRAMOEReplay
# # bash scripts/LLaVA/Eval_dom/1_eval_remotesensing.sh "" checkpoints/LLaVA/LoRAMOEReplay/Sci_llava_lora_MOE Sci_rs LoRAMOEReplay
# # bash scripts/LLaVA/Eval_dom/1_eval_remotesensing.sh "" checkpoints/LLaVA/LoRAMOEReplay/RemoteSensing_llava_lora_MOE RS_rs LoRAMOEReplay
# bash scripts/LLaVA/Eval_dom/2_eval_medical.sh "" checkpoints/LLaVA/LoRAMOEReplay/FinVis_llava_lora_MOE Fin_pathvqa LoRAMOEReplay
# # bash scripts/LLaVA/Eval_dom/2_eval_medical.sh "" checkpoints/LLaVA/LoRAMOEReplay/Sci_llava_lora_MOE Sci_pathvqa LoRAMOEReplay
# # bash scripts/LLaVA/Eval_dom/2_eval_medical.sh "" checkpoints/LLaVA/LoRAMOEReplay/DriveLM_llava_lora_MOE DriveLM_pathvqa LoRAMOEReplay
# # bash scripts/LLaVA/Eval_dom/2_eval_medical.sh "" checkpoints/LLaVA/LoRAMOEReplay/Medical_llava_lora_MOE PathVQA_pathvqa LoRAMOEReplay
# bash scripts/LLaVA/Eval_dom/3_eval_drivelm.sh "" checkpoints/LLaVA/LoRAMOEReplay/FinVis_llava_lora_MOE Fin_drivelm LoRAMOEReplay
# # bash scripts/LLaVA/Eval_dom/3_eval_drivelm.sh "" checkpoints/LLaVA/LoRAMOEReplay/Sci_llava_lora_MOE Sci_drivelm LoRAMOEReplay
# # bash scripts/LLaVA/Eval_dom/3_eval_drivelm.sh "" checkpoints/LLaVA/LoRAMOEReplay/DriveLM_llava_lora_MOE DriveLM_drivelm LoRAMOEReplay
# bash scripts/LLaVA/Eval_dom/4_eval_sci.sh "" checkpoints/LLaVA/LoRAMOEReplay/FinVis_llava_lora_MOE Fin_sci LoRAMOEReplay
# # bash scripts/LLaVA/Eval_dom/4_eval_sci.sh "" checkpoints/LLaVA/LoRAMOEReplay/Sci_llava_lora_MOE Sci_sci LoRAMOEReplay
# bash scripts/LLaVA/Eval_dom/5_eval_financial.sh "" checkpoints/LLaVA/LoRAMOEReplay/FinVis_llava_lora_MOE Fin_fin LoRAMOEReplay

# 能力维度LoRA回放
# bash scripts/LLaVA/Eval_ability/1_eval_OCR.sh "" checkpoints/LLaVA/AbilityReplay/OCR_llava_lora OCR_ocr AbilityReplay
# bash scripts/LLaVA/Eval_ability/1_eval_OCR.sh "" checkpoints/LLaVA/AbilityReplay/Count_llava_lora Count_ocr AbilityReplay
# bash scripts/LLaVA/Eval_ability/1_eval_OCR.sh "" checkpoints/LLaVA/AbilityReplay/MATH_llava_lora Math_ocr AbilityReplay
# bash scripts/LLaVA/Eval_ability/1_eval_OCR.sh "" checkpoints/LLaVA/AbilityReplay/APP_llava_lora_replay APP_ocr AbilityReplay
# bash scripts/LLaVA/Eval_ability/3_eval_VSR.sh "" checkpoints/LLaVA/AbilityReplay/Count_llava_lora Count_count AbilityReplay
# bash scripts/LLaVA/Eval_ability/3_eval_VSR.sh "" heckpoints/LLaVA/AbilityReplay/MATH_llava_lora Math_count AbilityReplay
# bash scripts/LLaVA/Eval_ability/3_eval_VSR.sh "" checkpoints/LLaVA/AbilityReplay/APP_llava_lora_replay APP_count AbilityReplay
# bash scripts/LLaVA/Eval_ability/4_eval_APP.sh "" checkpoints/LLaVA/AbilityReplay/APP_llava_lora_replay APP_app AbilityReplay

# 能力维度MoE回放
# bash scripts/LLaVA/Eval_ability/1_eval_OCR.sh "" checkpoints/LLaVA/AbiLoRAMOEReplay/OCR_llava_lora_MOE OCR_ocr AbiLoRAMOEReplay
# bash scripts/LLaVA/Eval_ability/1_eval_OCR.sh "" checkpoints/LLaVA/AbiLoRAMOEReplay/Count_llava_lora_MOE Count_ocr AbiLoRAMOEReplay
# bash scripts/LLaVA/Eval_ability/1_eval_OCR.sh "" checkpoints/LLaVA/AbiLoRAMOEReplay/MATH_llava_lora_MOE Math_ocr AbiLoRAMOEReplay
# bash scripts/LLaVA/Eval_ability/1_eval_OCR.sh "" checkpoints/LLaVA/AbiLoRAMOEReplay/APP_llava_lora_replay_MOE APP_ocr AbiLoRAMOEReplay
# # bash scripts/LLaVA/Eval_ability/3_eval_VSR.sh "" checkpoints/LLaVA/AbiLoRAMOEReplay/Count_llava_lora_MOE Count_count AbiLoRAMOEReplay
# bash scripts/LLaVA/Eval_ability/3_eval_VSR.sh "" checkpoints/LLaVA/AbiLoRAMOEReplay/MATH_llava_lora_MOE Math_count AbiLoRAMOEReplay
# bash scripts/LLaVA/Eval_ability/3_eval_VSR.sh "" checkpoints/LLaVA/AbiLoRAMOEReplay/APP_llava_lora_replay_MOE APP_count AbiLoRAMOEReplay
# bash scripts/LLaVA/Eval_ability/4_eval_APP.sh "" checkpoints/LLaVA/AbiLoRAMOEReplay/APP_llava_lora_replay_MOE APP_app AbiLoRAMOEReplay

# 补充错误的
# bash scripts/LLaVA/Eval_dom/1_eval_remotesensing.sh "" checkpoints/LLaVA/CoINReplay/DriveLM_llava_lora DriveLM_rs CoINReplay
# bash scripts/LLaVA/Eval_dom/1_eval_remotesensing.sh "" checkpoints/LLaVA/CoINReplay/FinVis_llava_lora Fin_rs CoINReplay
# # bash scripts/LLaVA/Eval_dom/1_eval_remotesensing.sh "" checkpoints/LLaVA/CoINReplay/Medical_llava_lora PathVQA_rs CoINReplay
# # bash scripts/LLaVA/Eval_dom/1_eval_remotesensing.sh "" checkpoints/LLaVA/CoINReplay/Sci_llava_lora Sci_rs CoINReplay
# # bash scripts/LLaVA/Eval_dom/1_eval_remotesensing.sh "" checkpoints/LLaVA/CoINReplay/RemoteSensing_llava_lora RS_rs CoINReplay
# bash scripts/LLaVA/Eval_dom/2_eval_medical.sh "" checkpoints/LLaVA/CoINReplay/FinVis_llava_lora Fin_pathvqa CoINReplay
# # bash scripts/LLaVA/Eval_dom/2_eval_medical.sh "" checkpoints/LLaVA/CoINReplay/Sci_llava_lora Sci_pathvqa CoINReplay
# # bash scripts/LLaVA/Eval_dom/2_eval_medical.sh "" checkpoints/LLaVA/CoINReplay/DriveLM_llava_lora DriveLM_pathvqa CoINReplay
# # bash scripts/LLaVA/Eval_dom/2_eval_medical.sh "" checkpoints/LLaVA/CoINReplay/Medical_llava_lora PathVQA_pathvqa CoINReplay
# bash scripts/LLaVA/Eval_dom/3_eval_drivelm.sh "" checkpoints/LLaVA/CoINReplay/FinVis_llava_lora Fin_drivelm CoINReplay
# # bash scripts/LLaVA/Eval_dom/3_eval_drivelm.sh "" checkpoints/LLaVA/CoINReplay/Sci_llava_lora Sci_drivelm CoINReplay
# # bash scripts/LLaVA/Eval_dom/3_eval_drivelm.sh "" checkpoints/LLaVA/CoINReplay/DriveLM_llava_lora DriveLM_drivelm CoINReplay
# bash scripts/LLaVA/Eval_dom/4_eval_sci.sh "" checkpoints/LLaVA/CoINReplay/FinVis_llava_lora Fin_sci CoINReplay
# # bash scripts/LLaVA/Eval_dom/4_eval_sci.sh "" checkpoints/LLaVA/CoINReplay/Sci_llava_lora Sci_sci CoINReplay
# bash scripts/LLaVA/Eval_dom/5_eval_financial.sh "" checkpoints/LLaVA/CoINReplay/FinVis_llava_lora Fin_fin CoINReplay
# # bash scripts/LLaVA/Eval_dom/1_eval_remotesensing.sh "" checkpoints/LLaVA/LoRAMOEReplay/DriveLM_llava_lora_MOE DriveLM_rs LoRAMOEReplay
# bash scripts/LLaVA/Eval_dom/1_eval_remotesensing.sh "" checkpoints/LLaVA/LoRAMOEReplay/FinVis_llava_lora_MOE Fin_rs LoRAMOEReplay
# # bash scripts/LLaVA/Eval_dom/1_eval_remotesensing.sh "" checkpoints/LLaVA/LoRAMOEReplay/Medical_llava_lora_MOE PathVQA_rs LoRAMOEReplay
# # bash scripts/LLaVA/Eval_dom/1_eval_remotesensing.sh "" checkpoints/LLaVA/LoRAMOEReplay/Sci_llava_lora_MOE Sci_rs LoRAMOEReplay
# # bash scripts/LLaVA/Eval_dom/1_eval_remotesensing.sh "" checkpoints/LLaVA/LoRAMOEReplay/RemoteSensing_llava_lora_MOE RS_rs LoRAMOEReplay
# bash scripts/LLaVA/Eval_dom/2_eval_medical.sh "" checkpoints/LLaVA/LoRAMOEReplay/FinVis_llava_lora_MOE Fin_pathvqa LoRAMOEReplay
# # bash scripts/LLaVA/Eval_dom/2_eval_medical.sh "" checkpoints/LLaVA/LoRAMOEReplay/Sci_llava_lora_MOE Sci_pathvqa LoRAMOEReplay
# # bash scripts/LLaVA/Eval_dom/2_eval_medical.sh "" checkpoints/LLaVA/LoRAMOEReplay/DriveLM_llava_lora_MOE DriveLM_pathvqa LoRAMOEReplay
# # bash scripts/LLaVA/Eval_dom/2_eval_medical.sh "" checkpoints/LLaVA/LoRAMOEReplay/Medical_llava_lora_MOE PathVQA_pathvqa LoRAMOEReplay
# bash scripts/LLaVA/Eval_dom/3_eval_drivelm.sh "" checkpoints/LLaVA/LoRAMOEReplay/FinVis_llava_lora_MOE Fin_drivelm LoRAMOEReplay
# # bash scripts/LLaVA/Eval_dom/3_eval_drivelm.sh "" checkpoints/LLaVA/LoRAMOEReplay/Sci_llava_lora_MOE Sci_drivelm LoRAMOEReplay
# # bash scripts/LLaVA/Eval_dom/3_eval_drivelm.sh "" checkpoints/LLaVA/LoRAMOEReplay/DriveLM_llava_lora_MOE DriveLM_drivelm LoRAMOEReplay
# bash scripts/LLaVA/Eval_dom/4_eval_sci.sh "" checkpoints/LLaVA/LoRAMOEReplay/FinVis_llava_lora_MOE Fin_sci LoRAMOEReplay
# # bash scripts/LLaVA/Eval_dom/4_eval_sci.sh "" checkpoints/LLaVA/LoRAMOEReplay/Sci_llava_lora_MOE Sci_sci LoRAMOEReplay
# bash scripts/LLaVA/Eval_dom/5_eval_financial.sh "" checkpoints/LLaVA/LoRAMOEReplay/FinVis_llava_lora_MOE Fin_fin LoRAMOEReplay
# bash scripts/LLaVA/Eval_ability/3_eval_VSR.sh "" checkpoints/LLaVA/AbilityReplay/Count_llava_lora Count_count AbilityReplay
# bash scripts/LLaVA/Eval_ability/3_eval_VSR.sh "" heckpoints/LLaVA/AbilityReplay/MATH_llava_lora Math_count AbilityReplay
# bash scripts/LLaVA/Eval_ability/3_eval_VSR.sh "" checkpoints/LLaVA/AbilityReplay/APP_llava_lora_replay APP_count AbilityReplay