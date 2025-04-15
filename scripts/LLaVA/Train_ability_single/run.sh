# bash scripts/LLaVA/Train_ability_single/1_SAT.sh 
# bash scripts/LLaVA/Train_ability_single/2_OCR.sh 
# bash scripts/LLaVA/Train_ability_single/3_math.sh 
# bash scripts/LLaVA/Train_ability_single/5_app.sh 

# eval
# bash scripts/LLaVA/Eval_ability/1_eval_OCR.sh
# bash scripts/LLaVA/Eval_ability/2_eval_VSR.sh
# bash scripts/LLaVA/Eval_ability/3_eval_math.sh
# bash scripts/LLaVA/Eval_ability/4_eval_APP.sh

# 交叉eval
# bash scripts/LLaVA/Eval_ability/1_eval_OCR.sh "" ./checkpoints/LLaVA/Ability/Count_llava_lora_visual-ep1_1e-5_ty_cl_sft Count_ocr
# bash scripts/LLaVA/Eval_ability/1_eval_OCR.sh "" ./checkpoints/LLaVA/Ability/MATH_llava_lora2e-5-visual-ep1 Math_ocr
# bash scripts/LLaVA/Eval_ability/1_eval_OCR.sh "" ./checkpoints/LLaVA/Ability/APP_llava_lora_visual_2e-5_ep3 APP_ocr

# bash scripts/LLaVA/Eval_ability/3_eval_math.sh "" ./checkpoints/LLaVA/Ability/Count_llava_lora_visual-ep1_1e-5_ty_cl_sft Count_math
# bash scripts/LLaVA/Eval_ability/3_eval_math.sh "" ./checkpoints/LLaVA/Ability/OCR_llava_lora-ep1-5e-6-visual-ep3-withoutdes OCR_math
# bash scripts/LLaVA/Eval_ability/3_eval_math.sh "" ./checkpoints/LLaVA/Ability/APP_llava_lora_visual_2e-5_ep3 APP_math

# bash scripts/LLaVA/Eval_ability/4_eval_APP.sh "" ./checkpoints/LLaVA/Ability/Count_llava_lora_visual-ep1_1e-5_ty_cl_sft Count_app
# bash scripts/LLaVA/Eval_ability/4_eval_APP.sh "" ./checkpoints/LLaVA/Ability/MATH_llava_lora2e-5-visual-ep1 Math_app
# bash scripts/LLaVA/Eval_ability/4_eval_APP.sh "" ./checkpoints/LLaVA/Ability/OCR_llava_lora-ep1-5e-6-visual-ep3-withoutdes OCR_app

# bash scripts/LLaVA/Eval_ability/2_eval_VSR.sh "" ./checkpoints/LLaVA/Ability/OCR_llava_lora-ep1-5e-6-visual-ep3-withoutdes OCR_count
# bash scripts/LLaVA/Eval_ability/2_eval_VSR.sh "" ./checkpoints/LLaVA/Ability/MATH_llava_lora2e-5-visual-ep1 Math_count
# bash scripts/LLaVA/Eval_ability/2_eval_VSR.sh "" ./checkpoints/LLaVA/Ability/APP_llava_lora_visual_2e-5_ep3 APP_count

# # continual FT
# # bash scripts/LLaVA/Train_ability/1_OCR.sh  Ability-CL
bash scripts/LLaVA/Train_ability/2_math.sh Ability-CL True
bash scripts/LLaVA/Train_ability/3_SAT.sh  Ability-CL True
bash scripts/LLaVA/Train_ability/4_app.sh  Ability-CL True

# # continual FT MOE
bash scripts/LLaVA/Train_ability/1_OCR.sh  Ability-CL-MOE "" 4
bash scripts/LLaVA/Train_ability/2_math.sh Ability-CL-MOE True 4
bash scripts/LLaVA/Train_ability/3_app.sh  Ability-CL-MOE True 4
bash scripts/LLaVA/Train_ability/4_SAT.sh  Ability-CL-MOE True 4