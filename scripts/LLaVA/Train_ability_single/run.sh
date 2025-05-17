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
# bash scripts/LLaVA/Eval_ability/1_eval_OCR.sh "" ./checkpoints/LLaVA/Ability_vision/APP_llava_lora APP_ocr VisionAbi
# bash scripts/LLaVA/Eval_ability/1_eval_OCR.sh "" ./checkpoints/LLaVA/Ability/MATH_llava_lora2e-5-visual-ep1 Math_ocr AbiDebug
# bash scripts/LLaVA/Eval_ability/1_eval_OCR.sh "" ./checkpoints/LLaVA/Ability_vision/Count_llava_lora Count_ocr VisionAbi
# bash scripts/LLaVA/Eval_ability/1_eval_OCR.sh "" ./checkpoints/LLaVA/Ability_vision/OCR_llava_lora OCR_ocr VisionAbi

# bash scripts/LLaVA/Eval_ability/2_eval_math.sh "" ./checkpoints/LLaVA/Ability_vision/APP_llava_lora APP_math VisionAbi
# bash scripts/LLaVA/Eval_ability/2_eval_math.sh "" ./checkpoints/LLaVA/Ability_vision/MATH_llava_lora Math_math VisionAbi
# bash scripts/LLaVA/Eval_ability/2_eval_math.sh "" ./checkpoints/LLaVA/Ability_vision/Count_llava_lora Count_math VisionAbi
# bash scripts/LLaVA/Eval_ability/2_eval_math.sh "" ./checkpoints/LLaVA/Ability_vision/OCR_llava_lora OCR_math VisionAbi

# bash scripts/LLaVA/Eval_ability/3_eval_VSR.sh "" ./checkpoints/LLaVA/Ability_vision/APP_llava_lora APP_count VisionAbi
# bash scripts/LLaVA/Eval_ability/3_eval_VSR.sh "" ./checkpoints/LLaVA/Ability_vision/MATH_llava_lora Math_count VisionAbi
# bash scripts/LLaVA/Eval_ability/3_eval_VSR.sh "" ./checkpoints/LLaVA/Ability_vision/Count_llava_lora Count_count VisionAbi
# bash scripts/LLaVA/Eval_ability/3_eval_VSR.sh "" ./checkpoints/LLaVA/Ability_vision/OCR_llava_lora OCR_count VisionAbi

# bash scripts/LLaVA/Eval_ability/4_eval_APP.sh "" ./checkpoints/LLaVA/Ability_vision/APP_llava_lora APP_app VisionAbi
# bash scripts/LLaVA/Eval_ability/4_eval_APP.sh "" ./checkpoints/LLaVA/Ability_vision/MATH_llava_lora Math_app VisionAbi
# bash scripts/LLaVA/Eval_ability/4_eval_APP.sh "" ./checkpoints/LLaVA/Ability_vision/Count_llava_lora Count_app VisionAbi
# bash scripts/LLaVA/Eval_ability/4_eval_APP.sh "" ./checkpoints/LLaVA/Ability_vision/OCR_llava_lora OCR_app VisionAbi
# # continual FT
# # bash scripts/LLaVA/Train_ability/1_OCR.sh  Ability-CL
# bash scripts/LLaVA/Train_ability/2_math.sh Ability-CL True
# bash scripts/LLaVA/Train_ability/3_SAT.sh  Ability-CL True
# bash scripts/LLaVA/Train_ability/4_app.sh  Ability-CL True

# # continual FT MOE
# bash scripts/LLaVA/Train_ability/1_OCR.sh  Ability-CL-MOE "" 4
# bash scripts/LLaVA/Train_ability/2_math.sh Ability-CL-MOE True 4
# bash scripts/LLaVA/Train_ability/3_SAT.sh  Ability-CL-MOE True 4
# bash scripts/LLaVA/Train_ability/4_app.sh  Ability-CL-MOE True 4
# bash scripts/LLaVA/Train_ability/5_app_replay.sh  Ability-CL-MOE True 4

# bash scripts/LLaVA/Train_ability/5_app_replay.sh  Ability-CL True