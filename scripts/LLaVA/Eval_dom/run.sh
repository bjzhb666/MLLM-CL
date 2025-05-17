# bash scripts/LLaVA/Eval_dom/4_eval_sci.sh "" checkpoints/LLaVA/CoIN/Sci_llava_lora_debug Sci_debug SciDebug
# bash scripts/LLaVA/Eval_dom/4_eval_sci.sh "" checkpoints/LLaVA/CoIN/Sci_llava_lora_debuglr2e-4 Sci_debuglr2e-4 SciDebug
# bash scripts/LLaVA/Eval_dom/4_eval_sci.sh "" checkpoints/LLaVA/CoIN/Sci_llava_lora_debugep3 Sci_debugeq3 SciDebug
# export CUDA_VISIBLE_DEVICES=0
# bash scripts/LLaVA/Train_ability_single/1_SAT.sh
# bash scripts/LLaVA/Eval_ability/3_eval_VSR.sh ""  checkpoints/LLaVA/Ability/Count_llava_lora_visual-ep1_2e-4_ty_cl_sft Count_count2e-4 DebugAbiTest

# bash scripts/LLaVA/Train_dom_single/12_Sci.sh
# bash scripts/LLaVA/Eval_dom/4_eval_sci.sh "" checkpoints/LLaVA/CoIN/Sci_llava_lora_ep3 Sci_debug_0ep3 SciDebug

# bash scripts/LLaVA/Eval_ability/4_eval_APP.sh "" checkpoints/LLaVA/AbiLoRAMOEReplay/APP_llava_lora_MOE APP_app AbiLoRAMOEReplay
# bash scripts/LLaVA/Eval_ability/3_eval_VSR.sh "" checkpoints/LLaVA/AbiLoRAMOEReplay/APP_llava_lora_MOE APP_count AbiLoRAMOEReplay
# bash scripts/LLaVA/Eval_ability/1_eval_OCR.sh "" checkpoints/LLaVA/AbiLoRAMOEReplay/APP_llava_lora_MOE APP_ocr AbiLoRAMOEReplay
# bash scripts/LLaVA/Eval_ability/2_eval_math.sh "" checkpoints/LLaVA/AbiLoRAMOEReplay/APP_llava_lora_MOE APP_math AbiLoRAMOEReplay

# bash scripts/LLaVA/Eval_ability/2_eval_math.sh ""  checkpoints/LLaVA/AbiLoRAMOEReplay/Count_llava_lora_MOE Count_math AbiLoRAMOEReplay
# bash scripts/LLaVA/Eval_ability/2_eval_math.sh "" checkpoints/LLaVA/AbiLoRAMOEReplay/MATH_llava_lora_MOE MATH_math AbiLoRAMOEReplay

# bash scripts/LLaVA/Eval_ability/2_eval_math.sh "" checkpoints/LLaVA/AbilityReplay/APP_llava_lora_replay APP_math AbilityReplay
# bash scripts/LLaVA/Eval_ability/2_eval_math.sh "" checkpoints/LLaVA/AbilityReplay/Count_llava_lora Count_math AbilityReplay
# bash scripts/LLaVA/Eval_ability/2_eval_math.sh "" checkpoints/LLaVA/AbilityReplay/MATH_llava_lora MATH_math AbilityReplay