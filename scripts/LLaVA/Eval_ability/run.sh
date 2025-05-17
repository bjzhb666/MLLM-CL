# LLaVA1.5 test
# bash scripts/LLaVA/Eval_ability/1_eval_OCR.sh "" checkpoints/LLaVA/Vicuna/llava-7b-v1.5 llava_ocr
# bash scripts/LLaVA/Eval_ability/3_eval_math.sh "" checkpoints/LLaVA/Vicuna/llava-7b-v1.5 llava_math
# bash scripts/LLaVA/Eval_ability/2_eval_VSR.sh "" checkpoints/LLaVA/Vicuna/llava-7b-v1.5 llava_sat
# bash scripts/LLaVA/Eval_ability/4_eval_APP.sh "" checkpoints/LLaVA/Vicuna/llava-7b-v1.5 llava_app

# LoRA FT Test
# bash scripts/LLaVA/Eval_ability/1_eval_OCR.sh "" checkpoints/LLaVA/Ability-CL/OCR_llava_lora OCR_ocr LoRAMatrix
# bash scripts/LLaVA/Eval_ability/1_eval_OCR.sh "" checkpoints/LLaVA/Ability-CL/Count_llava_lora Count_ocr LoRAMatrix
# bash scripts/LLaVA/Eval_ability/1_eval_OCR.sh "" checkpoints/LLaVA/Ability-CL/MATH_llava_lora Math_ocr LoRAMatrix
# bash scripts/LLaVA/Eval_ability/1_eval_OCR.sh "" checkpoints/LLaVA/Ability-CL/APP_llava_lora APP_ocr LoRAMatrix
# bash scripts/LLaVA/Eval_ability/2_eval_VSR.sh "" checkpoints/LLaVA/Ability-CL/Count_llava_lora Count_count LoRAMatrix
# bash scripts/LLaVA/Eval_ability/2_eval_VSR.sh "" checkpoints/LLaVA/Ability-CL/MATH_llava_lora Math_count LoRAMatrix
# bash scripts/LLaVA/Eval_ability/2_eval_VSR.sh "" checkpoints/LLaVA/Ability-CL/APP_llava_lora APP_count LoRAMatrix
# bash scripts/LLaVA/Eval_ability/3_eval_math.sh "" checkpoints/LLaVA/Ability-CL/MATH_llava_lora Math_math LoRAMatrix
# bash scripts/LLaVA/Eval_ability/3_eval_math.sh "" checkpoints/LLaVA/Ability-CL/APP_llava_lora APP_math LoRAMatrix
# bash scripts/LLaVA/Eval_ability/4_eval_APP.sh "" checkpoints/LLaVA/Ability-CL/APP_llava_lora APP_app LoRAMatrix
# bash scripts/LLaVA/Eval_ability/3_eval_math.sh "" checkpoints/LLaVA/Ability-CL/Count_llava_lora Count_math LoRAMatrix

# LoRAMoE Test
# bash scripts/LLaVA/Eval_ability/1_eval_OCR.sh "" checkpoints/LLaVA/Ability-CL-MOE/OCR_llava_lora_MOE OCR_ocr LoRAMoEMatrix
# bash scripts/LLaVA/Eval_ability/1_eval_OCR.sh "" checkpoints/LLaVA/Ability-CL-MOE/Count_llava_lora_MOE Count_ocr LoRAMoEMatrix
# bash scripts/LLaVA/Eval_ability/1_eval_OCR.sh "" checkpoints/LLaVA/Ability-CL-MOE/MATH_llava_lora_MOE Math_ocr LoRAMoEMatrix
# bash scripts/LLaVA/Eval_ability/1_eval_OCR.sh "" checkpoints/LLaVA/Ability-CL-MOE/APP_llava_lora_MOE APP_ocr LoRAMoEMatrix
# bash scripts/LLaVA/Eval_ability/2_eval_VSR.sh "" checkpoints/LLaVA/Ability-CL-MOE/Count_llava_lora_MOE Count_count LoRAMoEMatrix
# bash scripts/LLaVA/Eval_ability/2_eval_VSR.sh "" checkpoints/LLaVA/Ability-CL-MOE/MATH_llava_lora_MOE Math_count LoRAMoEMatrix
# bash scripts/LLaVA/Eval_ability/2_eval_VSR.sh "" checkpoints/LLaVA/Ability-CL-MOE/APP_llava_lora_MOE APP_count LoRAMoEMatrix
# bash scripts/LLaVA/Eval_ability/3_eval_math.sh "" checkpoints/LLaVA/Ability-CL-MOE/MATH_llava_lora_MOE Math_math LoRAMoEMatrix
# bash scripts/LLaVA/Eval_ability/3_eval_math.sh "" checkpoints/LLaVA/Ability-CL-MOE/APP_llava_lora_MOE APP_math LoRAMoEMatrix
# bash scripts/LLaVA/Eval_ability/4_eval_APP.sh "" checkpoints/LLaVA/Ability-CL-MOE/APP_llava_lora_MOE APP_app LoRAMoEMatrix
# bash scripts/LLaVA/Eval_ability/3_eval_math.sh "" checkpoints/LLaVA/Ability-CL-MOE/Count_llava_lora_MOE Count_math LoRAMoEMatrix

# Test router
# bash scripts/LLaVA/Eval_ability/0_eval_router.sh OCR_test ocr 
# bash scripts/LLaVA/Eval_ability/0_eval_router.sh APP_test app  
# bash scripts/LLaVA/Eval_ability/0_eval_router.sh math_test math
# bash scripts/LLaVA/Eval_ability/0_eval_router.sh CV-Benchv2 count 

# bash scripts/LLaVA/Eval_ability/0_eval_router.sh OCR_test ocr ep20 checkpoints/Router_Ability/Router_llava_lora_5e-6-ep20
# bash scripts/LLaVA/Eval_ability/0_eval_router.sh APP_test app  ep20 checkpoints/Router_Ability/Router_llava_lora_5e-6-ep20
# bash scripts/LLaVA/Eval_ability/0_eval_router.sh math_test math ep20 checkpoints/Router_Ability/Router_llava_lora_5e-6-ep20
# bash scripts/LLaVA/Eval_ability/0_eval_router.sh CV-Benchv2 count  ep20  checkpoints/Router_Ability/Router_llava_lora_5e-6-ep20

# bash scripts/LLaVA/Eval_ability/0_eval_router.sh OCR_test ocr ep35 checkpoints/Router_Ability/Router_llava_lora_5e-6-ep35
# bash scripts/LLaVA/Eval_ability/0_eval_router.sh APP_test app  ep35 checkpoints/Router_Ability/Router_llava_lora_5e-6-ep35
# bash scripts/LLaVA/Eval_ability/0_eval_router.sh math_test math ep35 checkpoints/Router_Ability/Router_llava_lora_5e-6-ep35
# bash scripts/LLaVA/Eval_ability/0_eval_router.sh CV-Benchv2 count  ep35  checkpoints/Router_Ability/Router_llava_lora_5e-6-ep35

# bash scripts/LLaVA/Eval_ability/0_eval_router.sh OCR_test ocr ep10 checkpoints/Router_Ability/task4-Router_llava_lora_5e-6-ep10
# bash scripts/LLaVA/Eval_ability/0_eval_router.sh APP_test app  ep10 checkpoints/Router_Ability/task4-Router_llava_lora_5e-6-ep10
# bash scripts/LLaVA/Eval_ability/0_eval_router.sh math_test math ep10 checkpoints/Router_Ability/task4-Router_llava_lora_5e-6-ep10
# bash scripts/LLaVA/Eval_ability/0_eval_router.sh CV-Benchv2 count ep10 checkpoints/Router_Ability/task4-Router_llava_lora_5e-6-ep10

# bash scripts/LLaVA/Eval_ability/0_eval_router.sh OCR_test ocr sample100ep12 checkpoints/Router_Ability/task4-Router_llava_lora_5e-6-ep12-sample100
# bash scripts/LLaVA/Eval_ability/0_eval_router.sh APP_test app  sample100ep12 checkpoints/Router_Ability/task4-Router_llava_lora_5e-6-ep12-sample100
# bash scripts/LLaVA/Eval_ability/0_eval_router.sh math_test math sample100ep12 checkpoints/Router_Ability/task4-Router_llava_lora_5e-6-ep12-sample100
# bash scripts/LLaVA/Eval_ability/0_eval_router.sh CV-Benchv2 count  sample100ep12 checkpoints/Router_Ability/task4-Router_llava_lora_5e-6-ep12-sample100

# bash scripts/LLaVA/Eval_ability/0_eval_router.sh OCR_test ocr sample100ep6 checkpoints/Router_Ability/task4-Router_llava_lora_5e-6-ep6-sample100
# bash scripts/LLaVA/Eval_ability/0_eval_router.sh APP_test app  sample100ep6 checkpoints/Router_Ability/task4-Router_llava_lora_5e-6-ep6-sample100
# bash scripts/LLaVA/Eval_ability/0_eval_router.sh math_test math sample100ep6 checkpoints/Router_Ability/task4-Router_llava_lora_5e-6-ep6-sample100
# bash scripts/LLaVA/Eval_ability/0_eval_router.sh CV-Benchv2 count  sample100ep6 checkpoints/Router_Ability/task4-Router_llava_lora_5e-6-ep6-sample100

# bash scripts/LLaVA/Eval_ability/2_eval_math.sh "" checkpoints/LLaVA/Ability/MATH_llava_lora2e-5-visual-ep1 Math_math TestMath

# bash scripts/LLaVA/Eval_ability/0_eval_router.sh math_test math sample20task2 checkpoints/Router_Ability/task2-Router_llava_lora_5e-6-ep35-sample20
# bash scripts/LLaVA/Eval_ability/0_eval_router.sh math_test math sample20task3 checkpoints/Router_Ability/task3-Router_llava_lora_5e-6-ep35-sample20
# bash scripts/LLaVA/Eval_ability/0_eval_router.sh OCR_test ocr sample20task2 checkpoints/Router_Ability/task3-Router_llava_lora_5e-6-ep35-sample20
# bash scripts/LLaVA/Eval_ability/0_eval_router.sh OCR_test ocr sample20task3 checkpoints/Router_Ability/task2-Router_llava_lora_5e-6-ep35-sample20
bash scripts/LLaVA/Eval_ability/0_eval_router.sh CV-Benchv2 count sample20task3 checkpoints/Router_Ability/task3-Router_llava_lora_5e-6-ep35-sample20