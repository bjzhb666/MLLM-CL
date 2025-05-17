
bash scripts/LLaVA/Train_ability/2_math.sh Ability_vision True
bash scripts/LLaVA/Train_ability/3_SAT.sh Ability_vision  True
bash scripts/LLaVA/Train_ability/4_app.sh Ability_vision  True
bash scripts/LLaVA/Train_ability_single/run.sh
bash scripts/LLaVA/Train_ability_single/0_train_router.sh
# bash scripts/LLaVA/Train_ability/2_math_re.sh AbiLoRAMOEReplay True 4
# bash scripts/LLaVA/Train_ability/3_SAT_re.sh AbiLoRAMOEReplay True 4
# bash scripts/LLaVA/Train_ability/4_app_replay.sh AbiLoRAMOEReplay True 4