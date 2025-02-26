# /bin/bash

# bash scripts/LLaVA/Train/3_ImageNet.sh
# wait
# bash scripts/LLaVA/Train/4_GQA.sh
# wait
# bash scripts/LLaVA/Train/5_VizWiz.sh
# wait
# bash scripts/LLaVA/Train/6_Grounding.sh
# wait
# bash scripts/LLaVA/Train/7_vqav2.sh
# wait
# bash scripts/LLaVA/Train/8_OCRVQA.sh

bash scripts/LLaVA/Train/9_Mov.sh ScienceQA
wait
bash scripts/LLaVA/Train/9_Mov.sh TextVQA
wait
bash scripts/LLaVA/Train/9_Mov.sh ImageNet
wait
bash scripts/LLaVA/Train/9_Mov.sh GQA
wait
bash scripts/LLaVA/Train/9_Mov.sh VizWiz
wait
bash scripts/LLaVA/Train/9_Mov.sh Grounding
wait
bash scripts/LLaVA/Train/9_Mov.sh VQAv2
wait
bash scripts/LLaVA/Train/9_Mov.sh OCR-VQA