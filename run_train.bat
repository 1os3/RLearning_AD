@echo off
call .\.venv\Scripts\Activate

python train.py --checkpoint f:\RLearning_AD_Wind\checkpoints\training\20250501-113950\

pause