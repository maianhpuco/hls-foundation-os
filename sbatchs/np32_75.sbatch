#!/bin/bash

#SBATCH --job-name=np32_75
#SBATCH --output=sbatchs/logs/num_32_75.log
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=24:00:00

mim train mmseg configs/sen1floods11_config_prompt_tuning_32_train_75.py