#!/bin/bash

#SBATCH --job-name=np32_50
#SBATCH --output=sbatchs/logs/num_32_50.log
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=24:00:00

mim train mmseg configs/sen1floods11_config_prompt_tuning_32_train_50.py