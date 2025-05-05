#!/bin/bash

#SBATCH --job-name=np08
#SBATCH --output=sbatchs/logs/num_08.log
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=24:00:00

mim train mmseg configs/sen1floods11_config_prompt_tuning_08.py