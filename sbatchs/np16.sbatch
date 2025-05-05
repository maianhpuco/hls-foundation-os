#!/bin/bash

#SBATCH --job-name=np16
#SBATCH --output=sbatchs/logs/num_16.log
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=24:00:00

mim train mmseg configs/sen1floods11_config_prompt_tuning_16.py