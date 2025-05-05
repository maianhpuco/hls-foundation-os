#!/bin/bash

#SBATCH --job-name=full_25
#SBATCH --output=sbatchs/logs/full_25.log
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=24:00:00

mim train mmseg configs/sen1flood11_config_25.py