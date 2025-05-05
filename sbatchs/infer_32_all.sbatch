#!/bin/bash

#SBATCH --job-name=infer_32
#SBATCH --output=sbatchs/logs/infer_32.log
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=24:00:00

make infer_32_all