#!/bin/bash

#SBATCH --job-name=infer_pc_all
#SBATCH --output=sbatchs/logs/infer_pc_all.log
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=24:00:00

make infer_pc_all