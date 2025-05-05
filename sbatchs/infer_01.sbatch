#!/bin/bash

#SBATCH --job-name=infer_01
#SBATCH --output=sbatchs/logs/infer_01.log
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=24:00:00

make infer_01