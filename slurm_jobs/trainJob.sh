#!/bin/bash
#SBATCH --job-name=gpt_training
#SBATCH --gres=gpu:2
#SBATCH --partition=amperenodes-medium
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=18:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

module load CUDA/12.1.1
module load cuDNN/8.9.2.26-CUDA-12.1.1
module load Anaconda3

conda activate generation

srun python3 $HOME/GPT-From-Scratch/main.py

conda deactivate
