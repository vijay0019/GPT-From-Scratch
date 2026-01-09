#!/bin/bash
#SBATCH --job-name=train_data_download
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --partition=pascalnodes
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

module load CUDA/12.1.1
module load cuDNN/8.9.2.26-CUDA-12.1.1
module load Anaconda3

conda activate generation

python $HOME/GPT-From-Scratch/finewebedu.py

conda deactivate