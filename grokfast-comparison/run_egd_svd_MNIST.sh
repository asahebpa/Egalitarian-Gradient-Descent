#!/bin/bash
#SBATCH --job-name=egd_svd_1k_wdem3
#SBATCH --output=egd_svd_1k_wdem3.txt
#SBATCH --error=egd_svd_1k_wdem3.txt
#SBATCH --ntasks=1
#SBATCH --time=4:00:00
#SBATCH --mem=10Gb
#SBATCH --gres=gpu:1 
#SBATCH -c 16

module load anaconda/3
conda activate grok
python main_mnist.py --label egd_svd_1k_wdem3 --mode egd
# python main_mnist.py --label egd_svd_1k_wdem3 --mode egd --weight_decay 2.0


