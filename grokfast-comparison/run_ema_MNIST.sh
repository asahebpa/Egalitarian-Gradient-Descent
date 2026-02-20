#!/bin/bash
#SBATCH --job-name=ema_1k_wdem3
#SBATCH --output=ema_1k_wdem3.txt
#SBATCH --error=ema_1k_wdem3.txt
#SBATCH --ntasks=1
#SBATCH --time=4:00:00
#SBATCH --mem=10Gb
#SBATCH --gres=gpu:1 
#SBATCH -c 16

module load anaconda/3
conda activate grok
python main_mnist.py --label ema_1k__wdem3 --alpha 0.8 --lamb 0.1 --filter ema
# python main_mnist.py --label ema_1k --alpha 0.8 --lamb 0.1 --weight_decay 2.0 --filter ema


