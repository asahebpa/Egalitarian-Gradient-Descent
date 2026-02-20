#!/bin/bash
#SBATCH --job-name=both_egd_emad_rsvd16_1k
#SBATCH --output=both_egd_emad_rsvd16_1k.txt
#SBATCH --error=both_egd_emad_rsvd16_1k.txt
#SBATCH --ntasks=1
#SBATCH --time=4:00:00
#SBATCH --mem=10Gb
#SBATCH --gres=gpu:1 
#SBATCH -c 16

module load anaconda/3
conda activate grok
python main_mnist.py --label both_egd_ema_rsvd_16_1k --mode egd --rank_svd 16 --alpha 0.8 --lamb 0.1 --filter ema --weight_decay 2.0
# python main_mnist.py --label ddegd_rsvd_8_1k --mode egd --weight_decay 2.0 --rank_svd 8
