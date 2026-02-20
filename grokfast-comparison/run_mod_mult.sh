#!/bin/bash
#SBATCH --job-name=vanilla_1k_wd2
#SBATCH --output=vanilla_1k_wd2.txt
#SBATCH --error=vanilla_1k_wd2.txt
#SBATCH --ntasks=1
#SBATCH --time=2:00:00
#SBATCH --mem=10Gb
#SBATCH --gres=gpu:1 
#SBATCH -c 16

module load anaconda/3
conda activate grok
python main_modmult.py --label vanilla_1k_wd2
# python main_modmult.py --label grokfast --filter ema --alpha 0.98 --lamb 2.0 --weight_decay 0.005
# python main_modmult.py --label egd_svd --mode egd --weight_decay 0.005
# python main_modmult.py --label egd_rsvd_36 --mode egd --weight_decay 0.005 --rank_svd 36

