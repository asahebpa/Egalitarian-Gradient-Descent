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
python main_mnist.py --label vanilla_1k_wd2 --weight_decay 2.0
# python main_mnist.py --label ema --alpha 0.8 --lamb 0.1 --weight_decay 2.0 --filter ema
# python main_mnist.py --label egd_svd --mode egd --weight_decay 2.0
# python main_mnist.py --label egd_rsvd_16 --mode egd --weight_decay 2.0 --rank_svd 16
# python main_mnist.py --label egd_rsvd_8 --mode egd --weight_decay 2.0 --rank_svd 8
# python main_mnist.py --label egd_rsvd_4 --mode egd --weight_decay 2.0 --rank_svd 4

