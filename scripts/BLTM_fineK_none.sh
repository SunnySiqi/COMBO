#!/bin/bash
source /projectnb/ivc-ml/siqiwang/anaconda3/etc/profile.d/conda.sh
conda activate env
python main.py -d 9 \
-c configs/CIFAR10/BLTM_fineK_none.json --traintools train_synthesized --no_wandb \
--warmup 1