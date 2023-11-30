#!/bin/bash
source /projectnb/ivc-ml/siqiwang/anaconda3/etc/profile.d/conda.sh
conda activate env
python main.py -d 1 \
-c configs/CIFAR10/dualT_fineK_none.json --traintools train_synthesized --no_wandb \
--warmup 1
 
