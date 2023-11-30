#!/bin/bash

python main.py -d 0 \
-c configs/CIFAR10/tv_fineK_none.json --traintools train_synthesized --no_wandb \
--warmup 1