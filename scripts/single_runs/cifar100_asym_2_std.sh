#!/bin/bash -l

# Set SCC project
#$ -P morphem

# Request num CPUs
#$ -pe omp 3
# -l mem_per_core=8G


# Request num GPU
#$ -l gpus=1

# mem for GPU
# -l gpu_memory=8G

# file name
#$ -o logs/$JOB_ID_$JOB_NAME.log

# Specify GPU type
# -l gpu_type=A6000|L40|RTX6000|RTX8000

# Specify the minimum GPU compute capability
#$ -l gpu_c=7.0

# merge error and sim_scores
#$ -j y

# Specify hard time limit (use default)
#$ -l h_rt=24:00:00

# get email when job begins
#$ -m beas

source /projectnb/ivc-ml/siqiwang/anaconda3/etc/profile.d/conda.sh
conda activate env
python main.py -c configs/cifar100.json --traintools train_synthesized --noise_ratio 0.2 --noise_type asym --estimation_method none --detection_method none --train_noise_method none --warmup 150
