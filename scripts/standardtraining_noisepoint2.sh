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
# -l gpu_type=A6000|A40|A100

# Specify the minimum GPU compute capability
#$ -l gpu_c=7.0

# merge error and sim_scores
#$ -j y

# Specify hard time limit (use default)
#$ -l h_rt=24:00:00

# get email when job begins
#$ -m beas

module load miniconda
conda deactivate
conda activate /projectnb/ivc-ml/siqiwang/anaconda3/envs/env
python main.py -c configs/CIFAR10/template.json --traintools train_synthesized --noise_ratio 0.4 --warmup 120 --noise_type asym --estimation_method BLTM --detection_method FINE+K --train_noise_method unicon --num_model 1 --seed 2023 --sccid=$JOB_ID