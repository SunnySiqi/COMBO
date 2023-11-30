#!/bin/bash -l

# Set SCC project
#$ -P morphem

# Request num CPUs
#$ -pe omp 8
#$ -l mem_per_core=16G


# Request num GPU
#$ -l gpus=2

# mem for GPU
# -l gpu_memory=8G

# file name
#$ -o logs/$JOB_ID_$JOB_NAME.log

# Specify GPU type
#$ -l gpu_type=A6000|A40|A100|L40|RTX8000

# Specify the minimum GPU compute capability
# -l gpu_c=7.5

# merge error and sim_scores
#$ -j y

# Specify hard time limit (use default)
#$ -l h_rt=48:00:00

# get email when job begins
#$ -m beas

python -m memory_profiler main.py -c configs/clothing_test_spence.json --num_imgs 256000 --warmup 1  --traintools train_realworld \
	--estimation_method growing_cluster --detection_method UNICON+K --train_noise_method ours \
	--num_model 1 --seed 75909 --device 0,1
