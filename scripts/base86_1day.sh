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
#$ -l gpu_type=A6000|A40|A100

# Specify the minimum GPU compute capability
# -l gpu_c=7.5

# merge error and sim_scores
#$ -j y

# Specify hard time limit (use default)
#$ -l h_rt=24:00:00

# get email when job begins
#$ -m beas

module load miniconda
conda deactivate
conda activate /projectnb/ivc-ml/siqiwang/anaconda3/envs/env