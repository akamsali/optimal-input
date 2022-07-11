#!/bin/sh -l
# FILENAME: my_job

#SBATCH -A partner
#SBATCH --nodes=1 --gpus-per-node=1 --mem=49GB
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/gilbreth/akamsali/Research/Makin/optimal-input/experiments/outputs.txt
#SBATCH --error=/scratch/gilbreth/akamsali/Research/Makin/optimal-input/experiments/error.txt
#SBATCH --job-name W2L

# Print the hostname of the compute node on which this job is running.
/bin/hostname

. ~/.bashrc
conda activate research

python /scratch/gilbreth/akamsali/Research/Makin/optimal-input/experiments/w2l_opt.py

