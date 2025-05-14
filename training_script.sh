#!/bin/bash

#SBATCH -A trn040
#SBATCH -J training
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 10:00:00
#SBATCH -N 1

unset SLURM_EXPORT_ENV

module load PrgEnv-gnu/8.6.0
module load miniforge3/23.11.0
module load rocm/6.2.4
module load craype-accel-amd-gfx90a

source activate /gpfs/wolf2/olcf/trn040/scratch/kmn3/envs/hf-transformers1


working_dir="/gpfs/wolf2/olcf/trn040/scratch/kmn3"
cd "${working_dir}"

echo "Running on node: $(hostname)"
echo "Python3 executable path: $(which python3)"
echo "Current working directory: $(pwd)"

python3 train_model.py