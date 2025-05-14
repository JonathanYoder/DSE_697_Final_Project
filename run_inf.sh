#!/bin/bash

#SBATCH -A trn040
#SBATCH -J inference
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 01:00:00
#SBATCH -N 1

unset SLURM_EXPORT_ENV

module load PrgEnv-gnu/8.6.0
module load miniforge3/23.11.0
module load rocm/6.2.4
module load craype-accel-amd-gfx90a
source activate /gpfs/wolf2/olcf/trn040/scratch/kmn3/envs/hf-transformers1

working_dir="/gpfs/wolf2/olcf/trn040/scratch/kmn3"
cd "${working_dir}"

python3 inference.py