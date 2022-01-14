#!/bin/bash

#SBATCH -p inria
#SBATCH --time=00:01:00
#SBATCH -o run_experiment.slurm-%j.out
#SBATCH -e run_experiment.slurm-%j.err
#SBATCH array=0-$1

module purge
module load python/3.7.1
conda activate autoDiscTool

srun python run.py --seed $SLURM_ARRAY_TASK_ID ${@:2} &