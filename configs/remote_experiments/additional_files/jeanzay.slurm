#!/bin/bash

#SBATCH -A imi@cpu
#SBATCH --qos=qos_cpu-t3
#SBATCH --time=00:01:00
#SBATCH --input=none
#SBATCH --mail-user=mayalen.etcheverry@inria.fr
#SBATCH -o run_experiment.slurm.out
#SBATCH -e run_experiment.slurm.err