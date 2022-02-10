#!/bin/bash
sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=auto_disc_experiment # job name
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --time=03:00:00             # maximum execution time (HH:MM:SS)
#SBATCH --output=%x-%j.out           # output file name
#SBATCH --error=%x-%j.err           # output file name
#SBATCH --account=imi@gpu
#SBATCH --qos=qos_gpu-t3 # dev
#SBATCH --gres=gpu:1
#SBATCH --array=0-$(($1-1))

module purge
module load python/3.7.6
conda activate autoDiscTool

srun python libs/auto_disc/run.py --seed \$SLURM_ARRAY_TASK_ID ${@:2}
EOT