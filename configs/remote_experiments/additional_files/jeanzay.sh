#!/bin/bash
sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=auto_disc_experiment # job name
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --time=00:10:00             # maximum execution time (HH:MM:SS)
#SBATCH --output=%x-%j.out           # output file name
#SBATCH --error=%x-%j.err           # output file name
#SBATCH --account=imi@cpu
#SBATCH --qos=qos_cpu-dev # t3
#SBATCH --array=0-$(($1-1))

module purge
module load python/3.11
conda activate autoDiscTool

srun python auto_disc/auto_disc/run.py --seed \$SLURM_ARRAY_TASK_ID ${@:2}
EOT