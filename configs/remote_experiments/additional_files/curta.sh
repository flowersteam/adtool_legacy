#!/bin/bash
sbatch <<EOT
#!/bin/sh
#SBATCH -p inria
#SBATCH -t 00:30:00
#SBATCH --job-name=auto_disc_experiment
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --array=0-$(($1-1))
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/gpfs/home/mperie/miniconda3/lib;
srun /gpfs/home/mperie/miniconda3/envs/autoDiscTool/bin/python auto_disc/auto_disc/run.py --seed \$SLURM_ARRAY_TASK_ID ${@:2}
EOT