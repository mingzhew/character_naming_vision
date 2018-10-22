#!/bin/bash
#
#SBATCH --job-name=test_job
#SBATCH --output=test_output.txt
#
#SBATCH --mail-user=cvprking@umich.edu
#SBATCH --mail-type=ALL
#
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=1096
#SBATCH --gres=gpu:0
#SBATCH --time=01-00:00:00
#
#SBATCH --workdir=/

# Put your job commands after this line
srun hostname
srun echo "Hello, world."
srun sleep 60
