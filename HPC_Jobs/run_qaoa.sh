#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks=4
#SBATCH -t 00:10:00
#SBATCH -o ~/hpc_runs/qaoa-%j.out

source ~/QARPdemo/venv/bin/activate

mpirun -np 4 python ~/qaoa_jobs/qaoa_hpc_runner.py $1 $2