#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks=4
#SBATCH -t 00:10:00

log_dir="${3:-$HOME/qaoa_logs}"

mkdir -p "$HOME/hpc_runs" "$log_dir"
exec >"$log_dir/slurm-${SLURM_JOB_ID}.out" 2>&1

echo "[$(date)] Starting QAOA3 job ${SLURM_JOB_ID} on $(hostname)"
echo "Hamiltonian input: $1"
echo "Result output: $2"
echo "Log directory: $log_dir"

source "$HOME/QARPdemo/venv/bin/activate"


mpirun -np 4 python "$HOME/qaoa_jobs/qaoa_HPC_runner.py" "$1" "$2"
status=$?

echo "[$(date)] Finished QAOA3 job ${SLURM_JOB_ID} with status ${status}"
exit "${status}"
