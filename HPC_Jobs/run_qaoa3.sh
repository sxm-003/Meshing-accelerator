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

export QAOA_REPS="${QAOA_REPS:-2}"
export QAOA_MAXITER="${QAOA_MAXITER:-50}"
export QAOA_SHOTS="${QAOA_SHOTS:-4096}"
export QAOA_SPSA_LEARNING_RATE="${QAOA_SPSA_LEARNING_RATE:-0.4}"
export QAOA_SPSA_PERTURBATION="${QAOA_SPSA_PERTURBATION:-0.2}"
export QAOA_SPSA_ALPHA="${QAOA_SPSA_ALPHA:-0.602}"
export QAOA_SPSA_GAMMA="${QAOA_SPSA_GAMMA:-0.101}"
export QAOA_RANDOM_SEED="${QAOA_RANDOM_SEED:-42}"

echo "QAOA config: reps=${QAOA_REPS}, maxiter=${QAOA_MAXITER}, shots=${QAOA_SHOTS}, optimizer=SPSA"


mpirun -np 4 python "$HOME/qaoa_jobs/qaoa_HPC_runner.py" "$1" "$2"
status=$?

echo "[$(date)] Finished QAOA3 job ${SLURM_JOB_ID} with status ${status}"
exit "${status}"
