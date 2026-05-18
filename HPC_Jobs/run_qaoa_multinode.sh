#!/bin/bash
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH -t 02:00:00

log_dir="${3:-$HOME/qaoa_logs}"

mkdir -p "$HOME/hpc_runs" "$log_dir"
exec >"$log_dir/slurm-${SLURM_JOB_ID}.out" 2>&1

echo "[$(date)] Starting multi-node QAOA job ${SLURM_JOB_ID}"
echo "Nodes: ${SLURM_JOB_NUM_NODES}"
echo "Tasks: ${SLURM_NTASKS}"
echo "Hamiltonian input: $1"
echo "Result output: $2"
echo "Log directory: $log_dir"

source "$HOME/QARPdemo/venv/bin/activate"

export QAOA_REPS="${QAOA_REPS:-1}"
export QAOA_MAXITER="${QAOA_MAXITER:-20}"
export QAOA_SHOTS="${QAOA_SHOTS:-2048}"
export QAOA_SPSA_LEARNING_RATE="${QAOA_SPSA_LEARNING_RATE:-0.4}"
export QAOA_SPSA_PERTURBATION="${QAOA_SPSA_PERTURBATION:-0.2}"
export QAOA_SPSA_ALPHA="${QAOA_SPSA_ALPHA:-0.602}"
export QAOA_SPSA_GAMMA="${QAOA_SPSA_GAMMA:-0.101}"
export QAOA_RANDOM_SEED="${QAOA_RANDOM_SEED:-42}"

echo "QAOA config: reps=${QAOA_REPS}, maxiter=${QAOA_MAXITER}, shots=${QAOA_SHOTS}, optimizer=SPSA"
echo "MPI command: mpirun -N 1 -npernode 1 -n ${SLURM_NTASKS} python qaoa_HPC_runner_multinode.py"

mpirun -N 1 -npernode 1 -n "${SLURM_NTASKS}" \
    python "$HOME/qaoa_jobs/qaoa_HPC_runner_multinode.py" "$1" "$2"
status=$?

echo "[$(date)] Finished multi-node QAOA job ${SLURM_JOB_ID} with status ${status}"
exit "${status}"
