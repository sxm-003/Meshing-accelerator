# Multi-node QAOA Candidate

These files are an experimental multi-node path and do not replace the current
one-node runner:

- `run_qaoa_multinode.sh`
- `qaoa_HPC_runner_multinode.py`
- `qaoa_qulacs_mpi_observable.py`

## Why this path exists

The current one-node runner builds full `2^n` arrays for basis energies and
probabilities. That makes 28+ qubits painful even if the Slurm job requests more
nodes.

The multi-node candidate changes the QAOA evaluator to:

- build a Qulacs `Observable` from the Hamiltonian;
- use `observable.get_expectation_value(state)` instead of a full
  `basis_energies` vector;
- make every MPI rank participate in each QAOA state simulation;
- sample only at the end and evaluate sampled bitstrings directly.

This is much closer to the distributed-statevector execution pattern needed for
larger qubit counts.

## How to try it on the HPC

Copy the three files to the remote `~/qaoa_jobs/` directory.

Submit with:

```bash
sbatch ~/qaoa_jobs/run_qaoa_multinode.sh input_hamiltonian.npz output.json ~/qaoa_logs/test
```

The default script requests:

```bash
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
```

For 28 qubits, start with 4 or 8 nodes and low optimizer settings:

```bash
export QAOA_MAXITER=5
export QAOA_SHOTS=512
export QAOA_REPS=1
```

Then scale only after the small test completes.

## Important

This still depends on the remote Qulacs build actually being MPI/distributed.
If the installed `QuantumState` is not distributed across MPI ranks, requesting
more nodes will not solve the memory problem.
