import numpy as np
from scipy.optimize import minimize
import time
import os

from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2
from qiskit.primitives import StatevectorEstimator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.circuit.library import QAOAAnsatz
from qiskit.quantum_info import SparsePauliOp


# ---------------------------------------------------------------------------
# Qiskit primitives are created INSIDE run_qaoa_aer (not at module level)
# so that each Dask worker / process gets its own fresh instances and avoids
# pickling issues with stateful C++ backends.
# ---------------------------------------------------------------------------


def _build_primitives(
    aer_max_parallel_threads=None,
    aer_max_parallel_experiments=1,
    aer_max_parallel_shots=1,
):
    """Create fresh Qiskit primitive instances (safe for multiprocess)."""
    backend_options = {"method": "statevector"}
    if aer_max_parallel_threads is not None:
        backend_options["max_parallel_threads"] = int(aer_max_parallel_threads)
    if aer_max_parallel_experiments is not None:
        backend_options["max_parallel_experiments"] = int(aer_max_parallel_experiments)
    if aer_max_parallel_shots is not None:
        backend_options["max_parallel_shots"] = int(aer_max_parallel_shots)

    simulator = AerSimulator(**backend_options)
    estimator = StatevectorEstimator()
    sampler = SamplerV2(options={"backend_options": dict(backend_options)})
    return simulator, estimator, sampler


def ansatz_builder(H, reps=4):
    """Build QAOA ansatz circuit from Hamiltonian."""
    return QAOAAnsatz(H, reps=reps)


def transpile_circuit(circuit, backend):
    """Transpile circuit for the target backend."""
    pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
    return pm.run(circuit)


def cost_func_estimator(params, ansatz, hamiltonian, estimator_instance):
    """Cost function for QAOA optimization using estimator."""
    isa_hamiltonian = hamiltonian.apply_layout(ansatz.layout)
    pub = (ansatz, isa_hamiltonian, params)
    job = estimator_instance.run([pub])
    results = job.result()[0]
    cost = float(np.real(results.data.evs))
    return cost


def optimize_parameters(init_params, candidate_circuit, hamiltonian,
                        estimator_instance, method="SLSQP",
                        maxiter=500, ftol=1e-6):
    """Optimize QAOA parameters using scipy minimize."""
    eval_count = [0]

    def _cost(params):
        eval_count[0] += 1
        return cost_func_estimator(
            params, candidate_circuit, hamiltonian, estimator_instance
        )

    res = minimize(
        _cost,
        init_params,
        method=method,
        options={"maxiter": maxiter, "ftol": ftol},
    )
    return res.x, res.fun, eval_count[0]


def sample_circuit(candidate_circuit, optimal_params, sampler_instance):
    """Sample from the optimized QAOA circuit."""
    measured = candidate_circuit.copy()
    measured.measure_all()
    bound_circuit = measured.assign_parameters(optimal_params)

    job = sampler_instance.run([bound_circuit])
    counts_int = job.result()[0].data.meas.get_int_counts()
    shots = sum(counts_int.values())
    return {key: val / shots for key, val in counts_int.items()}


def to_bitstring(integer, num_bits):
    """Convert integer to bitstring list."""
    result = np.binary_repr(integer, width=num_bits)
    return [int(digit) for digit in result]


def bitstring_energy(bitstring, sparse_terms):
    """
    Compute classical Ising energy E(z) = <z|H|z> for a sampled bitstring.

    This Hamiltonian builder currently emits diagonal terms (Z/ZZ), so this
    evaluates energies exactly for computational-basis samples.
    """
    z = 1 - 2 * np.asarray(bitstring, dtype=float)  
    energy = 0.0
    for op, positions, coeff in sparse_terms:
        c = float(np.real(coeff))
        if op == "Z":
            energy += c * z[int(positions[0])]
        elif op == "ZZ":
            i, j = int(positions[0]), int(positions[1])
            energy += c * z[i] * z[j]
        elif op == "I":
            energy += c
    return float(energy)


def extract_lowest_energy_bitstring(distribution, num_qubits, sparse_terms):
    """Return sampled bitstring with minimum energy."""
    best_bitstring = None
    best_energy = np.inf

    for key in distribution.keys():
        bitstring = to_bitstring(int(key), num_qubits)
        bitstring.reverse()
        energy = bitstring_energy(bitstring, sparse_terms)

        if energy < best_energy:
            best_bitstring = bitstring
            best_energy = energy

    return best_bitstring, float(best_energy)


def extract_top_k_bitstrings(distribution, num_qubits, top_k=10):
    """
    Return top-k sampled bitstrings with their absolute (global) probabilities.

    Probabilities are taken directly from the sampler output and therefore sum
    to <= 1.0 over the reported top-k entries.
    """
    if top_k is None or top_k < 1:
        top_k = 10

    ranked = sorted(
        distribution.items(),
        key=lambda kv: (-float(kv[1]), int(kv[0])),
    )[:int(top_k)]

    out = []
    for key, prob in ranked:
        bitstring = to_bitstring(int(key), num_qubits)
        bitstring.reverse()  # Keep ordering consistent with selected bitstrings.
        out.append({
            "bitstring": "".join(str(b) for b in bitstring),
            "probability": float(prob),
            "state_int": int(key),
        })
    return out


def run_qaoa_aer(
    hamiltonian_path,
    reps=4,
    init_params=None,
    optimization_method="SLSQP",
    maxiter=500,
    ftol=1e-3,
    aer_max_parallel_threads=1,
    aer_max_parallel_experiments=1,
    aer_max_parallel_shots=1,
    log_backend_config=False,
    return_topk=False,
    top_k=10,
):
    """
    Complete QAOA pipeline (single run, random init).

    Args:
        hamiltonian_path: Path to saved Hamiltonian (.npz)
        reps: QAOA layers (default 4)
        init_params: Initial parameters (default: random uniform [0, 2π])
        optimization_method: scipy method (default SLSQP)
        maxiter: Max optimizer iterations (default 500)
        ftol: Function tolerance (default 1e-6)
        aer_max_parallel_threads: Aer CPU threads per task (1 forces sequential)
        aer_max_parallel_experiments: Aer experiment-level parallelism
        aer_max_parallel_shots: Aer shot-level parallelism
        log_backend_config: Print effective Aer/OpenMP config for debugging

    Returns:
        (best_bitstring, optimal_energy) by default.
        If return_topk=True, returns:
            (best_bitstring, optimal_energy, top_bitstrings)
    """
    t0 = time.time()

    # Fresh primitives for this call (avoids module-level pickle issues)
    simulator, estimator, sampler = _build_primitives(
        aer_max_parallel_threads=aer_max_parallel_threads,
        aer_max_parallel_experiments=aer_max_parallel_experiments,
        aer_max_parallel_shots=aer_max_parallel_shots,
    )

    if log_backend_config:
        print(
            "  Aer config: "
            f"max_parallel_threads={aer_max_parallel_threads}, "
            f"max_parallel_experiments={aer_max_parallel_experiments}, "
            f"max_parallel_shots={aer_max_parallel_shots}, "
            f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}, "
            f"MKL_NUM_THREADS={os.environ.get('MKL_NUM_THREADS')}"
        )

    # Load Hamiltonian
    data = np.load(hamiltonian_path, allow_pickle=False)
    if "SparsePauliOp" in data and "coeffs" in data:
        ops = data["SparsePauliOp"]
        positions = data["coeffs"]
        coeffs = data["coeffs"]
        num_qubits = int(data["num_qubits"][0])

        sparse_list = []
        for op, pos_row, coeff in zip(ops, positions, coeffs):
            op = str(op)
            if op == "Z":
                pos = [int(pos_row[0])]
            elif op == "ZZ":
                pos = [int(pos_row[0]), int(pos_row[1])]
            else:
                raise ValueError(f"Unsupported sparse op in Hamiltonian file: {op}")
            sparse_list.append((op, pos, complex(coeff)))
        hamiltonian = SparsePauliOp.from_sparse_list(sparse_list, num_qubits=num_qubits)
    else:
        # Backward compatibility with older Hamiltonian files.
        paulis = data["paulis"]
        coeffs = data["coeffs"]
        hamiltonian = SparsePauliOp(paulis, coeffs)
    num_qubits = hamiltonian.num_qubits
    n_params = 2 * reps

    # Build & transpile ansatz
    ansatz = ansatz_builder(hamiltonian, reps=reps)
    candidate_circuit = transpile_circuit(ansatz, simulator)

    # Random initialization unless explicitly provided
    if init_params is None:
        np.random.seed(42)  # For reproducibility
        p0 = np.random.uniform(0, 2 * np.pi, size=n_params)
    else:
        p0 = np.asarray(init_params, dtype=float)

    opt_params, opt_cost, n_evals = optimize_parameters(
        p0, candidate_circuit, hamiltonian, estimator,
        method=optimization_method, maxiter=maxiter, ftol=ftol,
    )

    # Sample from the optimised solution and select the minimum-energy sample
    distribution = sample_circuit(candidate_circuit, opt_params, sampler)
    sparse_terms = hamiltonian.to_sparse_list()
    best_bitstring, best_sample_energy = extract_lowest_energy_bitstring(
        distribution, num_qubits, sparse_terms
    )
    top_bitstrings = extract_top_k_bitstrings(
        distribution, num_qubits, top_k=top_k
    ) if return_topk else None

    elapsed = time.time() - t0
    print(
        f"  QAOA [{num_qubits}q, {reps}p]: "
        f"expectation={opt_cost:.6f}, sample_min={best_sample_energy:.6f}, "
        f"evals={n_evals}, "
        f"selected={sum(best_bitstring)}/{num_qubits} nodes, "
        f"time={elapsed:.1f}s"
    )

    if return_topk:
        return best_bitstring, float(best_sample_energy), top_bitstrings
    return best_bitstring, float(best_sample_energy)

def qaoa_test (
    hamiltonian_path,
    reps=4,
    init_params=None,
    optimization_method="SLSQP",
    maxiter=500,
    ftol=1e-3,
    aer_max_parallel_threads=1,
    aer_max_parallel_experiments=1,
    aer_max_parallel_shots=1,
    log_backend_config=False,
    return_topk=False,
    top_k=10,
):
    """
    Test pipeline.

    Args:
        hamiltonian_path: Path to Hamiltonian in records array
        reps: QAOA layers (default 4)
        init_params: Initial parameters (default: random uniform [0, 2π])
        optimization_method: scipy method (default SLSQP)
        maxiter: Max optimizer iterations (default 500)
        ftol: Function tolerance (default 1e-6)
        aer_max_parallel_threads: Aer CPU threads per task (1 forces sequential)
        aer_max_parallel_experiments: Aer experiment-level parallelism
        aer_max_parallel_shots: Aer shot-level parallelism
        log_backend_config: Print effective Aer/OpenMP config for debugging

    Returns:
        (best_bitstring, optimal_energy) by default.
        If return_topk=True, returns:
            (best_bitstring, optimal_energy, top_bitstrings)
    """
    t0 = time.time()

    # Fresh primitives for this call (avoids module-level pickle issues)
    simulator, estimator, sampler = _build_primitives(
        aer_max_parallel_threads=aer_max_parallel_threads,
        aer_max_parallel_experiments=aer_max_parallel_experiments,
        aer_max_parallel_shots=aer_max_parallel_shots,
    )

    if log_backend_config:
        print(
            "  Aer config: "
            f"max_parallel_threads={aer_max_parallel_threads}, "
            f"max_parallel_experiments={aer_max_parallel_experiments}, "
            f"max_parallel_shots={aer_max_parallel_shots}, "
            f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}, "
            f"MKL_NUM_THREADS={os.environ.get('MKL_NUM_THREADS')}"
        )

    # Load Hamiltonian. Preferred format is a SparsePauliOp already stored
    # on each record (record.hamiltonian_path = SparsePauliOp(...)).
    if isinstance(hamiltonian_path, SparsePauliOp):
        hamiltonian = hamiltonian_path
    elif isinstance(hamiltonian_path, (str, os.PathLike)):
        data = np.load(hamiltonian_path, allow_pickle=False)
        if "SparsePauliOp" in data and "coeffs" in data:
            ops = data["SparsePauliOp"]
            positions = data["positions"]
            coeffs = data["coeffs"]
            num_qubits = int(data["num_qubits"][0])

            sparse_list = []
            for op, pos_row, coeff in zip(ops, positions, coeffs):
                op = str(op)
                if op == "Z":
                    pos = [int(pos_row[0])]
                elif op == "ZZ":
                    pos = [int(pos_row[0]), int(pos_row[1])]
                elif op == "I":
                    pos = []
                else:
                    raise ValueError(f"Unsupported sparse op in Hamiltonian file: {op}")
                sparse_list.append((op, pos, complex(coeff)))
            hamiltonian = SparsePauliOp.from_sparse_list(sparse_list, num_qubits=num_qubits)
        else:
            # Backward compatibility with older Hamiltonian files.
            paulis = data["paulis"]
            coeffs = data["coeffs"]
            hamiltonian = SparsePauliOp(paulis, coeffs)
    elif isinstance(hamiltonian_path, dict):
        # Accept in-memory dict-like payloads for ad hoc testing.
        if "SparsePauliOp" in hamiltonian_path and "coeffs" in hamiltonian_path:
            ops = hamiltonian_path["SparsePauliOp"]
            positions = hamiltonian_path["positions"]
            coeffs = hamiltonian_path["coeffs"]
            num_qubits = int(hamiltonian_path["num_qubits"][0])

            sparse_list = []
            for op, pos_row, coeff in zip(ops, positions, coeffs):
                op = str(op)
                if op == "Z":
                    pos = [int(pos_row[0])]
                elif op == "ZZ":
                    pos = [int(pos_row[0]), int(pos_row[1])]
                elif op == "I":
                    pos = []
                else:
                    raise ValueError(f"Unsupported sparse op in Hamiltonian payload: {op}")
                sparse_list.append((op, pos, complex(coeff)))
            hamiltonian = SparsePauliOp.from_sparse_list(sparse_list, num_qubits=num_qubits)
        else:
            paulis = hamiltonian_path["paulis"]
            coeffs = hamiltonian_path["coeffs"]
            hamiltonian = SparsePauliOp(paulis, coeffs)
    else:
        raise TypeError(
            "hamiltonian_path must be SparsePauliOp, a .npz path, or a dict payload."
        )
    num_qubits = hamiltonian.num_qubits
    n_params = 2 * reps

    # Build & transpile ansatz
    ansatz = ansatz_builder(hamiltonian, reps=reps)
    candidate_circuit = transpile_circuit(ansatz, simulator)

    # Random initialization unless explicitly provided
    if init_params is None:
        np.random.seed(42)  # For reproducibility
        p0 = np.random.uniform(0, 2 * np.pi, size=n_params)
    else:
        p0 = np.asarray(init_params, dtype=float)

    opt_params, opt_cost, n_evals = optimize_parameters(
        p0, candidate_circuit, hamiltonian, estimator,
        method=optimization_method, maxiter=maxiter, ftol=ftol,
    )

    # Sample from the optimised solution and select the minimum-energy sample
    distribution = sample_circuit(candidate_circuit, opt_params, sampler)
    sparse_terms = hamiltonian.to_sparse_list()
    best_bitstring, best_sample_energy = extract_lowest_energy_bitstring(
        distribution, num_qubits, sparse_terms
    )
    top_bitstrings = extract_top_k_bitstrings(
        distribution, num_qubits, top_k=top_k
    ) if return_topk else None

    elapsed = time.time() - t0
    print(
        f"  QAOA [{num_qubits}q, {reps}p]: "
        f"expectation={opt_cost:.6f}, sample_min={best_sample_energy:.6f}, "
        f"evals={n_evals}, "
        f"selected={sum(best_bitstring)}/{num_qubits} nodes, "
        f"time={elapsed:.1f}s"
    )

    if return_topk:
        return best_bitstring, float(best_sample_energy), top_bitstrings
    return best_bitstring, float(best_sample_energy)
