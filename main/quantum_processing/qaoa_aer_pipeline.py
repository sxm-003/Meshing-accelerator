import numpy as np
from scipy.optimize import minimize
import time

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


def _build_primitives():
    """Create fresh Qiskit primitive instances (safe for multiprocess)."""
    simulator = AerSimulator(method="statevector")
    estimator = StatevectorEstimator()
    sampler = SamplerV2()
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


def extract_best_bitstring(distribution, num_qubits):
    """Extract the most likely bitstring from distribution."""
    keys = list(distribution.keys())
    values = list(distribution.values())
    most_likely = keys[np.argmax(np.abs(values))]
    most_likely_bitstring = to_bitstring(most_likely, num_qubits)
    most_likely_bitstring.reverse()
    return most_likely_bitstring


def run_qaoa_aer(
    hamiltonian_path,
    reps=4,
    init_params=None,
    optimization_method="SLSQP",
    maxiter=500,
    ftol=1e-6,
    num_restarts=3,
):
    """
    Complete QAOA pipeline with multi-restart (matches notebook behaviour).

    Key differences from the old version:
      • reps=4 (was 2)  – doubles circuit expressivity
      • ftol=1e-6 (was 1e-3) – optimizer now converges properly
      • random init params (was fixed [π, π/2, …])
      • multi-restart: runs `num_restarts` attempts, keeps best
      • primitives created per-call (safe for Dask workers)

    Args:
        hamiltonian_path: Path to saved Hamiltonian (.npz)
        reps: QAOA layers (default 4)
        init_params: Initial parameters (default: random uniform [0, 2π])
        optimization_method: scipy method (default SLSQP)
        maxiter: Max optimizer iterations per restart (default 500)
        ftol: Function tolerance (default 1e-6)
        num_restarts: Number of random restarts (default 3)

    Returns:
        (best_bitstring, optimal_energy)
    """
    t0 = time.time()

    # Fresh primitives for this call (avoids module-level pickle issues)
    simulator, estimator, sampler = _build_primitives()

    # Load Hamiltonian
    data = np.load(hamiltonian_path, allow_pickle=False)
    paulis = data["paulis"]
    coeffs = data["coeffs"]
    hamiltonian = SparsePauliOp(paulis, coeffs)
    num_qubits = hamiltonian.num_qubits
    n_params = 2 * reps

    # Build & transpile ansatz (done once, reused across restarts)
    ansatz = ansatz_builder(hamiltonian, reps=reps)
    candidate_circuit = transpile_circuit(ansatz, simulator)

    best_energy = float("inf")
    best_params = None
    total_evals = 0

    for attempt in range(num_restarts):
        # Random initialization (matching notebook behaviour)
        if init_params is not None and attempt == 0:
            p0 = np.asarray(init_params, dtype=float)
        else:
            p0 = np.random.uniform(0, 2 * np.pi, size=n_params)

        opt_params, opt_cost, n_evals = optimize_parameters(
            p0, candidate_circuit, hamiltonian, estimator,
            method=optimization_method, maxiter=maxiter, ftol=ftol,
        )
        total_evals += n_evals

        if opt_cost < best_energy:
            best_energy = opt_cost
            best_params = opt_params

    # Sample from the best solution
    distribution = sample_circuit(candidate_circuit, best_params, sampler)
    best_bitstring = extract_best_bitstring(distribution, num_qubits)

    elapsed = time.time() - t0
    print(
        f"  QAOA [{num_qubits}q, {reps}p, {num_restarts}×restart]: "
        f"energy={best_energy:.6f}, evals={total_evals}, "
        f"selected={sum(best_bitstring)}/{num_qubits} nodes, "
        f"time={elapsed:.1f}s"
    )

    return best_bitstring, float(best_energy)