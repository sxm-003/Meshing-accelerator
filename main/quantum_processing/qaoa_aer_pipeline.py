import numpy as np
from scipy.optimize import minimize
import qiskit
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.primitives import StatevectorEstimator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.circuit.library import QAOAAnsatz
from qiskit.quantum_info import SparsePauliOp

estimator = StatevectorEstimator()
simulator = AerSimulator(method="statevector")
sampler = SamplerV2()


def ansatz_builder(H, reps=1):
    """Build QAOA ansatz circuit from Hamiltonian."""
    ansatz = QAOAAnsatz(H, reps=reps)
    return ansatz


def transpile_circuit(circuit, backend=None):
    """Transpile circuit for the target backend."""
    if backend is None:
        backend = simulator
    pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
    transpiled = pm.run(circuit)
    return transpiled


def cost_func_estimator(params, ansatz, hamiltonian, estimator_instance):
    """Cost function for QAOA optimization using estimator."""
    # Transform the observable to physical qubits
    isa_hamiltonian = hamiltonian.apply_layout(ansatz.layout)
    
    pub = (ansatz, isa_hamiltonian, params)
    job = estimator_instance.run([pub])
    
    results = job.result()[0]
    cost = results.data.evs
    
    return cost


def optimize_parameters(init_params, candidate_circuit, hamiltonian, estimator_instance, 
                       method="SLSQP", maxiter=2000, ftol=1e-3):
    """Optimize QAOA parameters using scipy minimize."""
    res = minimize(
        cost_func_estimator,
        init_params,
        args=(candidate_circuit, hamiltonian, estimator_instance),
        method=method,
        options={'maxiter': maxiter, 'ftol': ftol}
    )
    
    return res.x, res.fun


def sample_circuit(candidate_circuit, optimal_params, sampler_instance):
    """Sample from the optimized QAOA circuit."""
    measured = candidate_circuit.copy()
    measured.measure_all()
    
    bound_circuit = measured.assign_parameters(optimal_params)
    
    job = sampler_instance.run([bound_circuit])
    counts_int = job.result()[0].data.meas.get_int_counts()
    shots = sum(counts_int.values())
    final_distribution_int = {key: val / shots for key, val in counts_int.items()}
    
    return final_distribution_int


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


def run_qaoa_aer(hamiltonian_path, reps=2, init_params=None, 
                     optimization_method="SLSQP", maxiter=2000, ftol=1e-3):
    """
    Complete QAOA pipeline that returns the best bitstring and energy.
    
    Args:
        hamiltonian_path: Path to saved Hamiltonian (.npz file with 'paulis' and 'coeffs')
        reps: Number of QAOA repetitions (default: 2)
        init_params: Initial parameters (default: alternating pi and pi/2)
        optimization_method: Optimization method (default: SLSQP)
        maxiter: Maximum iterations for optimizer (default: 2000)
        ftol: Function tolerance for optimizer (default: 1e-3)
    
    Returns:
        tuple: (best_bitstring, optimal_energy)
            - best_bitstring: list of 0s and 1s
            - optimal_energy: float value of the minimum energy found
    """
    # Load Hamiltonian from file
    data = np.load(hamiltonian_path)
    paulis = data['paulis']
    coeffs = data['coeffs']
    hamiltonian = SparsePauliOp(paulis, coeffs)
    
    # Get number of qubits
    num_qubits = hamiltonian._op_shape.num_qargs[0]
    
    # Set default initial parameters if not provided
    if init_params is None:
        init_params = [np.pi, np.pi/2] * reps
    
    # Build ansatz
    ansatz = ansatz_builder(hamiltonian, reps=reps)
    
    # Transpile circuit
    candidate_circuit = transpile_circuit(ansatz, simulator)
    
    # Optimize parameters
    optimal_params, optimal_cost = optimize_parameters(
        init_params, candidate_circuit, hamiltonian, estimator,
        method=optimization_method, maxiter=maxiter, ftol=ftol
    )
    
    # Sample from optimized circuit
    distribution = sample_circuit(candidate_circuit, optimal_params, sampler)
    
    # Extract best bitstring
    best_bitstring = extract_best_bitstring(distribution, num_qubits)
    
    return best_bitstring, float(optimal_cost)