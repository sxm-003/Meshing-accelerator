import numpy as np
from qulacs import QuantumState, QuantumCircuit
from qulacs.gate import PauliRotation
from mpi4py import MPI


def load_hamiltonian(npz_path):
    data = np.load(npz_path)
    return (
        data["sparse_ops"],
        data["sparse_positions"],
        data["coeffs"],
        int(data["num_qubits"][0]),
    )


def build_cost_unitary(n_qubits, ops, pos, coeffs, gamma):
    circuit = QuantumCircuit(n_qubits)

    for op, p, c in zip(ops, pos, coeffs):
        valid_pos = [x for x in p if x >= 0]
        if not valid_pos:
            continue

        pauli_ids = []
        for ch in op:
            if ch == "X":
                pauli_ids.append(1)
            elif ch == "Y":
                pauli_ids.append(2)
            elif ch == "Z":
                pauli_ids.append(3)

        circuit.add_gate(
            PauliRotation(valid_pos, pauli_ids, 2 * gamma * c.real)
        )

    return circuit


def run_qaoa_qulacs_mpi(npz_path, p=1):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    ops, pos, coeffs, n_qubits = load_hamiltonian(npz_path)

    state = QuantumState(n_qubits)
    state.set_zero_state()

    circuit = QuantumCircuit(n_qubits)

    # Initial H layer
    for i in range(n_qubits):
        circuit.add_H_gate(i)

    gamma = 0.5
    beta = 0.3

    cost = build_cost_unitary(n_qubits, ops, pos, coeffs, gamma)
    circuit.merge_circuit(cost)

    # Mixer
    for i in range(n_qubits):
        circuit.add_RX_gate(i, 2 * beta)

    circuit.update_quantum_state(state)

    probs = np.abs(state.get_vector())**2

    if rank == 0:
        idx = np.argmax(probs)
        bitstring = format(idx, f"0{n_qubits}b")
        return bitstring, float(np.max(probs))

    return None, None