import numpy as np
from qulacs import QuantumState, QuantumCircuit
from qulacs.gate import PauliRotation
from mpi4py import MPI


def load_hamiltonian(npz_path):
    data = np.load(npz_path, allow_pickle=False)
    coeff_key = "coeffs" if "coeffs" in data else "sparse_coeffs"
    num_qubits_raw = data["num_qubits"]
    num_qubits = int(np.asarray(num_qubits_raw).reshape(-1)[0])
    return (
        data["sparse_ops"],
        data["sparse_positions"],
        data[coeff_key],
        num_qubits,
    )


def build_cost_unitary(n_qubits, ops, pos, coeffs, gamma):
    circuit = QuantumCircuit(n_qubits)

    for op, p, c in zip(ops, pos, coeffs):
        op = str(op)
        valid_pos = [int(x) for x in p if x >= 0]
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
            PauliRotation(valid_pos, pauli_ids, 2.0 * gamma * float(np.real(c)))
        )

    return circuit


def bitstring_energy(bitstring, ops, pos, coeffs):
    z = 1 - 2 * np.asarray(bitstring, dtype=float)
    energy = 0.0

    for op, p, c in zip(ops, pos, coeffs):
        op = str(op)
        valid_pos = [int(x) for x in p if x >= 0]
        coeff = float(np.real(c))

        if not valid_pos:
            continue

        if set(op) == {"Z"} and len(valid_pos) == len(op):
            term_value = 1.0
            for idx in valid_pos:
                term_value *= z[idx]
            energy += coeff * term_value

    return float(energy)


def run_qaoa_qulacs_mpi3(npz_path, p=1):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    ops, pos, coeffs, n_qubits = load_hamiltonian(npz_path)

    state = QuantumState(n_qubits)
    state.set_zero_state()

    circuit = QuantumCircuit(n_qubits)

    for i in range(n_qubits):
        circuit.add_H_gate(i)

    gamma = 0.5
    beta = 0.3

    cost = build_cost_unitary(n_qubits, ops, pos, coeffs, gamma)
    circuit.merge_circuit(cost)

    for i in range(n_qubits):
        circuit.add_RX_gate(i, 2 * beta)

    circuit.update_quantum_state(state)
    probs = np.abs(state.get_vector()) ** 2

    if rank == 0:
        idx = int(np.argmax(probs))
        bitstring_list = [int(b) for b in format(idx, f"0{n_qubits}b")]
        bitstring_list.reverse()
        bitstring = "".join(str(b) for b in bitstring_list)
        energy = bitstring_energy(bitstring_list, ops, pos, coeffs)
        return bitstring, energy

    return None, None
