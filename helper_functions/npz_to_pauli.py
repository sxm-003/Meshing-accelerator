import numpy as np
from qiskit.quantum_info import SparsePauliOp

def load_sparse_pauli_from_npz(path):
    data = np.load(path, allow_pickle=False)

    ops = data["sparse_ops"]
    positions = data["sparse_positions"]
    coeffs = data["coeffs"]
    num_qubits = int(np.asarray(data["num_qubits"]).reshape(-1)[0])

    sparse_list = []
    for op, pos_row, coeff in zip(ops, positions, coeffs):
        pos = [int(x) for x in np.asarray(pos_row).reshape(-1) if int(x) >= 0]
        sparse_list.append((str(op), pos, complex(coeff)))

    return SparsePauliOp.from_sparse_list(sparse_list, num_qubits=num_qubits)
