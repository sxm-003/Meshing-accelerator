import dimod
from scipy.ndimage import label

def sparse_pauli_to_ising(H, atol = 1e-12):
    
    h= {}
    J = {}
    offset = 0.0

    num_qubits = H.num_qubits

    for string, coeff in H.to_list():
        coeff = float(complex(coeff).real)
        if abs(coeff) < atol:
            continue

        if any(p in string for p in "XY"):
            raise ValueError("Cannot convert Hamiltonian with non-Z Pauli terms to Ising model.")
        

        z_qubits = [
            num_qubits - 1 - pos
            for pos, ch in enumerate(string)
            if ch == "Z"
        ]

        if len(z_qubits) == 0:
            offset += coeff
        
        elif len(z_qubits) == 1:
            i = z_qubits[0]
            h[i] = h.get(i, 0.0) + coeff
        
        elif len(z_qubits) == 2:
            i, j = sorted(z_qubits)
            J[(i, j)] = J.get((i, j), 0.0) + coeff
        
        else:
            raise ValueError(f"Higher-order term found: {string}")

    return h, J, offset

def sparse_pauli_to_bqm(H, atol=1e-12):
    h, J, offset = sparse_pauli_to_ising(H, atol=atol)
    return dimod.BinaryQuadraticModel.from_ising(h, J, offset)


def sparse_pauli_to_qubo(H, atol=1e-12):
    h, J, offset = sparse_pauli_to_ising(H, atol=atol)
    return dimod.ising_to_qubo(h, J, offset)