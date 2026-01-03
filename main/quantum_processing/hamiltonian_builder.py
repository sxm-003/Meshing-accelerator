import numpy as np
from itertools import combinations
from qiskit.quantum_info import SparsePauliOp

def pauli_Z(n,k):
    p = ["I"] * n
    p[k] = "Z"
    return "".join(p)

def pauli_ZZ(n,k,l):
    p = ["I"] * n
    p[k] = "Z"
    p[l] = "Z"
    return "".join(p)
    
def phi_circle_field(nodes, R=1.0):
    x = nodes[:, 0]
    y = nodes[:, 1]
    return np.sqrt(x*x + y*y) - R

def domain_penalty_strings(phi, alpha):
     n = len(phi)
     terms = {}

     for i , phi_i in enumerate(phi):
         coefficient = (alpha*max(phi_i,0)**2)/2
         if coefficient == 0:
             continue 

         terms[pauli_Z(n,i)] = terms.get(pauli_Z(n,i),0) - coefficient 

     return terms

 
def shape_penalty_strings(r, neighbors, L, gamma):
    n = len(r)
    terms = {}

    for k, l in neighbors:
        distance = np.linalg.norm(np.array(r[k]) - np.array(r[l]))
        w_kl = (distance - L)**2

        if w_kl == 0:
            continue

        # ZZ term
        vals_zz = pauli_ZZ(n, k, l)
        terms[vals_zz] = terms.get(vals_zz, 0.0) + gamma * w_kl / 4

        # single-Z terms
        for q in (k, l):
            vals_z = pauli_Z(n, q)
            terms[vals_z] = terms.get(vals_z, 0.0) - gamma * w_kl / 4

    return terms

def sparsity_penalty_strings(n, N, mu):
    terms = {}
    
    h = mu * (N - 0.5)
    for k in range(n):
        vals_z = pauli_Z(n, k)
        terms[vals_z] = terms.get(vals_z, 0) + h

    for k, l in combinations(range(n), 2):
        vals_zz = pauli_ZZ(n, k, l)
        terms[vals_zz] = terms.get(vals_zz, 0) + mu / 2

    return terms

def repulsion_penalty_strings(r, d_min, eta):
    n = len(r)
    terms = {}

    for k,l in combinations(range(n), 2):
        distance = np.linalg.norm(np.array(r[k]) - np.array(r[l]))
        if distance >= d_min:
            continue

        w_kl = (d_min - distance)**2

        vals_zz = pauli_ZZ(n, k, l)
        terms[vals_zz] = terms.get(vals_zz, 0) + eta * w_kl / 4
        
        for q in (k,l):
            vals_z = pauli_Z(n, q)
            terms[vals_z] = terms.get(vals_z, 0.0) - eta * w_kl / 4

    return terms

def hamiltonian_builder(
    phi,
    r,
    neighbors,
    L,
    alpha,
    gamma,
    use_sparsity=False,
    N=None,
    mu=0.0,
    use_repulsion=False,
    d_min=None,
    eta=0.0,
):


    n = len(phi)
    H_terms = {}

    for p, c in domain_penalty_strings(phi, alpha).items():
        H_terms[p] = H_terms.get(p, 0.0) + c

    for p, c in shape_penalty_strings(r, neighbors, L, gamma).items():
        H_terms[p] = H_terms.get(p, 0.0) + c

    if use_sparsity:
        if N is None:
            raise ValueError("N must be provided when use_sparsity=True")
        for p, c in sparsity_penalty_strings(n, N, mu).items():
            H_terms[p] = H_terms.get(p, 0.0) + c

    if use_repulsion:
        if d_min is None:
            raise ValueError("d_min must be provided when use_repulsion=True")
        for p, c in repulsion_penalty_strings(r, d_min, eta).items():
            H_terms[p] = H_terms.get(p, 0.0) + c

  
    paulis = list(H_terms.keys())
    coeffs = list(H_terms.values())

    return SparsePauliOp(paulis, coeffs)

