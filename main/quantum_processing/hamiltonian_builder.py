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


def build_all_pairs(n):
    """
    n: number of nodes

    returns:
        list of (i, j) with i < j
    """
    return list(combinations(range(n), 2))


def build_radius_bend_triples(r, radius, max_degree=8):
    """
    r: array of shape (n, 2) – node coordinates
    radius: interaction radius
    max_degree: cap neighbors per node to keep locality

    returns:
        list of (i, j, k) triples
        meaning: j and k are local neighbors of i
    """
    n = len(r)

    # Step 1: find local neighbors per node
    neighbors = [[] for _ in range(n)]

    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(r[i] - r[j])
            if d <= radius:
                dists.append((d, i, j))

    dists.sort()

    degree = [0] * n
    for _, i, j in dists:
        if degree[i] < max_degree and degree[j] < max_degree:
            neighbors[i].append(j)
            neighbors[j].append(i)
            degree[i] += 1
            degree[j] += 1

    # Step 2: build bend triples
    bend_triples = []
    for i in range(n):
        nbrs = neighbors[i]
        if len(nbrs) < 2:
            continue
        for j, k in combinations(nbrs, 2):
            bend_triples.append((i, j, k))

    return bend_triples

def domain_penalty_strings(phi, alpha, band):
    n = len(phi)
    terms = {}

    for i, phi_i in enumerate(phi):
        g = max(phi_i + band, 0.0)
        coeff = 0.5 * alpha * g * g

        if coeff > 0:
            terms[pauli_Z(n, i)] = terms.get(pauli_Z(n, i), 0) - coeff

    return terms


def phi_circle_field_local(nodes, R):
    center = nodes.mean(axis=0)
    x = nodes[:,0] - center[0]
    y = nodes[:,1] - center[1]
    return np.sqrt(x*x + y*y) - R


 
def spacing_penalty_strings(r, neighbors, L, gamma):
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
    """
    Proper cardinality penalty:
        mu * (sum x_i - N)^2
    """
    terms = {}

    A = mu * (N - n/2)
    B = mu / 2

    # Linear Z terms
    for i in range(n):
        zi = pauli_Z(n, i)
        terms[zi] = terms.get(zi, 0.0) + A

    # Quadratic ZZ terms
    for i in range(n):
        for j in range(i+1, n):
            zij = pauli_ZZ(n, i, j)
            terms[zij] = terms.get(zij, 0.0) + B

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

def bend_penalty_strings(r, bend_triples, kappa):
    """
    r: node coordinates
    bend_triples: (i, j, k) with j,k neighbors of i
    kappa: weight
    """
    n = len(r)
    terms = {}

    for i, j, k in bend_triples:
        rij = np.linalg.norm(np.array(r[j]) - np.array(r[i]))
        rik = np.linalg.norm(np.array(r[k]) - np.array(r[i]))
        rjk = np.linalg.norm(np.array(r[j]) - np.array(r[k]))

        w = (rij**2 + rik**2 - rjk**2)**2
        if w == 0:
            continue

        # distribute as ZZ terms
        for a, b in [(i, j), (i, k), (j, k)]:
            zz = pauli_ZZ(n, a, b)
            terms[zz] = terms.get(zz, 0.0) + kappa * w / 12

            for q in (a, b):
                z = pauli_Z(n, q)
                terms[z] = terms.get(z, 0.0) - kappa * w / 12

    return terms


def hamiltonian_builder(
    phi,
    r,
    L,
    alpha,
    band,
    gamma,
    use_sparsity=False,
    N=None,
    mu=0.0,
    use_repulsion=False,
    d_min=None,
    eta=0.0,
    use_bend=False,
    kappa=1.0

):


    n = len(phi)
    H_terms = {}
    neighbors_3 = build_radius_bend_triples(r, 1.3*L)
    neighbors_2 = build_all_pairs(n)

    for p, c in domain_penalty_strings(phi, alpha).items():
        H_terms[p] = H_terms.get(p, 0.0) + c

    for p, c in spacing_penalty_strings(r, neighbors_2, L, gamma).items():
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

    if use_bend:
        for p, c in bend_penalty_strings(r, neighbors_3, kappa).items():
            H_terms[p] = H_terms.get(p, 0.0) + c
  
    paulis = list(H_terms.keys())
    coeffs = list(H_terms.values())

    return SparsePauliOp(paulis, coeffs)

