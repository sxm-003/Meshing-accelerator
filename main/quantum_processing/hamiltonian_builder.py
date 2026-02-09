import numpy as np
from itertools import combinations
from qiskit.quantum_info import SparsePauliOp


def compute_distance_matrix(r):
    """Precompute all pairwise distances using vectorized numpy."""
    r = np.asarray(r, dtype=float)
    diff = r[:, np.newaxis, :] - r[np.newaxis, :, :]
    return np.sqrt((diff ** 2).sum(axis=-1))


# PAULI OPERATORS

def pauli_Z(n,k):
    p = ["I"] * n
    p[k] = "Z"
    return "".join(p)

def pauli_ZZ(n,k,l):
    p = ["I"] * n
    p[k] = "Z"
    p[l] = "Z"
    return "".join(p)



def build_radius_bend_triples(r, radius, max_degree=8, dmat=None):
    """
    r: array of shape (n, 2) – node coordinates
    radius: interaction radius
    max_degree: cap neighbors per node to keep locality
    dmat: optional precomputed distance matrix (avoids redundant O(n^2) work)

    returns:
        list of (i, j, k) triples
        meaning: j and k are local neighbors of i
    """
    n = len(r)
    if dmat is None:
        dmat = compute_distance_matrix(r)

    # Step 1: find local neighbors per node (vectorized distance lookup)
    triu_i, triu_j = np.triu_indices(n, k=1)
    d = dmat[triu_i, triu_j]
    mask = d <= radius
    valid_idx = np.where(mask)[0]
    sort_order = np.argsort(d[valid_idx])
    valid_sorted = valid_idx[sort_order]

    neighbors = [[] for _ in range(n)]
    degree = [0] * n
    for idx in valid_sorted:
        i, j = int(triu_i[idx]), int(triu_j[idx])
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


def max_edge_penalty_strings(r, d_max, eta, dmat=None):
    """Penalize pairs of nodes that are too far apart"""
    n = len(r)
    if dmat is None:
        dmat = compute_distance_matrix(r)
    terms = {}

    for i in range(n):
        for j in range(i+1, n):
            dij = dmat[i, j]
            if dij <= d_max:
                continue

            w = eta * (dij - d_max)**2 / 4

            zz = pauli_ZZ(n, i, j)
            terms[zz] = terms.get(zz, 0.0) + w

            for q in (i, j):
                z = pauli_Z(n, q)
                terms[z] = terms.get(z, 0.0) - w

    return terms


def count_penalty_strings(n, target_n, lam):
    """Penalize deviation from target number of nodes"""
    terms = {}
    A = lam * (target_n - n/2)
    B = lam / 2

    for i in range(n):
        zi = pauli_Z(n, i)
        terms[zi] = terms.get(zi, 0.0) + A

    for i in range(n):
        for j in range(i+1, n):
            zij = pauli_ZZ(n, i, j)
            terms[zij] = terms.get(zij, 0.0) + B

    return terms


def build_radius_neighbors(r, radius, dmat=None):
    """Build neighbor dictionary within radius.
    
    Args:
        r: node coordinates
        radius: interaction radius
        dmat: optional precomputed distance matrix
    """
    n = len(r)
    neighbors = {}
    if dmat is None:
        r = np.array(r)
        dmat = compute_distance_matrix(r)

    for i in range(n):
        neighbors[i] = np.where((dmat[i] < radius) & (dmat[i] > 0))[0].tolist()

    return neighbors


def build_collinear_pairs(r, neighbors, cos_thresh=0.85):
    """Find nearly collinear neighbor pairs"""
    collinear_pairs = []
    r = np.array(r)

    for i, neighs in neighbors.items():
        ri = r[i]
        for a in range(len(neighs)):
            for b in range(a + 1, len(neighs)):
                j, k = neighs[a], neighs[b]

                u = r[j] - ri
                v = r[k] - ri
                u_norm = np.linalg.norm(u)
                v_norm = np.linalg.norm(v)
                
                if u_norm == 0 or v_norm == 0:
                    continue
                    
                u /= u_norm
                v /= v_norm

                if abs(np.dot(u, v)) > cos_thresh:
                    collinear_pairs.append((j, k))

    return collinear_pairs


def collinearity_penalty_strings(n, collinear_pairs, eta_col):
    """Penalize collinear node selection"""
    terms = {}
    w = eta_col / 4

    for j, k in collinear_pairs:
        terms[pauli_ZZ(n, j, k)] = terms.get(pauli_ZZ(n, j, k), 0.0) + w
        terms[pauli_Z(n, j)] = terms.get(pauli_Z(n, j), 0.0) - w
        terms[pauli_Z(n, k)] = terms.get(pauli_Z(n, k), 0.0) - w

    return terms

def compute_angular_bins(r, num_bins=6):
    """Partition nodes into angular bins around center"""
    r = np.array(r)
    center = r.mean(axis=0)
    
    # Compute angles from center
    angles = np.arctan2(r[:, 1] - center[1], r[:, 0] - center[0])
    angles = (angles + np.pi) / (2 * np.pi)  # Normalize to [0, 1]
    
    # Assign to bins
    bin_indices = (angles * num_bins).astype(int) % num_bins
    
    # Group nodes by bin
    bins = [[] for _ in range(num_bins)]
    for i, bin_id in enumerate(bin_indices):
        bins[bin_id].append(i)
    
    return bins

def angular_bins_penalty_strings(r, num_bins, eta_theta):
    """Penalize unbalanced angular distribution"""
    n = len(r)
    terms = {}
    
    bins = compute_angular_bins(r, num_bins=num_bins)
    
    # Target: each bin should have ~n/num_bins nodes
    target_per_bin = n / num_bins
    
    # Add penalty for imbalanced bins
    for bin_idx, bin_nodes in enumerate(bins):
        # Penalty = (count_in_bin - target)^2
        for i in bin_nodes:
            for j in bin_nodes:
                if i >= j:
                    continue
                w = eta_theta * (len(bin_nodes) - target_per_bin)**2 / (4 * num_bins)
                zz = pauli_ZZ(n, i, j)
                terms[zz] = terms.get(zz, 0.0) + w / 4
                
                z_i = pauli_Z(n, i)
                z_j = pauli_Z(n, j)
                terms[z_i] = terms.get(z_i, 0.0) - w / 8
                terms[z_j] = terms.get(z_j, 0.0) - w / 8
    
    return terms

def compute_boundary_normals(r, boundary_nodes, k=4):
    """
    Estimate boundary normals using local PCA.
    
    Args:
        r: array of (x,y) node coordinates
        boundary_nodes: list of indices on boundary
        k: number of neighbors for PCA
    
    Returns:
        normals: dictionary mapping boundary node index -> normal vector (outward)
    """
    r = np.array(r)
    normals = {}
    
    for i in boundary_nodes:
        # Find k-nearest neighbors on boundary
        boundary_coords = r[boundary_nodes]
        dists = np.linalg.norm(boundary_coords - r[i], axis=1)
        idx = np.argsort(dists)[1:k+1]  # Skip itself
        
        # Get neighbor points
        pts = boundary_coords[idx]
        pts_centered = pts - pts.mean(axis=0)
        
        # PCA: tangent = first eigenvector (largest variance)
        cov = pts_centered.T @ pts_centered
        eigvals, eigvecs = np.linalg.eigh(cov)
        tangent = eigvecs[:, np.argmax(eigvals)]
        
        # Normal = perpendicular to tangent
        normal = np.array([-tangent[1], tangent[0]])
        normal = normal / (np.linalg.norm(normal) + 1e-10)
        
        # Ensure normal points outward (away from center)
        center = r.mean(axis=0)
        if np.dot(normal, r[i] - center) < 0:
            normal = -normal
        
        normals[i] = normal
    
    return normals

def boundary_alignment_penalty_strings(r, boundary_nodes, boundary_normals, neighbors, beta):
    """
    Boundary alignment penalty - encourage edges on boundary to align with normals.
    
    This keeps the boundary well-defined by penalizing edges that cut across
    the boundary instead of following it.
    
    Args:
        r: node coordinates
        boundary_nodes: list of boundary node indices
        boundary_normals: dict of normal vectors for boundary nodes
        neighbors: neighbor dictionary from build_radius_neighbors
        beta: penalty weight
    
    Returns:
        terms: Pauli string penalty terms
    """
    n = len(r)
    terms = {}
    
    # For each boundary node
    for i in boundary_nodes:
        if i not in boundary_normals:
            continue
        
        ni = boundary_normals[i]  # Normal at boundary node i
        
        # Check edges to neighboring nodes
        if i not in neighbors:
            continue
        
        for j in neighbors[i]:
            if j == i:
                continue
            
            # Edge vector from i to j
            e = np.array(r[j]) - np.array(r[i])
            e_dist = np.linalg.norm(e)
            if e_dist < 1e-10:
                continue
            e = e / e_dist
            
            # Penalty term: dot product with normal squared
            # This penalizes edges perpendicular to boundary (which cut across it)
            alignment = np.dot(e, ni)**2
            coeff = beta * alignment / 4.0
            
            # Add ZZ term (penalizes edge if both endpoints selected)
            zz_key = pauli_ZZ(n, i, j)
            terms[zz_key] = terms.get(zz_key, 0) + coeff
            
            # Add Z terms (encourage selecting boundary nodes)
            z_i = pauli_Z(n, i)
            z_j = pauli_Z(n, j)
            terms[z_i] = terms.get(z_i, 0) - coeff / 2.0
            terms[z_j] = terms.get(z_j, 0) - coeff / 2.0
    
    return terms


def hamiltonian_builder(
    phi,r,L,
    alpha = 0, band = 0 , #domain
    gamma = 0, #spacing
    use_sparsity=False, N = None , mu = 0, #sparsity (count) penalty
    use_repulsion=False, d_min = 0, eta = 100, # repulsion penalty 
    use_bend=False, kappa=1.0, #bend penalty 
    use_max_edge = False, d_max = 0 , eta_max = 0, # max edge length penalty 
    use_density_field=False, density_radius=None, gamma_density=0.0, # density penalty for compactness
    use_angular_bins=False, num_angular_bins=6, eta_theta=0.0, #angular bins penalty to get equailateral triangle behavious
    use_collinearity_penalty=False, eta_col=0.0,#colinearity penalty 
    use_boundary_alignment=False, boundary_nodes=None, beta=0.0, #boundary geometry preserving penalty( only for boundary nodes)
    normalize=True,  # Enable normalization
    tuning_factors=None,  #  Dict of tuning factors per penalty
    ):


    n = len(phi)

    H_terms = {}

#Penalty coefficient tuning 
    default_tuning = {
        'domain': 1.0, 'spacing': 1.0, 'sparsity': 1.0, 'bend': 1.0,
        'max_edge': 1.0, 'density': 1.0, 'angular_bins': 1.0,
        'collinearity': 1.0, 'boundary_alignment': 1.0
    }
    if tuning_factors is not None:
        default_tuning.update(tuning_factors)
    tuning = default_tuning

    dmat = compute_distance_matrix(r)
    neighbors_dict = build_radius_neighbors(r, radius=3*L, dmat=dmat)
    neighbor_pairs = []
    for i in neighbors_dict:
        for j in neighbors_dict[i]:
            if i < j:
                neighbor_pairs.append((i, j))

    untuned_penalties = {}

    untuned_penalties['domain'] = domain_penalty_strings(phi, alpha, band)

    untuned_penalties['spacing'] = spacing_penalty_strings(r, neighbor_pairs, L, gamma)

    if use_sparsity and N is not None:
        untuned_penalties['sparsity'] = sparsity_penalty_strings(n, N, mu)

    if use_repulsion:
        untuned_penalties['repulsion'] = repulsion_penalty_strings(r, d_min, eta)

    if use_bend:
        bend_triples = build_radius_bend_triples(r, 3*L, dmat=dmat)
        untuned_penalties['bend'] = bend_penalty_strings(r, bend_triples, kappa)

    if use_max_edge:
        untuned_penalties['max_edge'] = max_edge_penalty_strings(r, d_max, eta_max, dmat=dmat)

    if use_density_field and density_radius is not None:
        density_field = phi_circle_field_local(r, density_radius)
        untuned_penalties['density'] = domain_penalty_strings(density_field, gamma_density, 0)

    if use_angular_bins and num_angular_bins > 0:
        untuned_penalties['angular_bins'] = angular_bins_penalty_strings(r, int(num_angular_bins), eta_theta)

    if use_collinearity_penalty:
        collinear_pairs = build_collinear_pairs(r, neighbors_dict)
        if collinear_pairs:
            untuned_penalties['collinearity'] = collinearity_penalty_strings(n, collinear_pairs, eta_col)

    if use_boundary_alignment and boundary_nodes is not None:
        boundary_normals = compute_boundary_normals(r, boundary_nodes, k=4)
        untuned_penalties['boundary_alignment'] = boundary_alignment_penalty_strings(r, boundary_nodes, boundary_normals, neighbors_dict, beta)


    penalty_norms = {}
    for name, terms in untuned_penalties.items():
        if terms:
            norm = np.sqrt(sum(c**2 for c in terms.values()))
            penalty_norms[name] = norm if norm > 1e-10 else 1.0
        else:
            penalty_norms[name] = 1.0

    for name, terms in untuned_penalties.items():
        
        if not terms:
            continue
        factor = tuning.get(name, 1.0)
        norm = penalty_norms[name]

        if normalize:
            scale = factor / norm
        else:
            scale = factor

        for key, coeff in terms.items():
            H_terms[key] = H_terms.get(key, 0.0) + coeff * scale

    pauli_keys = list(H_terms.keys())
    pauli_coeffs = list(H_terms.values())

    return SparsePauliOp(pauli_keys, pauli_coeffs)

        


