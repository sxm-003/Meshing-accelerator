import numpy as np 
import plotly.graph_objects as go

def compute_spacing(L, alpha=0.3):
  #L is resolution
    d_min = alpha * L
    return d_min

def domain_bounds(nodes):
    xmin = np.min(nodes[:, 0])
    xmax = np.max(nodes[:, 0])
    ymin = np.min(nodes[:, 1])
    ymax = np.max(nodes[:, 1])
    return xmin, xmax, ymin, ymax

def estimate_density(nodes):
    xmin, xmax, ymin, ymax = domain_bounds(nodes)
    area = (xmax - xmin) * (ymax - ymin)
    rho = len(nodes) / area
    return rho

def compute_patch_radii(nodes, L, Q_max, alpha=0.8):
    d_min = compute_spacing(L, alpha)
    r_int = 2.0 * d_min
#Qmax is max qubits in a patch , adhereing the hardware compatibility
    rho = estimate_density(nodes)
    A_patch = Q_max / rho
    r_patch = np.sqrt(A_patch / np.pi)
    r_halo = r_patch + r_int

    return r_patch, r_halo, d_min

def generate_patch_centers(nodes, r_patch):
    xmin, xmax, ymin, ymax = domain_bounds(nodes)

    spacing = 2.0 * r_patch #spacing between each patch centre(a grid)
    xs = np.arange(xmin + r_patch, xmax, spacing)
    ys = np.arange(ymin + r_patch, ymax, spacing)

    centers = []
    for x in xs:
        for y in ys:
            centers.append([x, y])

    return np.array(centers)

def distances(points, center):
    """
    points : Nx2 array
    center : length-2 array
    """
    diff = points - center
    return np.linalg.norm(diff, axis=1)

def generate_patches_with_overlap(nodes, centers, r_patch, r_halo, Q_max=None,
                                   overlap_factor=1.0, cad_boundary_idx=None):
    """
    Generate patches with interior and halo regions for overlap handling.
    
    This is the primary patch generation function with configurable overlap between patches.
    For each patch, it also identifies which nodes are CAD boundary nodes (nodes that
    lie on the actual DXF geometry boundary), so the boundary alignment penalty can
    anchor the mesh to the CAD shape.
    
    Args:
        nodes: (N, 2) array of node coordinates
        centers: (M, 2) array of patch center coordinates
        r_patch: Radius for interior nodes
        r_halo: Base radius for halo nodes (overlap region)
        Q_max: Maximum nodes per patch (optional, enforces qubit limit)
        overlap_factor: Controls the amount of overlap between patches (default 1.0)
                       - 0.0: No overlap (halo = patch boundary)
                       - 1.0: Standard overlap (halo = r_patch + r_int)
                       - >1.0: Increased overlap for better merging
        cad_boundary_idx: Global indices of CAD boundary nodes (from DXF geometry).
                         If provided, each patch will record which of its local nodes
                         are on the CAD boundary.
        
    Returns:
        patches: List of patch dictionaries with:
            'center', 'interior_idx', 'halo_idx', 'patch_id',
            'cad_boundary_idx_local' (local indices of CAD boundary nodes in this patch)
    """
    # Convert to a set for fast O(1) membership lookup
    cad_boundary_set = set(cad_boundary_idx) if cad_boundary_idx is not None else set()
    
    patches = []
    
    for ci, center in enumerate(centers):
        # Compute distances from center to all nodes
        dists = np.linalg.norm(nodes - center, axis=1)
        
        # Apply overlap factor to halo radius
        # overlap_factor = 0 means no overlap (r_halo_effective = r_patch)
        # overlap_factor = 1 means standard overlap
        overlap_extension = (r_halo - r_patch) * overlap_factor
        r_halo_effective = r_patch + overlap_extension
        
        # Classify nodes as interior or halo
        interior_idx = np.where(dists <= r_patch)[0]
        halo_idx = np.where((dists > r_patch) & (dists <= r_halo_effective))[0]
        
        # Enforce qubit limit if specified
        if Q_max is not None and len(interior_idx) > Q_max:
            # Sort by distance and take closest Q_max nodes
            interior_dists = dists[interior_idx]
            sorted_idx = np.argsort(interior_dists)
            interior_idx = interior_idx[sorted_idx[:Q_max]]
        
        # Cap total patch nodes (interior + halo) at Q_max.
        # The Hamiltonian is built for ALL patch nodes as qubits, so the
        # total must stay within Q_max to keep QAOA simulation tractable.
        # (statevector simulation is O(2^n) — even n=30 is infeasible)
        if Q_max is not None and len(halo_idx) > 0:
            total = len(interior_idx) + len(halo_idx)
            if total > Q_max:
                remaining = Q_max - len(interior_idx)
                if remaining > 0:
                    halo_dists = dists[halo_idx]
                    sorted_halo = np.argsort(halo_dists)
                    halo_idx = halo_idx[sorted_halo[:remaining]]
                else:
                    halo_idx = np.array([], dtype=int)
        
        # Combine interior and halo for full patch (global indices)
        all_patch_idx = np.concatenate([interior_idx, halo_idx]) if len(halo_idx) > 0 else interior_idx
        
        # Find which patch nodes are CAD boundary nodes
        # all_patch_idx contains global indices; check which are in cad_boundary_set
        cad_boundary_local = []
        if len(cad_boundary_set) > 0:
            for local_i, global_i in enumerate(all_patch_idx):
                if global_i in cad_boundary_set:
                    cad_boundary_local.append(local_i)
        
        cad_boundary_local = np.array(cad_boundary_local, dtype=int) if cad_boundary_local else None
        
        patch = {
            "center": center,
            "interior_idx": interior_idx,
            "halo_idx": halo_idx,
            "patch_id": ci,
            "cad_boundary_idx_local": cad_boundary_local  # Local indices of CAD boundary nodes
        }
        patches.append(patch)
    
    return patches


def generate_patch(L, nodes, Q_max, overlap_factor=1.0, cad_boundary_idx=None):
    """
    Generate patches with configurable overlap for QAOA mesh optimization.
    
    Args:
        L: Resolution / characteristic length scale
        nodes: (N, 2) array of node coordinates
        Q_max: Maximum qubits per patch (hardware constraint)
        overlap_factor: Controls overlap between patches (default 1.0)
                       - 0.0: No overlap
                       - 1.0: Standard overlap
                       - >1.0: Increased overlap
        cad_boundary_idx: Global indices of CAD boundary nodes (from DXF geometry)
    
    Returns:
        patches: List of patch dictionaries
    """
    r_patch, r_halo, d_min = compute_patch_radii(nodes, L, Q_max)
    centers = generate_patch_centers(nodes, r_patch)
    patches = generate_patches_with_overlap(
        nodes, centers, r_patch, r_halo, Q_max,
        overlap_factor=overlap_factor,
        cad_boundary_idx=cad_boundary_idx
    )

    return patches


def interactive_patch_view(nodes, patches, max_patches=100):
    fig = go.Figure()

    # All nodes
    fig.add_trace(go.Scattergl(
        x=nodes[:, 0],
        y=nodes[:, 1],
        mode='markers',
        marker=dict(size=2, color='black'),
        name='All nodes'
    ))

    # Patch interiors (limit for clarity)
    for i, p in enumerate(patches[:max_patches]):
        interior = nodes[p["interior_idx"]]
        if len(interior) == 0:
            continue

        fig.add_trace(go.Scattergl(
            x=interior[:, 0],
            y=interior[:, 1],
            mode='markers',
            marker=dict(size=5),
            name=f'Patch {i}'
        ))

    fig.update_layout(
        width=800,
        height=800,
        title="Interactive patch interiors (zoom & pan)",
        xaxis=dict(scaleanchor="y"),
        yaxis=dict(),
        showlegend=False
    )

    fig.show()
