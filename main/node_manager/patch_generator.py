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

def identify_boundary_nodes_in_patch(patch_nodes, center, percentile=85):
    """
    Identify boundary nodes in a patch based on distance from center.
    
    Nodes in the outer percentile of distances are considered boundary nodes.
    This is similar to the approach in Airfoil_QAOA notebook.
    
    Args:
        patch_nodes: (N, 2) array of patch node coordinates
        center: (2,) patch center coordinates
        percentile: Distance percentile threshold (default 85)
    
    Returns:
        boundary_idx: Local indices of boundary nodes within patch
    """
    if len(patch_nodes) == 0:
        return np.array([], dtype=int)
    
    # Compute distances from patch center
    dists = np.linalg.norm(patch_nodes - center, axis=1)
    
    # Nodes beyond the percentile threshold are boundary nodes
    threshold = np.percentile(dists, percentile)
    boundary_idx = np.where(dists >= threshold)[0]
    
    return boundary_idx


def generate_patches_with_overlap(nodes, centers, r_patch, r_halo, Q_max=None, overlap_factor=1.0):
    """
    Generate patches with interior and halo regions for overlap handling.
    
    This is the primary patch generation function with configurable overlap between patches.
    
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
        
    Returns:
        patches: List of patch dictionaries with 'center', 'interior_idx', 'halo_idx', 'patch_id', 'boundary_idx'
    """
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
        
        # Combine interior and halo for full patch
        all_patch_idx = np.concatenate([interior_idx, halo_idx])
        patch_nodes = nodes[all_patch_idx]
        
        # Identify boundary nodes within this patch (local indices)
        boundary_idx_local = identify_boundary_nodes_in_patch(patch_nodes, center, percentile=85)
        
        patch = {
            "center": center,
            "interior_idx": interior_idx,
            "halo_idx": halo_idx,
            "patch_id": ci,
            "boundary_idx_local": boundary_idx_local  # Local indices relative to patch nodes
        }
        patches.append(patch)
    
    return patches


def generate_patch(L, nodes, Q_max, overlap_factor=1.0):
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
    
    Returns:
        patches: List of patch dictionaries
    """
    r_patch, r_halo, d_min = compute_patch_radii(nodes, L, Q_max)
    centers = generate_patch_centers(nodes, r_patch)
    patches = generate_patches_with_overlap(
        nodes, centers, r_patch, r_halo, Q_max, overlap_factor=overlap_factor
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
