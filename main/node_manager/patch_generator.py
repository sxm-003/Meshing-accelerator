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

def build_patches(nodes, centers, r_patch, r_halo, Q_max):
    patches = []

    for ci, C in enumerate(centers):
        d = distances(nodes, C)

        interior_idx = np.where(d <= r_patch)[0]
        halo_idx = np.where((d > r_patch) & (d <= r_halo))[0]

        # Enforce qubit limit
        if len(interior_idx) > Q_max:
            interior_idx = interior_idx[:Q_max]

        patch = {
            "center": C,
            "interior_idx": interior_idx,
            "halo_idx": halo_idx
        }
        patches.append(patch)

    return patches

def generate_patch(L, nodes, Q_max):
    #L is resolution
    r_patch, r_halo, d_min = compute_patch_radii(nodes, L, Q_max)
    centers = generate_patch_centers(nodes, r_patch)
    patches = build_patches(nodes, centers, r_patch, r_halo, Q_max)

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
