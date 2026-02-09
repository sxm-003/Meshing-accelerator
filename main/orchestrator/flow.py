# orchestrator/flow.py
from prefect import flow, task
import os
import numpy as np

from node_manager.crude_generator import generate_crude_nodes
from node_manager.patch_generator import (
    generate_patch,)
from node_manager.gaussian_patch_merger import (
    merge_patch_results_gaussian,
    prepare_patch_for_qaoa,
)

from quantum_processing.hamiltonian_builder import (
    hamiltonian_builder,
    phi_circle_field_local,
)
from orchestrator.patch_record import PatchRecord
from prefect_dask.task_runners import DaskTaskRunner
from prefect.context import get_run_context
from pathlib import Path

from quantum_processing.qaoa_aer_pipeline import run_qaoa_aer
from orchestrator.visualize_patch_output import (
    patch_traces,
    combined_figure,
    single_patch_figure,
)
import matplotlib
matplotlib.use("Agg")          # non-interactive backend safe for Dask workers
import matplotlib.pyplot as plt




@task
def generate_nodes_task(dxf_path, jitter_factor=0.0):
    """
    Generate nodes from DXF file with optional jitter.
    
    Returns nodes and the global indices of CAD boundary nodes.
    These are the nodes sampled directly on the DXF geometry boundary.
    
    Args:
        dxf_path: Path to DXF file
        jitter_factor: Random jitter (0.0=uniform grid, 1.0=full jitter)
    
    Returns:
        nodes: (N, 2) full node array
        cad_boundary_idx: Global indices of CAD boundary nodes in the nodes array
    """
    nodes, interior_nodes, offset_nodes, boundary_nodes = generate_crude_nodes(
        dxf_path, jitter_factor=jitter_factor
    )
    
    # CAD boundary nodes are stacked last: [interior, offset, boundary]
    n_interior = len(interior_nodes)
    n_offset = len(offset_nodes)
    n_boundary = len(boundary_nodes)
    cad_boundary_idx = np.arange(n_interior + n_offset, n_interior + n_offset + n_boundary)
    
    return nodes, cad_boundary_idx


@task
def generate_patches_task(nodes, L, Q_max, overlap_factor=1.0, cad_boundary_idx=None):
    """
    Generate patches with configurable overlap.
    
    Args:
        nodes: Node coordinates
        L: Resolution / characteristic length
        Q_max: Maximum qubits per patch
        overlap_factor: Overlap control (0.0=no overlap, 1.0=standard, >1.0=more overlap)
        cad_boundary_idx: Global indices of CAD boundary nodes (from DXF geometry)
    """
    return generate_patch(L, nodes, Q_max, overlap_factor=overlap_factor,
                          cad_boundary_idx=cad_boundary_idx)


@task
def build_patch_records(nodes, patches):
    """
    Build patch records from patches, including CAD boundary node mapping.
    
    Args:
        nodes: Full node set
        patches: List of patch dictionaries with interior_idx, halo_idx,
                 and cad_boundary_idx_local (local indices of CAD boundary nodes)
    
    Returns:
        records: List of PatchRecord objects
    """
    records = []
    for p in patches:
        # Get interior and halo nodes
        interior_idx = p["interior_idx"]
        halo_idx = p.get("halo_idx", [])
        
        # Combine for full patch
        all_idx = np.concatenate([interior_idx, halo_idx]) if len(halo_idx) > 0 else interior_idx
        patch_nodes = nodes[all_idx]
        
        if len(patch_nodes) == 0:
            continue
        
        # Get CAD boundary node indices (local to patch)
        boundary_idx_local = p.get("cad_boundary_idx_local", None)
        
        records.append(PatchRecord(
            patch_nodes=patch_nodes,
            boundary_nodes_idx=boundary_idx_local
        ))
    
    # Diagnostic: show patch qubit counts so user can verify Q_max is respected
    sizes = [len(r.patch_nodes) for r in records]
    if sizes:
        print(f"\n  Patch summary: {len(records)} patches, "
              f"qubit counts: min={min(sizes)}, max={max(sizes)}, "
              f"avg={np.mean(sizes):.1f}")
    return records


@task()
def build_hamiltonian_task(record: PatchRecord, ham_dir: str, rec_dir: str):
    """
    Build Hamiltonian for a patch with optional boundary alignment penalty.
    
    If the patch contains boundary nodes, the boundary alignment penalty
    will be automatically enabled to preserve boundary geometry.
    """
    center = np.array(record.patch_nodes).mean(axis=0)
    dists = np.linalg.norm(np.array(record.patch_nodes) - center, axis=1)
    L = np.mean(np.linalg.norm(np.array(record.patch_nodes) - np.roll(np.array(record.patch_nodes), 
                                                                      shift=1, axis=0),axis=1))
    R = np.percentile(dists, 80)
    phi = phi_circle_field_local(record.patch_nodes, R=1.0)
    band = 0.8* R

    # Check if patch has boundary nodes
    has_boundary = (record.boundary_nodes_idx is not None and 
                   len(record.boundary_nodes_idx) > 0)
    
    # Enable boundary alignment if boundary nodes present
    boundary_nodes = record.boundary_nodes_idx if has_boundary else None

    tuning_params = { 'domain': 1.0, 'spacing': 1.0, 'sparsity': 1.0, 'bend': 1.0,
        'max_edge': 1.0, 'density': 1.0, 'angular_bins': 1.0,
        'collinearity': 1.0, 'boundary_alignment': 1.0
    }

    H, decomposition = hamiltonian_builder(
        phi=phi,
        r=record.patch_nodes,

    # geometric scale
        L=L,
    #domain constraint 
        alpha=10,band=band,
    # spacing 
        gamma=0,
    #  sparsity 
        use_sparsity=True,
        N=int(0.9 * len(phi)),   
        mu=0.25,
    #  short-range repulsion 
        use_repulsion=False,
        d_min=0.125,          
        eta=0.8,
    #  bend / angle preservation 
        use_bend=True,
        kappa=3.0,
    # max edge length
        use_max_edge=True,
        d_max=1.2*L,
        eta_max=40,
    # density regularization
        use_density_field=True,
        density_radius= 0.5*L,
        gamma_density=20,
    # angular distribution regularization
        use_angular_bins=True,
        num_angular_bins=6,
        eta_theta=20,
    # collinearity regularization
        use_collinearity_penalty=True,
        eta_col=20,
    # boundary alignment (auto-enabled if boundary nodes present)
        use_boundary_alignment=has_boundary,
        boundary_nodes=boundary_nodes,
        beta=50.0,  # Increased weight for boundary preservation
    # normalization and tuning
        normalize=True,
        tuning_factors=tuning_params,
    # return per-penalty breakdown for visualization
        return_decomposition=True,
        )

    record.phi = phi
    record.decomposition = decomposition


    ham_path = os.path.join(ham_dir, f"{record.patch_id}.npz")

    np.savez(
        ham_path,
        paulis=H.paulis.to_labels(),
        coeffs=H.coeffs,
    )

    # Save decomposition alongside Hamiltonian for later visualization
    decomp_path = os.path.join(ham_dir, f"{record.patch_id}_decomp.npz")
    np.savez(
        decomp_path,
        penalty_names=list(decomposition['scaled_norms'].keys()),
        scaled_norms=list(decomposition['scaled_norms'].values()),
        raw_norms=list(decomposition['penalty_norms'].values()),
        tuning_factors=[decomposition['tuning_factors'].get(k, 1.0)
                        for k in decomposition['scaled_norms'].keys()],
        n_qubits=decomposition['n_qubits'],
    )

    record.hamiltonian_path = ham_path
    record.save(rec_dir)
    return record


@task
def visualize_hamiltonian_coefficients(built_records, base_dir: str):
    """
    Produce a 4-panel Hamiltonian coefficient breakdown figure
    (matching the style in updated_energy_func_test_ENHANCED.ipynb).

    Aggregates decomposition data across all patches and saves
    the figure to <base_dir>/hamiltonian_coefficients.png.
    """
    # --- Collect decomposition data from all records ---
    all_scaled_norms = {}   # penalty_name -> list of norms across patches
    all_raw_norms = {}
    all_tuning = {}
    all_coeffs_by_penalty = {}   # penalty_name -> flat list of |coeff|
    total_qubits = []

    for rec in built_records:
        decomp = getattr(rec, 'decomposition', None)
        if decomp is None:
            continue
        total_qubits.append(decomp['n_qubits'])
        for name, norm in decomp['scaled_norms'].items():
            all_scaled_norms.setdefault(name, []).append(norm)
        for name, norm in decomp['penalty_norms'].items():
            all_raw_norms.setdefault(name, []).append(norm)
        for name, tf in decomp['tuning_factors'].items():
            all_tuning[name] = tf
        for name, terms in decomp['scaled_penalties'].items():
            all_coeffs_by_penalty.setdefault(name, []).extend(
                np.abs(list(terms.values()))
            )

    if not all_scaled_norms:
        print("  ⚠ No decomposition data — skipping Hamiltonian viz.")
        return

    # Averages across patches
    names = sorted(all_scaled_norms.keys())
    avg_scaled = [np.mean(all_scaled_norms[n]) for n in names]
    avg_raw = [np.mean(all_raw_norms.get(n, [0])) for n in names]
    tuning_vals = [all_tuning.get(n, 1.0) for n in names]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    colors = plt.cm.Set3(np.linspace(0, 1, len(names)))

    # ── Panel 1: Scaled contribution norms (bar chart) ──
    ax = axes[0, 0]
    bars = ax.bar(names, avg_scaled, color=colors, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Norm of Scaled Coefficients', fontsize=11, fontweight='bold')
    ax.set_title('Penalty Contribution Magnitudes (normalised + tuned)',
                 fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, avg_scaled):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)

    # ── Panel 2: Number of Pauli terms per penalty ──
    ax = axes[0, 1]
    term_counts = [len(all_coeffs_by_penalty.get(n, [])) for n in names]
    bars = ax.bar(names, term_counts, color=colors, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Number of Pauli Terms (all patches)', fontsize=11, fontweight='bold')
    ax.set_title('Total Generated Pauli Strings per Penalty',
                 fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, cnt in zip(bars, term_counts):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                f'{int(cnt)}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)

    # ── Panel 3: Raw vs Scaled norms (grouped bar) ──
    ax = axes[1, 0]
    x = np.arange(len(names))
    width = 0.35
    ax.bar(x - width / 2, avg_raw, width, label='Raw (untuned) Norm',
           color='steelblue', alpha=0.8, edgecolor='black')
    ax.bar(x + width / 2, avg_scaled, width, label='Scaled (normalised + tuned)',
           color='coral', alpha=0.8, edgecolor='black')
    ax.set_ylabel('Norm Value', fontsize=11, fontweight='bold')
    ax.set_title('Raw vs Scaled Contribution Norms', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # ── Panel 4: Coefficient magnitude histogram ──
    ax = axes[1, 1]
    flat_coeffs = []
    for n in names:
        flat_coeffs.extend(all_coeffs_by_penalty.get(n, []))
    if flat_coeffs:
        arr = np.array(flat_coeffs)
        ax.hist(arr, bins=40, color='steelblue', edgecolor='black', alpha=0.7)
        ax.set_xlabel('|Coefficient| Magnitude', fontsize=11, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax.set_title('Distribution of Coefficient Magnitudes (all patches)',
                     fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        stats_text = (f'Mean: {arr.mean():.4f}\n'
                      f'Std:  {arr.std():.4f}\n'
                      f'Max:  {arr.max():.4f}')
        ax.text(0.98, 0.97, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    out_path = os.path.join(base_dir, "hamiltonian_coefficients.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\n✓ Hamiltonian coefficient breakdown saved to {out_path}")


@task(
    tags=["qaoa-aer"],
    retries=1,
    retry_delay_seconds=10,

)
def run_qaoa_task(record: PatchRecord, rec_dir: str):
    bitstring, energy = run_qaoa_aer(record.hamiltonian_path)

    record.bitstring = "".join(str(b) for b in bitstring)
    record.energy = energy

    record.save(rec_dir)
    return record


@task
def visualize_task(record: PatchRecord):
    fig = single_patch_figure(
        patch_nodes=record.patch_nodes,
        phi=record.phi,
        bitstring=record.bitstring,
        title=f"Patch {record.patch_id}",
    )
    fig.show()


@task
def merge_patches_gaussian_task(qaoa_records, nodes, L):
    """
    Merge overlapping patch results using Gaussian-weighted interpolation.
    
    Args:
        qaoa_records: List of PatchRecord objects with QAOA results
        nodes: Full node set
        L: Characteristic length scale for boundary threshold
        
    Returns:
        merged_indices: Array of unique global node indices
    """
    print("\n" + "="*70)
    print("GAUSSIAN PATCH MERGING")
    print("="*70)
    
    # Convert PatchRecord objects to patch_results format
    patch_results = []
    for record in qaoa_records:
        if not hasattr(record, 'bitstring') or record.bitstring is None:
            continue
            
        # Parse bitstring to get selected nodes
        bitstring = [int(b) for b in record.bitstring]
        local_selected = [i for i, b in enumerate(bitstring) if b == 1]
        
        # Get patch indices (mapping from local to global)
        # Assuming patch_nodes are stored and we can find their global indices
        patch_indices = []
        for node in record.patch_nodes:
            # Find matching global index
            dists = np.linalg.norm(nodes - node, axis=1)
            closest_idx = np.argmin(dists)
            if dists[closest_idx] < 1e-6:  # Ensure exact match
                patch_indices.append(closest_idx)
        
        if len(patch_indices) > 0:
            patch_result = {
                'patch_id': record.patch_id,
                'local_selected': local_selected,
                'patch_indices': np.array(patch_indices),
            }
            patch_results.append(patch_result)
    
    # Merge using Gaussian weighting
    boundary_threshold = L * 0.5
    merged_indices = merge_patch_results_gaussian(
        patch_results,
        nodes,
        boundary_threshold=boundary_threshold
    )
    
    print(f"\n✓ Gaussian merging complete:")
    print(f"  Input patches: {len(patch_results)}")
    print(f"  Output nodes: {len(merged_indices)}")
    print("="*70)
    
    return merged_indices


@flow(task_runner=DaskTaskRunner( 
    cluster_kwargs={
        "n_workers": 16,  
        "threads_per_worker": 2, 
        "processes": True,
        "memory_limit": "auto",  
        "timeout": "60s",  
        "death_timeout": "120s",  
    }
))
def mesh_hamiltonian_pipeline(
    dxf_path: str,
    L: float = 0.5,
    Q_max: int = 14,
    overlap_factor: float = 1.0,
    jitter_factor: float = 0.0,
    use_gaussian_merging: bool = True,
    parallel_qaoa: bool = True,
):
    """
    QAOA-based mesh optimization pipeline with optional Gaussian patch merging.
    
    Args:
        dxf_path: Path to DXF file with mesh nodes
        L: Characteristic length scale for patch generation
        Q_max: Maximum qubits per patch
        overlap_factor: Controls overlap between patches (0.0=no overlap, 1.0=standard, >1.0=more)
        jitter_factor: Random jitter for node generation (0.0=uniform grid, 1.0=full jitter)
        use_gaussian_merging: Enable Gaussian-weighted merging of overlapping patches
        parallel_qaoa: If True, dispatch QAOA tasks to Dask workers in parallel.
                       If False (default), run QAOA sequentially to avoid Aer/OpenMP
                       conflicts with Dask multiprocess workers.
    """

    ctx = get_run_context()
    run_id = str(ctx.flow_run.id)
    
    base_dir = Path("outputs") / run_id

    ham_dir = base_dir / "hamiltonians"
    rec_dir = base_dir / "records"

    ham_dir.mkdir(parents=True, exist_ok=True)
    rec_dir.mkdir(parents=True, exist_ok=True)

    # --- pipeline ---
    nodes, cad_boundary_idx = generate_nodes_task(dxf_path, jitter_factor=jitter_factor)
    patches = generate_patches_task(nodes, L, Q_max, overlap_factor=overlap_factor,
                                     cad_boundary_idx=cad_boundary_idx)
    records = build_patch_records(nodes, patches)

    ham_futures = []
    for r in records:
        ham_futures.append(
            build_hamiltonian_task.submit(
                r,
                str(ham_dir),
                str(rec_dir),
            )
        )

    built_records = [f.result() for f in ham_futures]

    # QAOA — parallel or sequential based on user preference.
    # Sequential avoids Aer/OpenMP conflicts with Dask multiprocess workers;
    # parallel is faster but may deadlock on some systems.
    #
    # Strip the heavy decomposition dict before passing to QAOA — it only
    # needs hamiltonian_path and the decomposition would bloat every
    # serialize/deserialize round-trip for no reason.
    qaoa_records = []
    if parallel_qaoa:
        qaoa_futures = []
        for r in built_records:
            r_light = PatchRecord(
                patch_nodes=r.patch_nodes,
                phi=r.phi,
                boundary_nodes_idx=r.boundary_nodes_idx,
            )
            r_light.hamiltonian_path = r.hamiltonian_path
            qaoa_futures.append(
                run_qaoa_task.submit(r_light, str(rec_dir))
            )
        qaoa_records = [f.result() for f in qaoa_futures]
    else:
        for r in built_records:
            r_light = PatchRecord(
                patch_nodes=r.patch_nodes,
                phi=r.phi,
                boundary_nodes_idx=r.boundary_nodes_idx,
            )
            r_light.hamiltonian_path = r.hamiltonian_path
            qaoa_records.append(run_qaoa_task(r_light, str(rec_dir)))

    # --- Gaussian patch merging (optional) ---
    if use_gaussian_merging:
        merged_indices = merge_patches_gaussian_task(qaoa_records, nodes, L)
        print(f"\n✓ Merged mesh contains {len(merged_indices)} unique nodes")
        
        # Save merged indices
        merged_path = base_dir / "merged_indices.npy"
        np.save(merged_path, merged_indices)
        print(f" Saved merged indices to {merged_path}")

    # --- Hamiltonian coefficient visualization ---
    visualize_hamiltonian_coefficients(built_records, str(base_dir))

    # --- Visualization ---
    all_traces = []

    for r in qaoa_records:
        traces = patch_traces(
            patch_nodes=r.patch_nodes,
            phi=r.phi,
            bitstring=r.bitstring,
            patch_id=r.patch_id,
            show_phi=False,
    )
        all_traces.extend(traces)

    # ---- ONE combined plot ----
    fig_all = combined_figure(
        all_traces,
        title="All patches: selected nodes",
    )
    fig_all.show()
    
    # Return merged indices if available
    if use_gaussian_merging:
        return merged_indices
    else:
        return qaoa_records

if __name__ == "__main__":
    mesh_hamiltonian_pipeline("data/sample.dxf")
