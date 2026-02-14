# orchestrator/flow.py
from prefect import flow, task
import os
import numpy as np
from collections import deque

from node_manager.crude_generator import (
    generate_crude_nodes,
    load_dxf, extract_segments, segments_to_polygons,
)
from node_manager.adaptive_generator import generate_adaptive_nodes
from node_manager.patch_generator import (
    generate_patch,)
from node_manager.gaussian_patch_merger import (
    merge_patch_results_gaussian,
    prepare_patch_for_qaoa,
)
from node_manager.mesh_builder import build_and_save_mesh

from quantum_processing.hamiltonian_builder import (
    hamiltonian_builder,
    phi_circle_field_local,
)
from orchestrator.patch_record import PatchRecord
from prefect_dask.task_runners import DaskTaskRunner
from prefect.context import get_run_context
from pathlib import Path
from typing import Optional

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
def generate_nodes_task(dxf_path, jitter_factor=0.0, adaptive=False,
                        L=None, L_fine=None, L_coarse=None,
                        boundary_band=None, curvature_weight=0.5):
    """
    Generate nodes from DXF file.
    
    Two modes:
      - adaptive=False (default): uniform grid via generate_crude_nodes
      - adaptive=True:  spatially varying density — finer near boundaries
                        and high-curvature regions, coarser in interiors
    
    Returns:
        nodes: (N, 2) full node array
        cad_boundary_idx: Global indices of CAD boundary nodes
    """
    if adaptive:
        nodes, interior_nodes, offset_nodes, boundary_nodes = generate_adaptive_nodes(
            dxf_path,
            L=L,
            L_fine=L_fine,
            L_coarse=L_coarse,
            boundary_band=boundary_band,
            curvature_weight=curvature_weight,
            jitter_factor=jitter_factor,
        )
        print(f"\n  Adaptive node generation: {len(nodes)} nodes "
              f"(interior={len(interior_nodes)}, offset={len(offset_nodes)}, "
              f"boundary={len(boundary_nodes)})")
    else:
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
            boundary_nodes_idx=boundary_idx_local,
            global_indices=np.asarray(all_idx, dtype=np.intp),
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

    tuning_params = { 'domain': 0.9, 'spacing': 0.8, 'sparsity': 1.5, 'bend': 1.8,
        'max_edge': 1.7, 'density': 0, 'angular_bins': 1.3,
        'collinearity': 1.8, 'boundary_alignment': 1.0
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


    ham_path = os.path.join(ham_dir, f"{record.patch_id}.npz")

    sparse_terms = H.to_sparse_list()
    sparse_ops = np.asarray([t[0] for t in sparse_terms], dtype="U2")
    sparse_positions = np.full((len(sparse_terms), 2), -1, dtype=np.int32)
    sparse_coeffs = np.asarray([complex(t[2]) for t in sparse_terms], dtype=np.complex128)
    for idx, (_, pos, _) in enumerate(sparse_terms):
        sparse_positions[idx, :len(pos)] = np.asarray(pos, dtype=np.int32)

    np.savez(
        ham_path,
        sparse_ops=sparse_ops,
        sparse_positions=sparse_positions,
        coeffs=sparse_coeffs,
        num_qubits=np.asarray([H.num_qubits], dtype=np.int32),
    )

    # Save decomposition alongside Hamiltonian for later visualization
    decomp_path = os.path.join(ham_dir, f"{record.patch_id}_decomp.npz")
    penalty_names = list(decomposition["scaled_norms"].keys())
    scaled_norms = [decomposition["scaled_norms"].get(k, 0.0) for k in penalty_names]
    raw_norms = [decomposition["penalty_norms"].get(k, 0.0) for k in penalty_names]
    tuning_factors = [decomposition["tuning_factors"].get(k, 1.0) for k in penalty_names]
    scaled_penalties = decomposition.get("scaled_penalties", {})
    term_counts = []
    coeff_magnitudes = []
    for name in penalty_names:
        penalty_terms = scaled_penalties.get(name, {})
        if isinstance(penalty_terms, dict):
            term_counts.append(len(penalty_terms))
            coeff_magnitudes.extend(np.abs(list(penalty_terms.values())))
        else:
            term_counts.append(len(penalty_terms))
            coeff_magnitudes.extend(abs(float(t[2])) for t in penalty_terms)

    np.savez_compressed(
        decomp_path,
        penalty_names=np.asarray(penalty_names),
        scaled_norms=np.asarray(scaled_norms, dtype=float),
        raw_norms=np.asarray(raw_norms, dtype=float),
        tuning_factors=np.asarray(tuning_factors, dtype=float),
        term_counts=np.asarray(term_counts, dtype=int),
        coeff_magnitudes=np.asarray(coeff_magnitudes, dtype=float),
        n_qubits=np.asarray([decomposition["n_qubits"]], dtype=int),
    )

    record.hamiltonian_path = ham_path
    record.decomposition_path = decomp_path
    # Keep Dask payload small: full decomposition is persisted in decomp_path.
    record.decomposition = None
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
    all_term_counts = {}         # penalty_name -> total term count across patches
    all_coeff_magnitudes = []    # flat list of |coeff| across all penalties/patches
    total_qubits = []

    for rec in built_records:
        decomp = getattr(rec, 'decomposition', None)
        if decomp is not None:
            total_qubits.append(decomp['n_qubits'])
            for name, norm in decomp['scaled_norms'].items():
                all_scaled_norms.setdefault(name, []).append(norm)
            for name, norm in decomp['penalty_norms'].items():
                all_raw_norms.setdefault(name, []).append(norm)
            for name, tf in decomp['tuning_factors'].items():
                all_tuning[name] = tf

            if 'scaled_penalties' in decomp:
                for name, terms in decomp['scaled_penalties'].items():
                    if isinstance(terms, dict):
                        mags = np.abs(list(terms.values()))
                    else:
                        mags = [abs(float(t[2])) for t in terms]
                    all_term_counts[name] = all_term_counts.get(name, 0) + len(mags)
                    all_coeff_magnitudes.extend(mags)
            elif 'term_counts' in decomp:
                for name, cnt in decomp['term_counts'].items():
                    all_term_counts[name] = all_term_counts.get(name, 0) + int(cnt)
                all_coeff_magnitudes.extend(decomp.get('coeff_magnitudes', []))
            continue

        decomp_path = getattr(rec, "decomposition_path", None)
        if not decomp_path:
            continue
        if not os.path.exists(decomp_path):
            continue

        with np.load(decomp_path, allow_pickle=False) as npz:
            names = npz["penalty_names"].tolist()
            scaled_norms = npz["scaled_norms"].tolist()
            raw_norms = npz["raw_norms"].tolist()
            tuning_factors = npz["tuning_factors"].tolist()
            term_counts = npz["term_counts"].tolist() if "term_counts" in npz else [0] * len(names)
            coeff_magnitudes = npz["coeff_magnitudes"].tolist() if "coeff_magnitudes" in npz else []
            n_qubits = int(npz["n_qubits"][0]) if "n_qubits" in npz else None

        if n_qubits is not None:
            total_qubits.append(n_qubits)

        for i, name in enumerate(names):
            all_scaled_norms.setdefault(name, []).append(float(scaled_norms[i]))
            all_raw_norms.setdefault(name, []).append(float(raw_norms[i]))
            all_tuning[name] = float(tuning_factors[i])
            all_term_counts[name] = all_term_counts.get(name, 0) + int(term_counts[i])
        all_coeff_magnitudes.extend(float(c) for c in coeff_magnitudes)

    if not all_scaled_norms:
        print("  ⚠ No decomposition data — skipping Hamiltonian viz.")
        return

    # Averages across patches
    names = sorted(all_scaled_norms.keys())
    avg_scaled = [np.mean(all_scaled_norms[n]) for n in names]
    avg_raw = [np.mean(all_raw_norms.get(n, [0])) for n in names]

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
    term_counts = [all_term_counts.get(n, 0) for n in names]
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
    flat_coeffs = all_coeff_magnitudes
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
    retries=1,
    retry_delay_seconds=10,
)
def run_qaoa_task(
    record: PatchRecord,
    rec_dir: str,
    aer_max_parallel_threads: int = 1,
    aer_max_parallel_experiments: int = 1,
    aer_max_parallel_shots: int = 1,
    log_backend_config: bool = False,
):
    bitstring, energy = run_qaoa_aer(
        record.hamiltonian_path,
        aer_max_parallel_threads=aer_max_parallel_threads,
        aer_max_parallel_experiments=aer_max_parallel_experiments,
        aer_max_parallel_shots=aer_max_parallel_shots,
        log_backend_config=log_backend_config,
    )

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
        
        # Use stored global indices directly (O(1) lookup instead of O(P×N) search)
        if record.global_indices is not None:
            patch_indices = record.global_indices
        else:
            # Fallback: brute-force distance search (legacy path)
            patch_indices = []
            for node in record.patch_nodes:
                dists = np.linalg.norm(nodes - node, axis=1)
                closest_idx = np.argmin(dists)
                if dists[closest_idx] < 1e-6:
                    patch_indices.append(closest_idx)
            patch_indices = np.array(patch_indices)
        
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


@task
def build_mesh_task(nodes, merged_indices, dxf_path, output_dir,
                    cad_boundary_idx=None, smooth_iterations=5,
                    formats=("msh", "vtk", "obj")):
    """
    Triangulate selected nodes → Laplacian smooth → save mesh files.
    
    Args:
        nodes: Full node set
        merged_indices: Global indices of selected nodes from QAOA+merge
        dxf_path: DXF path (for polygon extraction / triangle clipping)
        output_dir: Where to save mesh files
        cad_boundary_idx: Boundary node indices (fixed during smoothing)
        smooth_iterations: Laplacian smoothing passes
        formats: Export formats
    
    Returns:
        mesh_info dict with nodes, triangles, quality, files
    """
    # Extract polygons for triangle filtering
    msp = load_dxf(str(dxf_path))
    polygons = None
    if msp is not None:
        segments = extract_segments(msp)
        polygons = segments_to_polygons(segments)

    mesh_info = build_and_save_mesh(
        nodes=nodes,
        selected_indices=merged_indices,
        output_dir=output_dir,
        polygons=polygons,
        smooth_iterations=smooth_iterations,
        boundary_node_indices=cad_boundary_idx,
        formats=formats,
    )

    q = mesh_info["quality"]
    print(f"\n" + "="*70)
    print(f"MESH CREATED")
    print(f"="*70)
    print(f"  Nodes:            {q.get('n_nodes', '?')}")
    print(f"  Elements:         {q.get('n_elements', '?')}")
    print(f"  Min Angle:        {q.get('min_angle', '?'):.2f}°")
    print(f"  Mean Min Angle:   {q.get('mean_min_angle', '?'):.2f}°")
    print(f"  Mean Aspect Ratio:{q.get('mean_aspect_ratio', '?'):.3f}")
    print(f"  Mean Skewness:    {q.get('mean_skewness', '?'):.4f}")
    print(f"  Saved files:")
    for fp in mesh_info["files"]:
        print(f"    → {fp}")
    print(f"="*70)

    return mesh_info


@flow(task_runner=DaskTaskRunner( 
    cluster_kwargs={
        "n_workers": 1,  
        "threads_per_worker": 1, 
        "processes": True,
        "memory_limit": "auto",  
        "timeout": "120s",  
        "death_timeout": "240s",  
    }
))
def mesh_hamiltonian_pipeline(
    dxf_path: str,
    L: float = 0.5,
    Q_max: int = 50,
    overlap_factor: float = 1.0,
    jitter_factor: float = 0.0,
    use_gaussian_merging: bool = True,
    hamiltonian_concurrency: int = 64,
    parallel_qaoa: bool = True,
    qaoa_concurrency: int = 4,
    qaoa_aer_max_parallel_threads: int = 2,
    qaoa_aer_max_parallel_experiments: int = 2,
    qaoa_aer_max_parallel_shots: int = 2,
    qaoa_log_backend_config: bool = False,
    adaptive_nodes: bool = True,
    L_fine: Optional[float] = None,
    L_coarse: Optional[float] = None,
    curvature_weight: float = 0.5,
    smooth_iterations: int = 5,
    export_formats: tuple = ("msh", "vtk", "obj"),
):
    """
    QAOA-based mesh optimization pipeline.
    
    Full pipeline: DXF → nodes → patches → Hamiltonians → QAOA → merge → mesh → export
    
    Args:
        dxf_path: Path to DXF file with mesh nodes
        L: Characteristic length scale for patch generation
        Q_max: Maximum qubits per patch
        overlap_factor: Controls overlap between patches (0.0=no overlap, 1.0=standard, >1.0=more)
        jitter_factor: Random jitter for node generation (0.0=uniform grid, 1.0=full jitter)
        use_gaussian_merging: Enable Gaussian-weighted merging of overlapping patches
        hamiltonian_concurrency: Max number of in-flight Hamiltonian tasks.
        parallel_qaoa: If True, dispatch QAOA tasks to Dask workers in parallel.
                       If False, run QAOA sequentially to avoid Aer/OpenMP conflicts.
        qaoa_concurrency: Max number of in-flight QAOA tasks when parallel_qaoa=True.
        qaoa_aer_max_parallel_threads: Aer threads per QAOA task.
        qaoa_aer_max_parallel_experiments: Aer experiment-level parallelism.
        qaoa_aer_max_parallel_shots: Aer shot-level parallelism.
        qaoa_log_backend_config: Print Aer/OpenMP config from QAOA tasks.
        adaptive_nodes: If True, use adaptive density node generation (finer near
                       boundaries/curvature, coarser in interior). If False, uniform grid.
        L_fine: Fine spacing for adaptive mode (auto from L if None)
        L_coarse: Coarse spacing for adaptive mode (auto from L if None)
        curvature_weight: How much boundary curvature affects node density (0..1)
        smooth_iterations: Laplacian smoothing passes on final mesh (0=disable)
        export_formats: Mesh file formats to export
    """

    ctx = get_run_context()
    run_id = str(ctx.flow_run.id)
    
    base_dir = Path("outputs") / run_id

    ham_dir = base_dir / "hamiltonians"
    rec_dir = base_dir / "records"

    ham_dir.mkdir(parents=True, exist_ok=True)
    rec_dir.mkdir(parents=True, exist_ok=True)

    # --- pipeline ---
    nodes, cad_boundary_idx = generate_nodes_task(
        dxf_path,
        jitter_factor=jitter_factor,
        adaptive=adaptive_nodes,
        L=L,
        L_fine=L_fine,
        L_coarse=L_coarse,
        curvature_weight=curvature_weight,
    )
    patches = generate_patches_task(nodes, L, Q_max, overlap_factor=overlap_factor,
                                     cad_boundary_idx=cad_boundary_idx)
    records = build_patch_records(nodes, patches)

    if hamiltonian_concurrency < 1:
        raise ValueError("hamiltonian_concurrency must be >= 1")

    ham_futures = deque()
    built_records = []
    for r in records:
        ham_futures.append(
            build_hamiltonian_task.submit(
                r,
                str(ham_dir),
                str(rec_dir),
            )
        )
        # Bound number of in-flight Hamiltonian tasks to avoid overwhelming
        # scheduler/client comms for large DXFs.
        if len(ham_futures) >= hamiltonian_concurrency:
            built_records.append(ham_futures.popleft().result())

    while ham_futures:
        built_records.append(ham_futures.popleft().result())

    # QAOA — parallel or sequential based on user preference.
    # Sequential avoids Aer/OpenMP conflicts with Dask multiprocess workers;
    # parallel is faster but may deadlock on some systems.
    #
    # Strip the heavy decomposition dict before passing to QAOA — it only
    # needs hamiltonian_path and the decomposition would bloat every
    # serialize/deserialize round-trip for no reason.
    qaoa_records = []
    if parallel_qaoa:
        if qaoa_concurrency < 1:
            raise ValueError("qaoa_concurrency must be >= 1")

        qaoa_futures = deque()
        for r in built_records:
            r_light = PatchRecord(
                patch_nodes=r.patch_nodes,
                phi=r.phi,
                boundary_nodes_idx=r.boundary_nodes_idx,
                global_indices=r.global_indices,
            )
            r_light.hamiltonian_path = r.hamiltonian_path
            qaoa_futures.append(
                run_qaoa_task.submit(
                    r_light,
                    str(rec_dir),
                    aer_max_parallel_threads=qaoa_aer_max_parallel_threads,
                    aer_max_parallel_experiments=qaoa_aer_max_parallel_experiments,
                    aer_max_parallel_shots=qaoa_aer_max_parallel_shots,
                    log_backend_config=qaoa_log_backend_config,
                )
            )

            # Bound the number of in-flight QAOA tasks to avoid oversubscription.
            if len(qaoa_futures) >= qaoa_concurrency:
                qaoa_records.append(qaoa_futures.popleft().result())

        while qaoa_futures:
            qaoa_records.append(qaoa_futures.popleft().result())
    else:
        for r in built_records:
            r_light = PatchRecord(
                patch_nodes=r.patch_nodes,
                phi=r.phi,
                boundary_nodes_idx=r.boundary_nodes_idx,
                global_indices=r.global_indices,
            )
            r_light.hamiltonian_path = r.hamiltonian_path
            qaoa_records.append(
                run_qaoa_task(
                    r_light,
                    str(rec_dir),
                    aer_max_parallel_threads=qaoa_aer_max_parallel_threads,
                    aer_max_parallel_experiments=qaoa_aer_max_parallel_experiments,
                    aer_max_parallel_shots=qaoa_aer_max_parallel_shots,
                    log_backend_config=qaoa_log_backend_config,
                )
            )

    # --- Gaussian patch merging (optional) ---
    mesh_info = None
    if use_gaussian_merging:
        merged_indices = merge_patches_gaussian_task(qaoa_records, nodes, L)
        print(f"\n✓ Merged mesh contains {len(merged_indices)} unique nodes")
        
        # Save merged indices
        merged_path = base_dir / "merged_indices.npy"
        np.save(merged_path, merged_indices)
        print(f"  Saved merged indices to {merged_path}")

        # --- Build final mesh: Delaunay triangulate → smooth → export ---
        mesh_dir = base_dir / "mesh"
        mesh_info = build_mesh_task(
            nodes, merged_indices, dxf_path, str(mesh_dir),
            cad_boundary_idx=cad_boundary_idx,
            smooth_iterations=smooth_iterations,
            formats=export_formats,
        )

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
            show_phi=True,
    )
        all_traces.extend(traces)

    # ---- ONE combined plot ----
    fig_all = combined_figure(
        all_traces,
        title="All patches: selected nodes",
    )
    fig_all.show()
    
    # Return results
    if use_gaussian_merging:
        return {
            "merged_indices": merged_indices,
            "mesh_info": mesh_info,
            "qaoa_records": qaoa_records,
        }
    else:
        return {"qaoa_records": qaoa_records}

if __name__ == "__main__":
    mesh_hamiltonian_pipeline("data/test.dxf")
