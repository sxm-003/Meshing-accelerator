# orchestrator/flow.py
from prefect import flow, task
import os
import numpy as np
import subprocess
import time
import json 
from collections import deque

from scipy.spatial import cKDTree

from node_manager.crude_generator import (
    generate_crude_nodes,
    load_dxf, extract_segments, segments_to_polygons,
)
from node_manager.adaptive_generator import generate_adaptive_nodes
from node_manager.critical_region_manager import build_hybrid_region_patches
from node_manager.gaussian_patch_merger import (
    merge_patch_results_gaussian,
)
from node_manager.mesh_builder import build_and_save_mesh

from quantum_processing.hamiltonian_builder import (
    hamiltonian_builder,
    phi_circle_field_local,
    compute_L,
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
def generate_region_patches_task(
    nodes,
    L,
    Q_max,
    overlap_factor=1.0,
    cad_boundary_idx=None,
    use_critical_regions=True,
    curvature_threshold_percentile=90.0,
    min_angle_threshold=15.0,
    edge_ratio_threshold=4.0,
    normal_region_qmax=None,
):
    """
    Split the mesh into critical/normal regions and generate patch sets.

    Critical patches are used for QAOA; normal patches are tagged for
    classical Delaunay handling.
    """
    region_data = build_hybrid_region_patches(
        nodes=nodes,
        L=L,
        Q_max=Q_max,
        overlap_factor=overlap_factor,
        cad_boundary_idx=cad_boundary_idx,
        use_critical_regions=use_critical_regions,
        curvature_threshold_percentile=curvature_threshold_percentile,
        min_angle_threshold=min_angle_threshold,
        edge_ratio_threshold=edge_ratio_threshold,
        normal_region_qmax=normal_region_qmax,
    )

    diagnostics = region_data["diagnostics"]
    print(
        "\n  Region segmentation: "
        f"critical={diagnostics['n_critical']} ({100.0 * diagnostics['n_critical'] / max(1, len(nodes)):.1f}%), "
        f"normal={diagnostics['n_normal']} ({100.0 * diagnostics['n_normal'] / max(1, len(nodes)):.1f}%)"
    )
    print(
        "  Patch split: "
        f"critical={len(region_data['critical_patches'])}, "
        f"normal={len(region_data['normal_patches'])}"
    )
    return region_data


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
            region_type=p.get("region_type", "critical"),
        ))
    
    sizes = [len(r.patch_nodes) for r in records]
    if sizes:
        n_critical = sum(1 for r in records if r.region_type == "critical")
        n_normal = sum(1 for r in records if r.region_type == "normal")
        print(
            f"\n  Patch summary: {len(records)} total "
            f"(critical={n_critical}, normal={n_normal}), "
            f"qubit counts min/max/avg={min(sizes)}/{max(sizes)}/{np.mean(sizes):.1f}"
        )
    return records


@task
def persist_patch_metadata_task(records, rec_dir: str):
    """
    Persist patch metadata records before optimization so both critical and
    normal patches are visible in outputs/records.
    """
    for record in records:
        record.save(rec_dir)
    return len(records)


def run_qaoa_hpc(record: PatchRecord, local_result_dir: str):
    if not record.hamiltonian_path:
        raise ValueError(f"Patch {record.patch_id} has no Hamiltonian path.")

    patch_id = record.patch_id
    ham_local = Path(record.hamiltonian_path)

    remote_ham = f"~/hpc_runs/{patch_id}.npz"
    remote_out = f"~/hpc_runs/{patch_id}.json"
    local_out = Path(local_result_dir) / f"{patch_id}_hpc.json"

    subprocess.run(["ssh", "qsim", "mkdir -p ~/hpc_runs"], check=True)

    subprocess.run(
        ["scp", str(ham_local), f"qsim:{remote_ham}"],
        check=True,
    )

    submit_cmd = f"sbatch ~/qaoa_jobs/run_qaoa.sh {remote_ham} {remote_out}"
    result = subprocess.check_output(
        ["ssh", "qsim", submit_cmd],
        text=True,
    ).strip()

    job_id = result.split()[-1]

    while True:
        status = subprocess.check_output(
            ["ssh", "qsim", f"squeue -h -j {job_id} -o %T"],
            text=True,
        ).strip()

        if not status:
            break

        time.sleep(3)

    subprocess.run(
        ["scp", f"qsim:{remote_out}", str(local_out)],
        check=True,
    )

    with open(local_out) as f:
        data = json.load(f)

    return data["bitstring"], float(data["energy"])


@task

def build_hamiltonian_task(record: PatchRecord, ham_dir: str, rec_dir: str):
    """
    Build Hamiltonian for a patch with optional boundary alignment penalty.
    
    If the patch contains boundary nodes, the boundary alignment penalty
    will be automatically enabled to preserve boundary geometry.
    """
    L = compute_L(record.patch_nodes)
    center = np.array(record.patch_nodes).mean(axis=0)
    dists = np.linalg.norm(np.array(record.patch_nodes) - center, axis=1)
    R = np.percentile(dists, 90)
    phi = phi_circle_field_local(record.patch_nodes, R=R)
    band = 0.8* R

    # Check if patch has boundary nodes
    has_boundary = (record.boundary_nodes_idx is not None and 
                   len(record.boundary_nodes_idx) > 0)
    
    # Enable boundary alignment if boundary nodes present
    boundary_nodes = record.boundary_nodes_idx if has_boundary else None

    tuning_params = { 'domain': 0, 'spacing': 0, 'sparsity': 2, 'bend': 1.8,
        'max_edge': 1.6, 'density': 0, 'angular_bins': 0,
        'collinearity': 2, 'boundary_alignment': 1.5
    }

    H, decomposition = hamiltonian_builder(
        phi=phi,
        r=record.patch_nodes,

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


def _normalize_hpc_bitstring(bitstring):
    if isinstance(bitstring, str):
        s = bitstring.strip()
        if set(s) <= {"0", "1"}:
            return s
        raise ValueError(f"Unexpected string bitstring format: {bitstring}")

    if isinstance(bitstring, np.ndarray):
        bitstring = bitstring.tolist()

    if isinstance(bitstring, (list, tuple)):
        # Your HPC format: 0 -> 0, 1023 -> 1
        return "".join("1" if int(x) != 0 else "0" for x in bitstring)

    raise TypeError(f"Unsupported HPC bitstring type: {type(bitstring)!r}")


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
    # bitstring, energy = run_qaoa_aer(
    #     record.hamiltonian_path,
    #     aer_max_parallel_threads=aer_max_parallel_threads,
    #     aer_max_parallel_experiments=aer_max_parallel_experiments,
    #     aer_max_parallel_shots=aer_max_parallel_shots,
    #     log_backend_config=log_backend_config,
    # )

    bitstring, energy = run_qaoa_hpc(record, rec_dir)

    record.bitstring = _normalize_hpc_bitstring(bitstring)
    record.energy = float(energy)

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
        "n_workers": 6,  
        "threads_per_worker": 2, 
        "processes": True,
        "memory_limit": "auto",  
        "timeout": "120s",  
        "death_timeout": "240s",  
    }
))
def mesh_hamiltonian_pipeline(
    dxf_path: str,
    L: float = 0.1,
    Q_max: int = 10,
    overlap_factor: float = 1.5,
    jitter_factor: float = 0.0,
    use_gaussian_merging: bool = True,
    hamiltonian_concurrency: int = 64,
    parallel_qaoa: bool = True,
    qaoa_concurrency: int = 8,
    qaoa_aer_max_parallel_threads: int = 2,
    qaoa_aer_max_parallel_experiments: int = 2,
    qaoa_aer_max_parallel_shots: int = 2,
    qaoa_log_backend_config: bool = False,
    adaptive_nodes: bool = True,
    L_fine: Optional[float] = None,
    L_coarse: Optional[float] = None,
    curvature_weight: float = 0.5,
    use_critical_regions: bool = False,
    critical_curvature_percentile: float = 80.0,
    critical_min_angle_threshold: float = 15.0,
    critical_edge_ratio_threshold: float = 4.0,
    normal_region_qmax: Optional[int] = None,
    smooth_iterations: int = 5,
    export_formats: tuple = ("msh", "vtk", "obj"),
):
    """
    QAOA-based mesh optimization pipeline.
    
    Full pipeline:
      DXF → adaptive nodes → critical/normal split →
      QAOA on critical patches + classical normal region →
      merge → mesh → export
    
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
        use_critical_regions: Enable critical-vs-normal region segmentation.
        critical_curvature_percentile: Curvature percentile threshold for critical nodes.
        critical_min_angle_threshold: Reserved for notebook parity.
        critical_edge_ratio_threshold: Edge-ratio threshold for critical nodes.
        normal_region_qmax: Patch node cap for normal/classical regions.
                            If None, defaults to max(100, 10*Q_max).
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
    region_data = generate_region_patches_task(
        nodes=nodes,
        L=L,
        Q_max=Q_max,
        overlap_factor=overlap_factor,
        cad_boundary_idx=cad_boundary_idx,
        use_critical_regions=use_critical_regions,
        curvature_threshold_percentile=critical_curvature_percentile,
        min_angle_threshold=critical_min_angle_threshold,
        edge_ratio_threshold=critical_edge_ratio_threshold,
        normal_region_qmax=normal_region_qmax,
    )
    records = build_patch_records(nodes, region_data["all_patches"])
    persist_patch_metadata_task(records, str(rec_dir))

    critical_records = [r for r in records if r.region_type == "critical"]
    normal_records = [r for r in records if r.region_type == "normal"]
    normal_indices = np.asarray(region_data["normal_indices"], dtype=np.intp)

    print(
        f"\n  Hybrid patch records: critical={len(critical_records)}, "
        f"normal={len(normal_records)}"
    )

    if hamiltonian_concurrency < 1:
        raise ValueError("hamiltonian_concurrency must be >= 1")

    # Build Hamiltonians only for critical patches (QAOA path).
    ham_futures = deque()
    built_records = []
    for r in critical_records:
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

    # QAOA is only applied to critical patches.
    qaoa_records = []
    if built_records:
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
                    region_type=r.region_type,
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
                    region_type=r.region_type,
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
    else:
        print("\n  No critical patches detected; skipping QAOA stage.")

    # --- Gaussian patch merging (critical patches only) + hybrid assembly ---
    mesh_info = None
    merged_critical_indices = np.empty(0, dtype=np.intp)
    merged_indices = normal_indices.copy()
    if use_gaussian_merging:
        if qaoa_records:
            merged_critical_indices = merge_patches_gaussian_task(qaoa_records, nodes, L)
            merged_critical_indices = np.asarray(merged_critical_indices, dtype=np.intp)
            print(f"\n✓ Critical QAOA merge contains {len(merged_critical_indices)} unique nodes")
        else:
            print("\n  No QAOA patch outputs to merge; using only normal-region nodes.")

        if len(merged_critical_indices) > 0:
            merged_indices = np.unique(np.concatenate([normal_indices, merged_critical_indices]))
        else:
            merged_indices = np.unique(normal_indices)

        print(
            "  Hybrid node assembly: "
            f"normal={len(normal_indices)}, critical_merged={len(merged_critical_indices)}, "
            f"total={len(merged_indices)}"
        )

        # Save selected index sets
        merged_path = base_dir / "merged_indices.npy"
        np.save(merged_path, merged_indices)
        critical_merged_path = base_dir / "critical_merged_indices.npy"
        np.save(critical_merged_path, merged_critical_indices)
        print(f"  Saved hybrid merged indices to {merged_path}")
        print(f"  Saved critical merged indices to {critical_merged_path}")

        # --- Build final mesh: Delaunay triangulate → smooth → export ---
        mesh_dir = base_dir / "mesh"
        mesh_info = build_mesh_task(
            nodes, merged_indices, dxf_path, str(mesh_dir),
            cad_boundary_idx=cad_boundary_idx,
            smooth_iterations=smooth_iterations,
            formats=export_formats,
        )

    # --- Visualization (critical/QAOA patches only) ---
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

    if all_traces:
        fig_all = combined_figure(
            all_traces,
            title="Critical patches: selected nodes",
        )
        fig_all.show()
    
    # Return results
    if use_gaussian_merging:
        return {
            "merged_indices": merged_indices,
            "critical_merged_indices": merged_critical_indices,
            "normal_indices": normal_indices,
            "mesh_info": mesh_info,
            "qaoa_records": qaoa_records,
        }
    else:
        return {"qaoa_records": qaoa_records}

if __name__ == "__main__":
    mesh_hamiltonian_pipeline("data/sample.dxf")
