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




@task
def generate_nodes_task(dxf_path):
    nodes, *_ = generate_crude_nodes(dxf_path)
    return nodes


@task
def generate_patches_task(nodes, L, Q_max, overlap_factor=1.0):
    """
    Generate patches with configurable overlap.
    
    Args:
        nodes: Node coordinates
        L: Resolution / characteristic length
        Q_max: Maximum qubits per patch
        overlap_factor: Overlap control (0.0=no overlap, 1.0=standard, >1.0=more overlap)
    """
    return generate_patch(L, nodes, Q_max, overlap_factor=overlap_factor)


@task
def build_patch_records(nodes, patches):
    records = []
    for p in patches:
        interior = nodes[p["interior_idx"]]
        if len(interior) == 0:
            continue
        records.append(PatchRecord(interior))
    return records


@task()
def build_hamiltonian_task(record: PatchRecord, ham_dir: str, rec_dir: str):

    center = np.array(record.patch_nodes).mean(axis=0),
    dists = np.linalg.norm(np.array(record.patch_nodes) - center, axis=1),
    L = np.mean(np.linalg.norm(np.array(record.patch_nodes) - np.roll(np.array(record.patch_nodes), 
                                                                      shift=1, axis=0),axis=1)),
    R = np.percentile(dists, 80)
    phi = phi_circle_field_local(record.patch_nodes, R=1.0)
    band = 0.8* R

    tuning_params = { 'domain': 1.0, 'spacing': 1.0, 'sparsity': 1.0, 'bend': 1.0,
        'max_edge': 1.0, 'density': 1.0, 'angular_bins': 1.0,
        'collinearity': 1.0, 'boundary_alignment': 1.0
    }

    H = hamiltonian_builder(
        phi=phi,
        r=record.patch_nodes,

    # geometric scale
        L=L,
    #domain constraint 
        alpha=10,band=band,
    # spacing 
        gamma=0,
    #  sparsity 
        use_sparsity=False,
        N=int(0.9 * len(phi)),   
        mu=0.25,
    #  short-range repulsion 
        use_repulsion=False,
        d_min=0.125,          
        eta=0.8,
    #  bend / angle preservation 
        use_bend=False,
        kappa=3.0,
    # max edge length
        use_max_edge=False,
        d_max=1.2*L,
        eta_max=40,
    # density regularization
        use_density_field=False,
        density_radius= 0.5*L,
        gamma_density=20,
    # angular distribution regularization
        use_angular_bins=False,
        num_angular_bins=6,
        eta_theta=20,
    # collinearity regularization
        use_collinearity_penalty=False,
        eta_col=20,
    # boundary alignment
        use_boundary_alignment=False,
        boundary_nodes=None,
        beta=20,
    # normalization and tuning
        normalize=True,
        tuning_factors=tuning_params
        )


    ham_path = os.path.join(ham_dir, f"{record.patch_id}.npz")

    np.savez(
        ham_path,
        paulis=H.paulis.to_labels(),
        coeffs=H.coeffs,
    )

    record.hamiltonian_path = ham_path
    record.save(rec_dir)
    return record

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
    use_gaussian_merging: bool = True,
):
    """
    QAOA-based mesh optimization pipeline with optional Gaussian patch merging.
    
    Args:
        dxf_path: Path to DXF file with mesh nodes
        L: Characteristic length scale for patch generation
        Q_max: Maximum qubits per patch
        overlap_factor: Controls overlap between patches (0.0=no overlap, 1.0=standard, >1.0=more)
        use_gaussian_merging: Enable Gaussian-weighted merging of overlapping patches
    """

    ctx = get_run_context()
    run_id = str(ctx.flow_run.id)
    
    base_dir = Path("outputs") / run_id

    ham_dir = base_dir / "hamiltonians"
    rec_dir = base_dir / "records"

    ham_dir.mkdir(parents=True, exist_ok=True)
    rec_dir.mkdir(parents=True, exist_ok=True)

    # --- pipeline ---
    nodes = generate_nodes_task(dxf_path)
    patches = generate_patches_task(nodes, L, Q_max, overlap_factor=overlap_factor)
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

    qaoa_records = []
    qaoa_futures = []
    for r in built_records:
        qaoa_futures.append(
            run_qaoa_task.submit(r, str(rec_dir))
        )
    qaoa_records = [f.result() for f in qaoa_futures]

    # --- Gaussian patch merging (optional) ---
    if use_gaussian_merging:
        merged_indices = merge_patches_gaussian_task(qaoa_records, nodes, L)
        print(f"\n✓ Merged mesh contains {len(merged_indices)} unique nodes")
        
        # Save merged indices
        merged_path = base_dir / "merged_indices.npy"
        np.save(merged_path, merged_indices)
        print(f" Saved merged indices to {merged_path}")

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
