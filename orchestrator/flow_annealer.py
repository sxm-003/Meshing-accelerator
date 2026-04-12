from prefect import flow, task
from prefect.context import get_run_context

from pathlib import Path
from typing import Optional
from collections import deque
import numpy as np

import orchestrator.flow as orc
from quantum_processing.qubo_bqm_conversion import sparse_pauli_to_qubo
from helper_functions.npz_to_pauli import load_sparse_pauli_from_npz

from dwave.system import DWaveSampler, EmbeddingComposite

def get_sampler():
    return EmbeddingComposite(DWaveSampler())


@task
def anneal_patch_task(record: orc.PatchRecord, rec_dir: str, num_reads: int = 1000):
    H = load_sparse_pauli_from_npz(record.hamiltonian_path)
    Q, offset = sparse_pauli_to_qubo(H)
    sampler = get_sampler()
    sampleset = sampler.sample_qubo(Q, num_reads=num_reads)
    best = sampleset.first

    sample = best.sample
    record.bitstring = ''.join(str(sample[i]) for i in sorted(sample))
    record.energy = float(best.energy)
    record.save(rec_dir)

    return record


@flow
def annealer_meshing(
    dxf_path: str,
    L_patch: float = 2,
    L_nodes: float = 0.5,
    Q_max: int = 20,
    overlap_factor: float = 1.5,
    jitter_factor: float = 0.0,
    use_gaussian_merging: bool = True,
    hamiltonian_concurrency: int = 64,
    adaptive_nodes: bool = False,
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
    ctx = get_run_context()
    run_id = str(ctx.flow_run.id)
    
    base_dir = Path("outputs") / run_id
    ham_dir = base_dir / "hamiltonians"
    rec_dir = base_dir / "records"
    ham_dir.mkdir(parents=True, exist_ok=True)
    rec_dir.mkdir(parents=True, exist_ok=True)

    nodes, cad_boundary_idx = orc.generate_nodes_task(
        dxf_path,
        jitter_factor=jitter_factor,
        adaptive=adaptive_nodes,
        L_nodes=L_nodes,
        L_fine=L_fine,
        L_coarse=L_coarse,
        curvature_weight=curvature_weight,
    )
    region_data = orc.generate_region_patches_task(
        nodes=nodes,
        L_patch=L_patch,
        Q_max=Q_max,
        overlap_factor=overlap_factor,
        cad_boundary_idx=cad_boundary_idx,
        use_critical_regions=use_critical_regions,
        curvature_threshold_percentile=critical_curvature_percentile,
        min_angle_threshold=critical_min_angle_threshold,
        edge_ratio_threshold=critical_edge_ratio_threshold,
        normal_region_qmax=normal_region_qmax,
    )

    records = orc.build_patch_records(nodes, region_data["all_patches"])
    orc.persist_patch_metadata_task(records, str(rec_dir))

    critical_records = [r for r in records if r.region_type == "critical"]
    normal_records = [r for r in records if r.region_type == "normal"]
    normal_indices = np.asarray(region_data["normal_indices"], dtype=np.intp)

    print(
        f"\n  Hybrid patch records: critical={len(critical_records)}, "
        f"normal={len(normal_records)}")
    
    print("\n  Building Hamiltonians...")
    if hamiltonian_concurrency < 1:
        raise ValueError("hamiltonian_concurrency must be at least 1.")
    
    ham_builds = deque()
    ham_records = []
    for rec in critical_records:
        ham_builds.append(orc.build_hamiltonian_task.submit(
            rec,
            str(ham_dir),
            str(rec_dir),))
                
        if len(ham_builds) >= hamiltonian_concurrency:
            ham_records.append(ham_builds.popleft().result())

    while ham_builds:
        ham_records.append(ham_builds.popleft().result())

    annealed_records = []
    for record in ham_records:
        annealed_records.append(anneal_patch_task.submit(
            record,
            str(rec_dir),
            num_reads=1000,
        ).result())

    if use_gaussian_merging:
        merged_critical_indices = orc.merge_patches_gaussian_task(
            annealed_records,
            nodes,
            L_patch,
    )
        merged_critical_indices = np.asarray(merged_critical_indices, dtype=np.intp)
        merged_indices = np.unique(
        np.concatenate([normal_indices, merged_critical_indices])
    )
    else:
        merged_indices = np.unique(normal_indices)

    print(f"  Total merged patch nodes: {len(merged_indices)}")
    print(f"  Total original nodes: {len(nodes)}")
    print("\n Building mesh...")
    
    mesh_dir = base_dir / "mesh"
    mesh_info = orc.build_mesh_task(
    nodes,
    merged_indices,
    dxf_path,
    str(mesh_dir),
    cad_boundary_idx=cad_boundary_idx,
    smooth_iterations=5,
    formats=export_formats,
    )

    all_traces = []
    for rec in annealed_records:
        traces = orc.patch_traces(
        patch_nodes = rec.patch_nodes,
        phi = rec.phi,
        bitstring= rec.bitstring,
        patch_id=rec.patch_id,
        show_phi=True,
        )
        all_traces.extend(traces)

    if all_traces:
        fig_all = orc.combined_figure(
            all_traces,
            title="Critical patches: selected nodes",
        )
        fig_all.show() 

    if use_gaussian_merging:
        return {
            "merged_indices": merged_indices,
            "critical_merged_indices": merged_critical_indices,
            "normal_indices": normal_indices,
            "mesh_info": mesh_info,
            "qaoa_records": annealed_records,
        }
    else:
        return {"qaoa_records": annealed_records}
    

if __name__ == "__main__":
    annealer_meshing(dxf_path="data/test.dxf")

