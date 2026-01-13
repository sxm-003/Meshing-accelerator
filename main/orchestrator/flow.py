# orchestrator/flow.py
from prefect import flow, task
import os
import numpy as np

from node_manager.crude_generator import generate_crude_nodes
from node_manager.patch_generator import generate_patch

from quantum_processing.hamiltonian_builder import (
    hamiltonian_builder,
    phi_circle_field,
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
def generate_patches_task(nodes, L, Q_max):
    return generate_patch(L, nodes, Q_max)


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
    phi = phi_circle_field(record.patch_nodes, R=1.0)
    record.phi = phi

    H = hamiltonian_builder(
        phi=phi,
        r=record.patch_nodes,
    # --- geometric scale ---
        L=0.5,

    # --- domain constraint ---
        alpha=10,

    # --- spacing ---
        gamma=0,

    # --- sparsity (DO NOT be too aggressive) ---
        use_sparsity=False,
        N=int(0.65 * len(phi)),   # keep ~65% nodes
        mu=0.25,

    # --- short-range repulsion ---
        use_repulsion=False,
        d_min=0.125,           # = 0.125
        eta=0.8,

    # --- bend / angle preservation ---
        use_bend=False,
        kappa=3.0
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
):


    ctx = get_run_context()
    run_id = str(ctx.flow_run.id)
    
    base_dir = Path("outputs") / run_id

    ham_dir = base_dir / "hamiltonians"
    rec_dir = base_dir / "records"

    ham_dir.mkdir(parents=True, exist_ok=True)
    rec_dir.mkdir(parents=True, exist_ok=True)

    # --- pipeline ---
    nodes = generate_nodes_task(dxf_path)
    patches = generate_patches_task(nodes, L, Q_max)
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
        title="All patches – selected nodes",
)
    fig_all.show()

if __name__ == "__main__":
    mesh_hamiltonian_pipeline("data/sample.dxf")
