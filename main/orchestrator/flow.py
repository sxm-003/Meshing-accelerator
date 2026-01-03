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


@task(tags=["hamiltonian-medium"])
def build_hamiltonian_task(record: PatchRecord, ham_dir: str, rec_dir: str):
    phi = phi_circle_field(record.patch_nodes, R=1.0)
    record.phi = phi

    H = hamiltonian_builder(
        phi=phi,
        r=record.patch_nodes,
        neighbors=[],
        L=1.0,
        alpha=1.0,
        gamma=1.0,
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



@flow(task_runner=DaskTaskRunner( 
    cluster_kwargs={"n_workers": 40, "threads_per_worker": 1}
    ))
def mesh_hamiltonian_pipeline(
    dxf_path: str,
    L: float = 0.4,
    Q_max: int = 30,
):
    ctx = get_run_context()
    run_id = str(ctx.flow_run.id)
    
    base_dir = Path("outputs") / run_id

    ham_dir = base_dir / "hamiltonians"
    rec_dir = base_dir / "records"

    ham_dir.mkdir(parents=True, exist_ok=True)
    rec_dir.mkdir(parents=True, exist_ok=True)
    nodes = generate_nodes_task(dxf_path)
    patches = generate_patches_task(nodes, L, Q_max)
    records = build_patch_records(nodes, patches)

    futures = []
    for r in records:
        futures.append(
        build_hamiltonian_task.submit(
            r,
            str(ham_dir),
            str(rec_dir),
        )
    )

    return futures


if __name__ == "__main__":
    mesh_hamiltonian_pipeline("data/sample.dxf")
