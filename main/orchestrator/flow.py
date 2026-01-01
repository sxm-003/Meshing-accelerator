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
def build_hamiltonian_task(record: PatchRecord):
    phi = phi_circle_field(record.patch_nodes, R=1.0)

    H = hamiltonian_builder(
        phi=phi,
        r=record.patch_nodes,
        neighbors=[],
        L=1.0,
        alpha=1.0,
        gamma=1.0,
    )

    os.makedirs("outputs/hamiltonians", exist_ok=True)
    path = f"outputs/hamiltonians/{record.patch_id}.npz"

    np.savez(
        path,
        paulis=H.paulis.to_labels(),
        coeffs=H.coeffs,
    )

    record.hamiltonian_path = path
    record.save()
    return record



@flow
def mesh_hamiltonian_pipeline(
    dxf_path: str,
    L: float = 0.4,
    Q_max: int = 30,
):
    nodes = generate_nodes_task(dxf_path)
    patches = generate_patches_task(nodes, L, Q_max)
    records = build_patch_records(nodes, patches)

    futures = []
    for r in records:
        futures.append(build_hamiltonian_task.submit(r))

    return futures


if __name__ == "__main__":
    mesh_hamiltonian_pipeline("data/sample.dxf")
