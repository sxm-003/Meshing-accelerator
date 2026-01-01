# orchestrator/flow.py
from prefect import flow, task
import asyncio
import os
import numpy as np

from node_manager.crude_generator import generate_crude_nodes
from node_manager.patch_generator import generate_patch

from quantum_processing.hamiltonian_builder import hamiltonian_builder, phi_circle_field
from orchestrator.patch_record import PatchRecord
from orchestrator.resource_config import estimate_hamiltonian_concurrency
from orchestrator.prefect_utils import ensure_concurrency_limit


@task
def generate_nodes_task(dxf_path): #check for the the other kinds of nodes as well 
    nodes, *_ = generate_crude_nodes(dxf_path)
    return nodes

@task
def generate_patches_task(nodes, L, Q_max):
    patches = generate_patch(L, nodes, Q_max)
    return patches

@task
def build_patch_records(nodes, patches):
    records = []

    for p in patches:
        interior_nodes = nodes[p["interior_idx"]] #check as well for other nodes
        if len(interior_nodes) == 0:
            continue

        record = PatchRecord(interior_nodes)
        records.append(record)

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
    concurrency = estimate_hamiltonian_concurrency()

    asyncio.run(
        ensure_concurrency_limit(
            name="hamiltonian-medium",
            limit=concurrency,
        )
    )

    nodes = generate_nodes_task(dxf_path)

    patches = generate_patches_task(nodes, L, Q_max)

    records = build_patch_records(nodes, patches)

    futures = []
    for record in records:
        futures.append(build_hamiltonian_task.submit(record))

    return futures

