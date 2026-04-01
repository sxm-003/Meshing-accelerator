from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import hashlib
import json
import os

@dataclass
class PatchRecord:
    patch_nodes: np.ndarray
    phi: Optional[np.ndarray] = None
    boundary_nodes_idx: Optional[np.ndarray] = None  # Local indices of boundary nodes
    global_indices: Optional[np.ndarray] = None       # Global indices into the full nodes array
    region_type: str = "critical"  # "critical" (QAOA) or "normal" (classical)

    patch_id: str = field(init=False)
    hamiltonian_path: Optional[str] = None
    bitstring: Optional[str] = None
    energy: Optional[float] = None

    def __post_init__(self):
        arr = np.asarray(self.patch_nodes).round(6)
        self.patch_id = hashlib.sha256(arr.tobytes()).hexdigest()[:12]

    def save(self, directory):
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, f"{self.patch_id}.json")

        json.dump(
            {
                "patch_id": self.patch_id,
                "patch_nodes": self.patch_nodes.tolist(),
                "phi": None if self.phi is None else self.phi.tolist(),
                "boundary_nodes_idx": None if self.boundary_nodes_idx is None else self.boundary_nodes_idx.tolist(),
                "global_indices": None if self.global_indices is None else self.global_indices.tolist(),
                "region_type": self.region_type,
                "hamiltonian_path": self.hamiltonian_path,
                "bitstring": self.bitstring,
                "energy": self.energy,
            },
            open(path, "w"),
            indent=2,
        )
