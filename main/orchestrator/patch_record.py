# orchestrator/patch_record.py

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import hashlib
import json
import os

@dataclass
class PatchRecord:
    patch_nodes: np.ndarray

    patch_id: str = field(init=False)
    hamiltonian_path: Optional[str] = None

    def __post_init__(self):
        arr = np.asarray(self.patch_nodes).round(6)
        self.patch_id = hashlib.sha256(arr.tobytes()).hexdigest()[:12]

    def save(self, directory):
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, f"{self.patch_id}.json")
        with open(path, "w") as f:
            json.dump(
                {
                    "patch_id": self.patch_id,  
                    "hamiltonian_path": self.hamiltonian_path,
                },
                f,
                indent=2
            )

