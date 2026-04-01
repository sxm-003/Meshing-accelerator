import sys
import json
from qaoa_qulacs_mpi import run_qaoa_qulacs_mpi

ham_path = sys.argv[1]
out_path = sys.argv[2]

bitstring, energy = run_qaoa_qulacs_mpi(ham_path)

if bitstring is not None:
    with open(out_path, "w") as f:
        json.dump({
            "bitstring": bitstring,
            "energy": energy
        }, f)