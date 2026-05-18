import json
import os
import sys

from qaoa_qulacs_mpi_observable import run_qaoa_qulacs_mpi_observable


def _env_float(name, default):
    raw = os.environ.get(name)
    return float(raw) if raw not in (None, "") else float(default)


def _env_int(name, default):
    raw = os.environ.get(name)
    return int(raw) if raw not in (None, "") else int(default)


def main():
    ham_path = sys.argv[1]
    out_path = sys.argv[2]

    reps = _env_int("QAOA_REPS", 1)
    maxiter = _env_int("QAOA_MAXITER", 20)
    shots = _env_int("QAOA_SHOTS", 2048)
    learning_rate = _env_float("QAOA_SPSA_LEARNING_RATE", 0.4)
    perturbation = _env_float("QAOA_SPSA_PERTURBATION", 0.2)
    alpha = _env_float("QAOA_SPSA_ALPHA", 0.602)
    gamma = _env_float("QAOA_SPSA_GAMMA", 0.101)
    stability_raw = os.environ.get("QAOA_SPSA_STABILITY")
    stability_constant = None if stability_raw in (None, "") else float(stability_raw)
    random_seed = _env_int("QAOA_RANDOM_SEED", 42)

    bitstring, energy, metadata = run_qaoa_qulacs_mpi_observable(
        ham_path,
        p=reps,
        maxiter=maxiter,
        shots=shots,
        learning_rate=learning_rate,
        perturbation=perturbation,
        alpha=alpha,
        gamma=gamma,
        stability_constant=stability_constant,
        random_seed=random_seed,
    )

    if bitstring is None:
        return

    payload = {
        "bitstring": bitstring,
        "energy": float(energy),
    }
    if metadata:
        payload.update(metadata)

    with open(out_path, "w") as f:
        json.dump(payload, f)


if __name__ == "__main__":
    main()
