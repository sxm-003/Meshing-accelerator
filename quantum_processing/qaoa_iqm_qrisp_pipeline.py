import json
import logging
import os
import time
import warnings
from pathlib import Path

import numpy as np
from qiskit.quantum_info import SparsePauliOp


def _resolve_token_json_path(token_json_path=None):
    if token_json_path is not None:
        candidate = Path(token_json_path).expanduser()
        if candidate.is_absolute():
            return candidate.resolve()

        repo_candidate = Path(__file__).resolve().parents[2] / candidate
        if repo_candidate.exists():
            return repo_candidate.resolve()

        cwd_candidate = (Path.cwd() / candidate).resolve()
        if cwd_candidate.exists():
            return cwd_candidate

        return repo_candidate.resolve()

    repo_token = Path(__file__).resolve().parents[2] / "token.json"
    if repo_token.exists():
        return repo_token

    return (Path.cwd() / "token.json").resolve()


def _load_iqm_token(token_json_path=None):
    path = _resolve_token_json_path(token_json_path)
    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    return (
        payload.get("iqm_token")
        or payload.get("token")
        or payload.get("apikey")
        or payload.get("api_key")
        or payload.get("access_token")
    )


def _load_hamiltonian(hamiltonian_input):
    if isinstance(hamiltonian_input, SparsePauliOp):
        return hamiltonian_input

    if isinstance(hamiltonian_input, (str, os.PathLike)):
        data = np.load(hamiltonian_input, allow_pickle=False)
        if "SparsePauliOp" in data and "coeffs" in data:
            ops = data["SparsePauliOp"]
            positions = data["positions"]
            coeffs = data["coeffs"]
            num_qubits = int(data["num_qubits"][0])

            sparse_list = []
            for op, pos_row, coeff in zip(ops, positions, coeffs):
                op = str(op)
                if op == "Z":
                    pos = [int(pos_row[0])]
                elif op == "ZZ":
                    pos = [int(pos_row[0]), int(pos_row[1])]
                elif op == "I":
                    pos = []
                else:
                    raise ValueError(f"Unsupported sparse op in Hamiltonian file: {op}")
                sparse_list.append((op, pos, complex(coeff)))
            return SparsePauliOp.from_sparse_list(sparse_list, num_qubits=num_qubits)

        paulis = data["paulis"]
        coeffs = data["coeffs"]
        return SparsePauliOp(paulis, coeffs)

    if isinstance(hamiltonian_input, dict):
        if "SparsePauliOp" in hamiltonian_input and "coeffs" in hamiltonian_input:
            ops = hamiltonian_input["SparsePauliOp"]
            positions = hamiltonian_input["positions"]
            coeffs = hamiltonian_input["coeffs"]
            num_qubits = int(hamiltonian_input["num_qubits"][0])

            sparse_list = []
            for op, pos_row, coeff in zip(ops, positions, coeffs):
                op = str(op)
                if op == "Z":
                    pos = [int(pos_row[0])]
                elif op == "ZZ":
                    pos = [int(pos_row[0]), int(pos_row[1])]
                elif op == "I":
                    pos = []
                else:
                    raise ValueError(
                        f"Unsupported sparse op in Hamiltonian payload: {op}"
                    )
                sparse_list.append((op, pos, complex(coeff)))
            return SparsePauliOp.from_sparse_list(sparse_list, num_qubits=num_qubits)

        paulis = hamiltonian_input["paulis"]
        coeffs = hamiltonian_input["coeffs"]
        return SparsePauliOp(paulis, coeffs)

    raise TypeError(
        "hamiltonian_path must be SparsePauliOp, a .npz path, or a dict payload."
    )


def _bitstring_energy(bitstring, sparse_terms):
    z = 1 - 2 * np.asarray(bitstring, dtype=float)
    energy = 0.0
    for op, positions, coeff in sparse_terms:
        c = float(np.real(coeff))
        if op == "Z":
            energy += c * z[int(positions[0])]
        elif op == "ZZ":
            i, j = int(positions[0]), int(positions[1])
            energy += c * z[i] * z[j]
        elif op == "I":
            energy += c
        else:
            raise ValueError(f"Unsupported Pauli term in sparse terms: {op}")
    return float(energy)


def _sparse_terms_to_qubo(sparse_terms, num_qubits):
    h = np.zeros(num_qubits, dtype=float)
    pairs = {}
    constant = 0.0

    for op, positions, coeff in sparse_terms:
        real = float(np.real(coeff))
        imag = float(np.imag(coeff))
        if abs(imag) > 1e-9:
            raise ValueError(
                "Hamiltonian has complex coefficients; Qrisp QUBO conversion requires "
                "real-valued Ising coefficients."
            )

        if op == "I":
            constant += real
        elif op == "Z":
            i = int(positions[0])
            h[i] += real
        elif op == "ZZ":
            i, j = int(positions[0]), int(positions[1])
            if i == j:
                constant += real
                continue
            a, b = (i, j) if i < j else (j, i)
            pairs[(a, b)] = pairs.get((a, b), 0.0) + real
        else:
            raise ValueError(
                f"Unsupported term '{op}' for Qrisp QUBO conversion. "
                "Expected only I/Z/ZZ terms."
            )

    linear = -2.0 * h
    for (i, j), val in pairs.items():
        linear[i] += -2.0 * val
        linear[j] += -2.0 * val

    qubo = np.zeros((num_qubits, num_qubits), dtype=float)
    for i in range(num_qubits):
        qubo[i, i] = linear[i]
    for (i, j), val in pairs.items():
        qubo[i, j] += 4.0 * val

    offset = float(constant + np.sum(h) + np.sum(list(pairs.values())))
    return qubo, offset


def _qrisp_key_to_bitstring(key, num_qubits):
    values = None
    if hasattr(key, "tolist"):
        try:
            values = list(key.tolist())
        except Exception:
            values = None

    if values is None and isinstance(key, (tuple, list, np.ndarray)):
        values = list(key)

    if values is not None:
        bits = []
        for value in values:
            if isinstance(value, str):
                bits.extend([int(ch) for ch in value if ch in "01"])
            else:
                bits.append(int(value))
    else:
        key_text = str(key)
        bits = [int(ch) for ch in key_text if ch in "01"]

    if len(bits) < num_qubits:
        bits = [0] * (num_qubits - len(bits)) + bits
    elif len(bits) > num_qubits:
        bits = bits[-num_qubits:]

    return "".join(str(int(bit)) for bit in bits)


def _extract_top_k(distribution, top_k=10):
    if top_k is None or top_k < 1:
        top_k = 10

    ranked = sorted(
        distribution.items(),
        key=lambda kv: (-float(kv[1]), kv[0]),
    )[: int(top_k)]

    return [
        {
            "bitstring": bitstring,
            "probability": float(prob),
            "state_int": int(bitstring, 2),
        }
        for bitstring, prob in ranked
    ]


def run_qaoa_iqm_qrisp_batch(
    hamiltonian_path,
    iqm_server_url=None,
    quantum_computer=None,
    iqm_token=None,
    token_json_path=None,
    reps=1,
    init_params=None,
    shots=1000,
    n_iterations=12,
    candidates_per_iteration=8,  # unused, kept for API compatibility
    search_scale=np.pi / 3,  # unused, kept for API compatibility
    search_decay=0.75,  # unused, kept for API compatibility
    optimization_level=3,  # unused, kept for API compatibility
    layout_method="sabre",  # unused, kept for API compatibility
    allow_direct_client_fallback=True,  # unused, kept for API compatibility
    result_timeout_secs=1800,  # unused, kept for API compatibility
    return_topk=False,
    top_k=10,
    random_seed=42,  # unused, kept for API compatibility
    qrisp_init_type="random",
    qrisp_optimizer="COBYLA",
    qrisp_optimizer_options=None,
    verbose=True,
):
    """
    Run QAOA with Qrisp on IQM hardware.

    Inputs are intentionally kept close to run_qaoa_iqm_batch so notebooks can
    switch with minimal edits.
    """
    del (
        candidates_per_iteration,
        search_scale,
        search_decay,
        optimization_level,
        layout_method,
        allow_direct_client_fallback,
        result_timeout_secs,
        random_seed,
    )

    t0 = time.time()

    # Qrisp imports JAX, which emits an irrelevant GPU fallback warning on systems
    # without a CUDA-enabled jaxlib. Force CPU to keep notebook output readable.
    os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    logging.getLogger("jax._src.xla_bridge").setLevel(logging.ERROR)

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=SyntaxWarning,
                module=r"qrisp(\..*)?",
            )
            from qrisp import QuantumArray, QuantumVariable
            from qrisp.interface import IQMBackend
            from qrisp.qaoa import QUBO_problem
    except ImportError as exc:
        raise ImportError(
            "Qrisp IQM pipeline requires qrisp with IQM support. Install with:\n"
            "  pip uninstall -y qiskit-iqm cirq-iqm\n"
            "  pip install --upgrade \"qrisp[iqm]\" \"iqm-client[qiskit,cirq]\"\n"
            "Use Python 3.10-3.12 (iqm-client >=33 does not support Python 3.13+)."
        ) from exc

    hamiltonian = _load_hamiltonian(hamiltonian_path)
    sparse_terms = hamiltonian.to_sparse_list()
    num_qubits = int(hamiltonian.num_qubits)
    qubo_matrix, _qubo_offset = _sparse_terms_to_qubo(sparse_terms, num_qubits)
    effective_shots = max(1, int(shots))
    effective_depth = max(1, int(reps))
    effective_max_iter = max(1, int(n_iterations))
    optimizer_options = (
        {} if qrisp_optimizer_options is None else dict(qrisp_optimizer_options)
    )

    token = iqm_token or _load_iqm_token(token_json_path) or os.environ.get("IQM_TOKEN")
    if not token:
        resolved = _resolve_token_json_path(token_json_path)
        raise ValueError(
            "No IQM token found for Qrisp execution. "
            f"Checked token_json_path={resolved} and IQM_TOKEN env var."
        )

    # Qrisp commonly targets Resonance by device_instance. If both are provided,
    # prefer device_instance to avoid endpoint path/version issues.
    backend_kwargs = {"api_token": str(token).strip()}
    effective_qpu = quantum_computer or os.environ.get("IQM_QUANTUM_COMPUTER") or "garnet"
    if effective_qpu:
        backend_kwargs["device_instance"] = str(effective_qpu)
    elif iqm_server_url:
        backend_kwargs["server_url"] = str(iqm_server_url).rstrip("/")
    else:
        raise ValueError(
            "Provide quantum_computer (recommended) or iqm_server_url for IQMBackend."
        )

    approx_backend_calls = effective_max_iter + 1
    if str(qrisp_init_type).lower() == "tqa":
        approx_backend_calls += 10

    if verbose:
        print(
            "  Qrisp IQM: building backend and QAOA problem "
            f"[qpu={effective_qpu}, qubits={num_qubits}, shots={effective_shots}, "
            f"depth={effective_depth}, init={qrisp_init_type}, "
            f"optimizer={qrisp_optimizer}, max_iter={effective_max_iter}, "
            f"~backend_calls={approx_backend_calls}]"
        )

    backend = IQMBackend(**backend_kwargs)
    qarg = QuantumArray(qtype=QuantumVariable(1), shape=num_qubits)
    problem = QUBO_problem(qubo_matrix)

    if init_params is not None:
        print("  Qrisp IQM note: init_params is not used by QUBO_problem.run().")

    if verbose:
        print("  Qrisp IQM: starting optimizer and waiting for IQM job completions...")

    raw_distribution = problem.run(
        qarg,
        depth=effective_depth,
        mes_kwargs={"backend": backend, "shots": effective_shots},
        max_iter=effective_max_iter,
        init_type=qrisp_init_type,
        optimizer=qrisp_optimizer,
        options=optimizer_options,
    )
    if not isinstance(raw_distribution, dict):
        raise RuntimeError(
            "Unexpected Qrisp QAOA output type. Expected dict-like distribution, "
            f"got {type(raw_distribution).__name__}."
        )

    distribution = {}
    for raw_key, raw_value in raw_distribution.items():
        bitstring = _qrisp_key_to_bitstring(raw_key, num_qubits)
        probability = float(raw_value)
        distribution[bitstring] = distribution.get(bitstring, 0.0) + probability

    total_prob = float(sum(distribution.values()))
    if total_prob <= 0:
        raise RuntimeError("Qrisp IQM run returned empty/zero measurement distribution.")
    distribution = {k: float(v) / total_prob for k, v in distribution.items()}

    best_bitstring = None
    best_energy = np.inf
    for bitstring, _prob in distribution.items():
        bits = [int(ch) for ch in bitstring]
        energy = _bitstring_energy(bits, sparse_terms)
        if energy < best_energy:
            best_energy = float(energy)
            best_bitstring = bits

    if best_bitstring is None:
        raise RuntimeError("Could not determine best bitstring from Qrisp IQM output.")

    top_bitstrings = _extract_top_k(distribution, top_k=top_k) if return_topk else None

    elapsed = time.time() - t0
    print(
        f"  Qrisp IQM QAOA [{num_qubits}q, {reps}p]: "
        f"sample_min={best_energy:.6f}, selected={sum(best_bitstring)}/{num_qubits}, "
        f"qpu={effective_qpu}, time={elapsed:.1f}s"
    )

    if return_topk:
        return best_bitstring, float(best_energy), top_bitstrings
    return best_bitstring, float(best_energy)


def run_qaoa_iqm_batch(*args, **kwargs):
    """Compatibility alias matching existing notebook/task call sites."""
    return run_qaoa_iqm_qrisp_batch(*args, **kwargs)
