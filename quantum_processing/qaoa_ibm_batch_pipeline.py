import json
import os
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from qiskit.primitives import StatevectorEstimator
from qiskit_aer import AerSimulator
from qiskit.circuit.library import QAOAAnsatz
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager


DEFAULT_IBM_INSTANCE_CRN = (
    "crn"
)


def _resolve_mesh_json_path(mesh_config_path=None):
    if mesh_config_path is not None:
        return Path(mesh_config_path).expanduser().resolve()

    repo_mesh = Path(__file__).resolve().parents[2] / "mesh.json"
    if repo_mesh.exists():
        return repo_mesh

    cwd_mesh = Path.cwd() / "mesh.json"
    return cwd_mesh.resolve()


def _load_ibm_api_key(mesh_config_path=None):
    path = _resolve_mesh_json_path(mesh_config_path)
    if not path.exists():
        raise FileNotFoundError(f"mesh.json not found at: {path}")

    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    api_key = payload.get("apikey") or payload.get("api_key") or payload.get("token")
    if not api_key:
        raise KeyError(
            f"No API key field found in {path}. Expected one of: apikey, api_key, token."
        )
    return api_key


def _extract_scalar_expectation(raw_evs):
    arr = np.asarray(raw_evs)
    if arr.size == 0:
        raise ValueError("Estimator returned empty expectation values.")
    return float(np.real(arr.reshape(-1)[0]))


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
                    raise ValueError(f"Unsupported sparse op in Hamiltonian payload: {op}")
                sparse_list.append((op, pos, complex(coeff)))

            return SparsePauliOp.from_sparse_list(sparse_list, num_qubits=num_qubits)

        paulis = hamiltonian_input["paulis"]
        coeffs = hamiltonian_input["coeffs"]
        return SparsePauliOp(paulis, coeffs)

    raise TypeError(
        "hamiltonian_path must be SparsePauliOp, a .npz path, or a dict payload."
    )


def _ansatz_builder(hamiltonian, reps=4):
    return QAOAAnsatz(hamiltonian, reps=reps)


def _transpile_circuit(circuit, backend, optimization_level=1):
    pm = generate_preset_pass_manager(
        optimization_level=int(optimization_level), backend=backend
    )
    return pm.run(circuit)


def _cost_func_estimator(params, ansatz, hamiltonian, estimator_instance):
    isa_hamiltonian = hamiltonian.apply_layout(ansatz.layout)
    pub = (ansatz, isa_hamiltonian, params)
    job = estimator_instance.run([pub])
    result = job.result()[0]
    return _extract_scalar_expectation(result.data.evs)


def _optimize_parameters(
    init_params,
    candidate_circuit,
    hamiltonian,
    estimator_instance,
    method="SLSQP",
    maxiter=500,
    ftol=1e-6,
):
    eval_count = [0]

    def _cost(params):
        eval_count[0] += 1
        return _cost_func_estimator(
            params, candidate_circuit, hamiltonian, estimator_instance
        )

    res = minimize(
        _cost,
        init_params,
        method=method,
        options={"maxiter": maxiter, "ftol": ftol},
    )
    return res.x, res.fun, eval_count[0]


def _sample_circuit(candidate_circuit, optimal_params, sampler_instance, shots=4096):
    measured = candidate_circuit.copy()
    measured.measure_all()
    bound = measured.assign_parameters(optimal_params)

    run_kwargs = {}
    if shots is not None:
        run_kwargs["shots"] = int(shots)

    try:
        job = sampler_instance.run([bound], **run_kwargs)
    except TypeError:
        job = sampler_instance.run([bound])

    counts_int = job.result()[0].data.meas.get_int_counts()
    total = sum(counts_int.values())
    return {key: val / total for key, val in counts_int.items()}


def _to_bitstring(integer, num_bits):
    result = np.binary_repr(integer, width=num_bits)
    return [int(digit) for digit in result]


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
    return float(energy)


def _extract_lowest_energy_bitstring(distribution, num_qubits, sparse_terms):
    best_bitstring = None
    best_energy = np.inf

    for key in distribution:
        bitstring = _to_bitstring(int(key), num_qubits)
        bitstring.reverse()
        energy = _bitstring_energy(bitstring, sparse_terms)
        if energy < best_energy:
            best_bitstring = bitstring
            best_energy = energy

    return best_bitstring, float(best_energy)


def _extract_top_k_bitstrings(distribution, num_qubits, top_k=10):
    if top_k is None or top_k < 1:
        top_k = 10

    ranked = sorted(
        distribution.items(),
        key=lambda kv: (-float(kv[1]), int(kv[0])),
    )[: int(top_k)]

    out = []
    for key, prob in ranked:
        bitstring = _to_bitstring(int(key), num_qubits)
        bitstring.reverse()
        out.append(
            {
                "bitstring": "".join(str(b) for b in bitstring),
                "probability": float(prob),
                "state_int": int(key),
            }
        )
    return out


def run_qaoa_ibm_batch(
    hamiltonian_path,
    reps=4,
    init_params=None,
    optimization_method="SLSQP",
    maxiter=500,
    ftol=1e-3,
    backend_name="ibm_torino",
    shots=4096,
    mesh_config_path=None,
    instance_crn=DEFAULT_IBM_INSTANCE_CRN,
    channel="ibm_cloud",
    optimize_on_hardware=True,
    batch_max_time="8h",
    log_backend_config=False,
    return_topk=False,
    top_k=10,
):
    """
    Run QAOA on an IBM Quantum backend using Runtime Batch mode.

    API key is read from mesh.json (defaults to repository root).
    By default, parameter optimization and sampling both run on IBM hardware.
    Set optimize_on_hardware=False to optimize locally and only sample on hardware.
    """
    try:
        from qiskit_ibm_runtime import (
            Batch,
            EstimatorV2 as RuntimeEstimatorV2,
            QiskitRuntimeService,
            SamplerV2 as RuntimeSamplerV2,
        )
    except ImportError as exc:
        raise ImportError(
            "qiskit-ibm-runtime is required for IBM backend execution. "
            "Install with: pip install qiskit-ibm-runtime"
        ) from exc

    t0 = time.time()

    api_key = _load_ibm_api_key(mesh_config_path=mesh_config_path)
    channel_candidates = [channel]
    for fallback in ("ibm_cloud", "ibm_quantum_platform"):
        if fallback not in channel_candidates:
            channel_candidates.append(fallback)

    service = None
    last_error = None
    active_channel = None
    for channel_name in channel_candidates:
        try:
            service = QiskitRuntimeService(
                channel=channel_name,
                token=api_key,
                instance=instance_crn,
            )
            active_channel = channel_name
            break
        except Exception as exc:  # pragma: no cover - remote auth path
            last_error = exc

    if service is None:
        raise RuntimeError(
            "Failed to initialize QiskitRuntimeService with provided channel/CRN."
        ) from last_error

    backend = service.backend(backend_name)

    if log_backend_config:
        print(
            "  IBM Runtime config: "
            f"channel={active_channel}, backend={backend_name}, shots={shots}, "
            f"instance={instance_crn}, optimize_on_hardware={optimize_on_hardware}, "
            f"batch_max_time={batch_max_time}"
        )

    hamiltonian = _load_hamiltonian(hamiltonian_path)
    num_qubits = hamiltonian.num_qubits
    n_params = 2 * reps

    ansatz = _ansatz_builder(hamiltonian, reps=reps)

    if init_params is None:
        np.random.seed(42)
        p0 = np.random.uniform(0, 2 * np.pi, size=n_params)
    else:
        p0 = np.asarray(init_params, dtype=float)

    opt_cost = np.nan
    n_evals = 0
    if optimize_on_hardware:
        opt_params = None
    else:
        # Local optimizer loop avoids repeated runtime jobs that can hit
        # interactive TTL limits, while final sampling still runs on hardware.
        local_backend = AerSimulator(method="statevector")
        local_circuit = _transpile_circuit(ansatz, local_backend, optimization_level=1)
        local_estimator = StatevectorEstimator()
        opt_params, opt_cost, n_evals = _optimize_parameters(
            p0,
            local_circuit,
            hamiltonian,
            local_estimator,
            method=optimization_method,
            maxiter=maxiter,
            ftol=ftol,
        )

    candidate_circuit = _transpile_circuit(ansatz, backend, optimization_level=1)
    with Batch(backend=backend, max_time=batch_max_time) as batch:
        sampler = RuntimeSamplerV2(mode=batch)
        if optimize_on_hardware:
            estimator = RuntimeEstimatorV2(mode=batch)
            opt_params, opt_cost, n_evals = _optimize_parameters(
                p0,
                candidate_circuit,
                hamiltonian,
                estimator,
                method=optimization_method,
                maxiter=maxiter,
                ftol=ftol,
            )

        distribution = _sample_circuit(
            candidate_circuit,
            opt_params,
            sampler,
            shots=shots,
        )

    sparse_terms = hamiltonian.to_sparse_list()
    best_bitstring, best_sample_energy = _extract_lowest_energy_bitstring(
        distribution, num_qubits, sparse_terms
    )
    top_bitstrings = (
        _extract_top_k_bitstrings(distribution, num_qubits, top_k=top_k)
        if return_topk
        else None
    )

    elapsed = time.time() - t0
    mode_label = "hw-opt" if optimize_on_hardware else "aer-opt+hw-sample"
    print(
        f"  IBM QAOA [{num_qubits}q, {reps}p, {backend_name}, {mode_label}]: "
        f"expectation={opt_cost:.6f}, sample_min={best_sample_energy:.6f}, "
        f"evals={n_evals}, selected={sum(best_bitstring)}/{num_qubits}, "
        f"time={elapsed:.1f}s"
    )

    if return_topk:
        return best_bitstring, float(best_sample_energy), top_bitstrings
    return best_bitstring, float(best_sample_energy)
