import os
import time
import json
import importlib
import importlib.metadata as importlib_metadata
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
from scipy.optimize import minimize
from qiskit import transpile
from qiskit.circuit.library import QAOAAnsatz
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler import CouplingMap


def _resolve_iqm_provider():
    import_candidates = [
        ("iqm.qiskit_iqm", "IQMProvider"),
        ("qiskit_iqm", "IQMProvider"),
        ("qiskit_on_iqm", "IQMProvider"),
    ]
    import_errors = []

    for module_name, attr_name in import_candidates:
        try:
            module = importlib.import_module(module_name)
            provider_cls = getattr(module, attr_name, None)
            if provider_cls is not None:
                return provider_cls
        except Exception as exc:  # pragma: no cover - import-path dependent
            import_errors.append(f"{module_name}: {exc.__class__.__name__}: {exc}")

    try:
        iqm_client_version = importlib_metadata.version("iqm-client")
    except Exception:
        iqm_client_version = None

    hint_lines = [
        "IQM Qiskit adapter is not installed in the active Python environment.",
        "Install/update IQM client per official installation guidance:",
        "  pip uninstall -y qiskit-iqm cirq-iqm",
        "  pip install --upgrade --force-reinstall \"iqm-client[qiskit,cirq]\"",
        f"Tried imports: {', '.join(path for path, _ in import_candidates)}",
    ]
    if iqm_client_version is not None:
        hint_lines.append(f"Detected iqm-client version: {iqm_client_version}")
        hint_lines.append(
            "Older iqm-client versions may not include the qiskit adapter module."
        )
    hint = "\n".join(hint_lines)
    if import_errors:
        hint += "\nImport errors:\n  " + "\n  ".join(import_errors)
    raise ImportError(hint)


def _resolve_token_json_path(token_json_path=None):
    if token_json_path is not None:
        candidate = Path(token_json_path).expanduser()
        if candidate.is_absolute():
            return candidate.resolve()

        # Prefer repository-root relative resolution for worker processes that
        # may execute from a different cwd.
        repo_candidate = Path(__file__).resolve().parents[2] / candidate
        if repo_candidate.exists():
            return repo_candidate.resolve()

        cwd_candidate = (Path.cwd() / candidate).resolve()
        if cwd_candidate.exists():
            return cwd_candidate

        # Return repo-root candidate as deterministic fallback path.
        return repo_candidate.resolve()

    repo_token = Path(__file__).resolve().parents[2] / "token.json"
    if repo_token.exists():
        return repo_token

    cwd_token = Path.cwd() / "token.json"
    return cwd_token.resolve()


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
    )


def _load_token_json_payload(token_json_path=None):
    path = _resolve_token_json_path(token_json_path)
    if not path.exists():
        return path, None
    with path.open("r", encoding="utf-8") as f:
        return path, json.load(f)


def _build_iqm_client_auth_args(iqm_token=None, token_json_path=None):
    path, payload = _load_token_json_payload(token_json_path=token_json_path)
    if payload and payload.get("access_token") and payload.get("auth_server_url"):
        return {"tokens_file": str(path)}, path

    token = iqm_token
    if not token and payload:
        token = (
            payload.get("iqm_token")
            or payload.get("token")
            or payload.get("apikey")
            or payload.get("api_key")
        )
    if not token:
        token = os.environ.get("IQM_TOKEN")

    if token:
        token = str(token).strip()
        if token:
            return {"token": token}, path
    return {}, path


def _candidate_iqm_client_urls(server_url, quantum_computer):
    original_url = str(server_url).rstrip("/")
    candidates = [original_url]

    parsed = urlparse(server_url)
    host = (parsed.netloc or "").lower()
    path = parsed.path.strip("/")
    qpu = str(quantum_computer).strip("/") if quantum_computer else ""

    if host == "resonance.meetiqm.com":
        candidates = [
            f"{parsed.scheme}://cocos.resonance.meetiqm.com",
            f"{parsed.scheme}://resonance.meetiqm.com",
            original_url,
        ]
        if qpu:
            candidates.append(f"{parsed.scheme}://cocos.resonance.meetiqm.com/{qpu}")

    if host == "cocos.resonance.meetiqm.com":
        # Prefer base host before path-suffixed URLs, because iqm-client calls
        # '/quantum-architecture' relative to this base.
        candidates = [
            f"{parsed.scheme}://cocos.resonance.meetiqm.com",
            original_url,
            f"{parsed.scheme}://resonance.meetiqm.com",
        ]
        if qpu and path != qpu:
            candidates.append(f"{parsed.scheme}://cocos.resonance.meetiqm.com/{qpu}")

    seen = set()
    out = []
    for url in candidates:
        if url and url not in seen:
            seen.add(url)
            out.append(url)
    return out


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


def _extract_scalar_expectation(raw_evs):
    arr = np.asarray(raw_evs)
    if arr.size == 0:
        raise ValueError("Estimator returned empty expectation values.")
    return float(np.real(arr.reshape(-1)[0]))


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


def _counts_to_distribution(counts, num_qubits):
    total = float(sum(counts.values()))
    if total <= 0:
        return {}

    distribution = {}
    for key, value in counts.items():
        if isinstance(key, int):
            state_int = int(key)
        else:
            bits = "".join(ch for ch in str(key) if ch in "01")
            if not bits:
                continue
            if len(bits) < num_qubits:
                bits = bits.rjust(num_qubits, "0")
            elif len(bits) > num_qubits:
                bits = bits[-num_qubits:]
            state_int = int(bits, 2)

        distribution[state_int] = distribution.get(state_int, 0.0) + (
            float(value) / total
        )
    return distribution


def _expected_energy_from_distribution(distribution, num_qubits, sparse_terms):
    expectation = 0.0
    for state_int, prob in distribution.items():
        bitstring = _to_bitstring(int(state_int), num_qubits)
        bitstring.reverse()
        expectation += float(prob) * _bitstring_energy(bitstring, sparse_terms)
    return float(expectation)


def _cost_func_estimator_local(params, ansatz, hamiltonian, estimator_instance):
    pub = (ansatz, hamiltonian, params)
    job = estimator_instance.run([pub])
    result = job.result()[0]
    return _extract_scalar_expectation(result.data.evs)


def _optimize_parameters_local(
    init_params,
    ansatz,
    hamiltonian,
    estimator_instance,
    method="SLSQP",
    maxiter=200,
    ftol=1e-6,
):
    eval_count = [0]

    def _cost(params):
        eval_count[0] += 1
        return _cost_func_estimator_local(
            params,
            ansatz,
            hamiltonian,
            estimator_instance,
        )

    res = minimize(
        _cost,
        init_params,
        method=method,
        options={"maxiter": int(maxiter), "ftol": float(ftol)},
    )
    return np.asarray(res.x, dtype=float), float(res.fun), int(eval_count[0])


def _wrap_periodic_params(params):
    return np.mod(np.asarray(params, dtype=float), 2.0 * np.pi)


def _optimize_parameters_spsa_hardware(
    init_params,
    evaluate_batch,
    maxiter=25,
    learning_rate=0.4,
    perturbation=0.2,
    alpha=0.602,
    gamma=0.101,
    stability_constant=None,
    random_seed=42,
):
    theta = _wrap_periodic_params(init_params)
    maxiter = max(1, int(maxiter))
    stability = (
        float(stability_constant)
        if stability_constant is not None
        else max(1.0, 0.1 * float(maxiter))
    )
    rng = np.random.default_rng(int(random_seed))
    n_evals = 0

    for iteration in range(maxiter):
        ak = float(learning_rate) / ((iteration + 1 + stability) ** float(alpha))
        ck = float(perturbation) / ((iteration + 1) ** float(gamma))
        delta = rng.choice([-1.0, 1.0], size=theta.shape)
        theta_plus = _wrap_periodic_params(theta + ck * delta)
        theta_minus = _wrap_periodic_params(theta - ck * delta)
        energies, _ = evaluate_batch([theta_plus, theta_minus])
        n_evals += 2
        y_plus = float(energies[0])
        y_minus = float(energies[1])
        ghat = ((y_plus - y_minus) / (2.0 * ck)) * delta
        theta = _wrap_periodic_params(theta - ak * ghat)
        print(
            f"  IQM SPSA iter {iteration + 1}/{maxiter}: "
            f"y_plus={y_plus:.6f}, y_minus={y_minus:.6f}"
        )

    final_energies, final_distributions = evaluate_batch([theta])
    n_evals += 1
    return (
        _wrap_periodic_params(theta),
        float(final_energies[0]),
        final_distributions[0],
        int(n_evals),
    )


def _evaluate_parameter_batch(
    backend,
    transpiled_template,
    parameter_values_batch,
    hamiltonian,
    sparse_terms,
    shots,
):
    circuits = [
        transpiled_template.assign_parameters(param_values)
        for param_values in parameter_values_batch
    ]
    job = backend.run(circuits, shots=int(shots))
    result = job.result()
    raw_counts = result.get_counts()
    if isinstance(raw_counts, dict):
        raw_counts = [raw_counts]

    distributions = [
        _counts_to_distribution(counts, hamiltonian.num_qubits) for counts in raw_counts
    ]
    energies = [
        _expected_energy_from_distribution(
            distribution, hamiltonian.num_qubits, sparse_terms
        )
        for distribution in distributions
    ]
    return energies, distributions


def _select_connected_qubits(available_qubits, connectivity, num_qubits):
    if num_qubits < 1:
        raise ValueError("num_qubits must be >= 1.")
    if num_qubits > len(available_qubits):
        raise ValueError(
            f"Requested {num_qubits} qubits, but architecture has only {len(available_qubits)}."
        )
    if num_qubits == 1:
        return [available_qubits[0]]

    adjacency = {q: set() for q in available_qubits}
    for edge in connectivity:
        if len(edge) != 2:
            continue
        a, b = edge
        if a in adjacency and b in adjacency:
            adjacency[a].add(b)
            adjacency[b].add(a)

    for start in available_qubits:
        queue = [start]
        seen = {start}
        ordered = [start]
        while queue and len(ordered) < num_qubits:
            node = queue.pop(0)
            for nbr in sorted(adjacency[node]):
                if nbr not in seen:
                    seen.add(nbr)
                    queue.append(nbr)
                    ordered.append(nbr)
                    if len(ordered) >= num_qubits:
                        break
        if len(ordered) >= num_qubits:
            return ordered[:num_qubits]

    raise RuntimeError(
        f"Could not find a connected {num_qubits}-qubit subset on this IQM architecture."
    )


def _build_coupling_map(selected_qubits, connectivity):
    if len(selected_qubits) <= 1:
        return None

    index_of = {name: idx for idx, name in enumerate(selected_qubits)}
    edges = []
    for edge in connectivity:
        if len(edge) != 2:
            continue
        a, b = edge
        if a in index_of and b in index_of:
            ia, ib = int(index_of[a]), int(index_of[b])
            edges.append([ia, ib])
            edges.append([ib, ia])

    if not edges:
        raise RuntimeError(
            "No coupling edges found for selected qubits; cannot transpile with CZ basis."
        )
    return CouplingMap(edges)


def _append_rz_as_prx(instructions, qubit_name, theta, Instruction):
    # Rz(theta) = Ry(-pi/2) * Rx(theta) * Ry(pi/2), with Ry implemented as prx phase=0.25.
    instructions.append(
        Instruction(
            name="prx",
            qubits=(qubit_name,),
            args={"angle_t": -0.25, "phase_t": 0.25},
        )
    )
    instructions.append(
        Instruction(
            name="prx",
            qubits=(qubit_name,),
            args={"angle_t": float(theta) / (2.0 * np.pi), "phase_t": 0.0},
        )
    )
    instructions.append(
        Instruction(
            name="prx",
            qubits=(qubit_name,),
            args={"angle_t": 0.25, "phase_t": 0.25},
        )
    )


def _qiskit_to_iqm_circuit(bound_circuit, selected_qubits, Circuit, Instruction, name):
    instructions = []
    num_qubits = int(bound_circuit.num_qubits)

    for inst in bound_circuit.data:
        op_name = inst.operation.name
        q_indices = [bound_circuit.find_bit(q).index for q in inst.qubits]
        q_names = tuple(selected_qubits[idx] for idx in q_indices)

        if op_name == "rx":
            theta = float(inst.operation.params[0])
            instructions.append(
                Instruction(
                    name="prx",
                    qubits=q_names,
                    args={"angle_t": theta / (2.0 * np.pi), "phase_t": 0.0},
                )
            )
        elif op_name == "rz":
            theta = float(inst.operation.params[0])
            _append_rz_as_prx(instructions, q_names[0], theta, Instruction)
        elif op_name == "cz":
            instructions.append(Instruction(name="cz", qubits=q_names, args={}))
        elif op_name == "barrier":
            instructions.append(Instruction(name="barrier", qubits=q_names, args={}))
        elif op_name == "measure":
            # Collapsed into a single trailing measurement instruction for stable keying.
            continue
        else:
            raise ValueError(
                "Unsupported gate in iqm-client fallback conversion: "
                f"{op_name}. Try lowering reps/optimization level."
            )

    instructions.append(
        Instruction(
            name="measure",
            qubits=tuple(selected_qubits[:num_qubits]),
            args={"key": "m"},
        )
    )
    return Circuit(name=name, instructions=tuple(instructions))


def _shots_to_counts_int(shots_matrix):
    counts = {}
    for shot_bits in shots_matrix:
        bits = "".join(str(int(b)) for b in reversed(shot_bits))
        state_int = int(bits, 2) if bits else 0
        counts[state_int] = counts.get(state_int, 0) + 1
    return counts


def _evaluate_parameter_batch_iqm_client(
    client,
    transpiled_template,
    parameter_values_batch,
    hamiltonian,
    sparse_terms,
    shots,
    selected_qubits,
    Circuit,
    Instruction,
    timeout_secs=1800,
):
    iqm_circuits = []
    for idx, param_values in enumerate(parameter_values_batch):
        bound = transpiled_template.assign_parameters(param_values)
        iqm_circuit = _qiskit_to_iqm_circuit(
            bound,
            selected_qubits,
            Circuit,
            Instruction,
            name=f"qaoa_{idx}",
        )
        iqm_circuits.append(iqm_circuit)

    job_id = client.submit_circuits(iqm_circuits, shots=int(shots))
    run_result = client.wait_for_results(job_id, timeout_secs=float(timeout_secs))
    if run_result.measurements is None:
        raise RuntimeError(
            f"IQM client run did not return measurements. Status={run_result.status}, "
            f"message={run_result.message}"
        )

    distributions = []
    for measurement_map in run_result.measurements:
        if not measurement_map:
            distributions.append({})
            continue
        key = "m" if "m" in measurement_map else next(iter(measurement_map.keys()))
        counts_int = _shots_to_counts_int(measurement_map[key])
        distributions.append(_counts_to_distribution(counts_int, hamiltonian.num_qubits))

    energies = [
        _expected_energy_from_distribution(
            distribution, hamiltonian.num_qubits, sparse_terms
        )
        for distribution in distributions
    ]
    return energies, distributions


def run_qaoa_iqm_batch(
    hamiltonian_path,
    iqm_server_url=None,
    quantum_computer=None,
    iqm_token=None,
    token_json_path=None,
    reps=1,
    init_params=None,
    shots=1000,
    n_iterations=12,
    candidates_per_iteration=8,
    search_scale=np.pi / 3,
    search_decay=0.75,
    optimization_level=3,
    layout_method="sabre",
    allow_direct_client_fallback=True,
    result_timeout_secs=1800,
    return_topk=False,
    top_k=10,
    random_seed=42,
    optimization_method="SLSQP",
    ftol=1e-6,
    optimize_on_hardware=False,
    spsa_learning_rate=0.4,
    spsa_perturbation=0.2,
    spsa_alpha=0.602,
    spsa_gamma=0.101,
    spsa_stability_constant=None,
):
    """
    Run QAOA on IQM hardware using batched circuit execution.

    Follows IQM guidance to transpile the parameterized circuit once and then
    run a batch of parameter-bound circuits with equal shots.
    IQM token is taken from iqm_token, then token.json, then IQM_TOKEN env var.
    If IQM Qiskit adapter is unavailable, falls back to direct iqm-client mode.
    """
    server_url = iqm_server_url or os.environ.get("IQM_SERVER_URL")
    if not server_url:
        raise ValueError(
            "iqm_server_url is required (or set IQM_SERVER_URL env variable)."
        )

    quantum_computer = quantum_computer or os.environ.get("IQM_QUANTUM_COMPUTER")
    client_auth_args, resolved_token_json_path = _build_iqm_client_auth_args(
        iqm_token=iqm_token,
        token_json_path=token_json_path,
    )
    token = client_auth_args.get("token")

    requires_auth = "meetiqm.com" in (urlparse(server_url).netloc or "").lower()
    if requires_auth and not client_auth_args:
        raise ValueError(
            "No IQM authentication credentials found. "
            f"Checked token_json_path={resolved_token_json_path} and IQM_TOKEN env var."
        )

    t0 = time.time()
    hamiltonian = _load_hamiltonian(hamiltonian_path)
    num_qubits = hamiltonian.num_qubits
    sparse_terms = hamiltonian.to_sparse_list()

    ansatz = QAOAAnsatz(hamiltonian, reps=reps)
    n_params = int(ansatz.num_parameters)
    measured_template = ansatz.copy()
    measured_template.measure_all()
    execution_mode = None
    evaluate_batch = None

    provider_error = None
    if allow_direct_client_fallback:
        try:
            IQMProvider = _resolve_iqm_provider()
            provider_kwargs = {}
            if quantum_computer:
                provider_kwargs["quantum_computer"] = quantum_computer
            if token:
                provider_kwargs["token"] = token

            provider = IQMProvider(server_url, **provider_kwargs)
            backend = provider.get_backend()
            transpiled_template = transpile(
                measured_template,
                backend=backend,
                optimization_level=int(optimization_level),
                layout_method=layout_method,
            )

            def evaluate_batch(parameter_values_batch):
                return _evaluate_parameter_batch(
                    backend,
                    transpiled_template,
                    parameter_values_batch,
                    hamiltonian,
                    sparse_terms,
                    shots,
                )

            execution_mode = "iqm-provider"
        except ImportError as exc:
            provider_error = exc
    else:
        IQMProvider = _resolve_iqm_provider()
        provider_kwargs = {}
        if quantum_computer:
            provider_kwargs["quantum_computer"] = quantum_computer
        if token:
            provider_kwargs["token"] = token
        provider = IQMProvider(server_url, **provider_kwargs)
        backend = provider.get_backend()
        transpiled_template = transpile(
            measured_template,
            backend=backend,
            optimization_level=int(optimization_level),
            layout_method=layout_method,
        )

        def evaluate_batch(parameter_values_batch):
            return _evaluate_parameter_batch(
                backend,
                transpiled_template,
                parameter_values_batch,
                hamiltonian,
                sparse_terms,
                shots,
            )

        execution_mode = "iqm-provider"

    if execution_mode is None:
        try:
            from iqm.iqm_client import (
                Circuit,
                IQMClient,
                Instruction,
            )
        except ImportError as exc:
            if provider_error is not None:
                raise ImportError(f"{provider_error}") from exc
            raise

        architecture = None
        client = None
        attempted_urls = []
        auth_errors = []
        try:
            iqm_client_version = importlib_metadata.version("iqm-client")
        except Exception:  # pragma: no cover - env/package dependent
            iqm_client_version = "unknown"
        for candidate_url in _candidate_iqm_client_urls(server_url, quantum_computer):
            attempted_urls.append(candidate_url)
            try:
                candidate_client = IQMClient(candidate_url, **client_auth_args)
                candidate_arch = candidate_client.get_quantum_architecture().quantum_architecture
                client = candidate_client
                architecture = candidate_arch
                server_url = candidate_url
                break
            except Exception as exc:  # pragma: no cover - network/auth dependent
                auth_errors.append(f"{candidate_url}: {type(exc).__name__}: {exc}")

        if client is None or architecture is None:
            raise RuntimeError(
                "IQM connection failed for all candidate server URLs. "
                f"Attempted: {attempted_urls}. "
                "Verify endpoint, token validity/scope, and whether the base host "
                "should omit the quantum computer path suffix (for example '/garnet'). "
                f"Detected iqm-client version: {iqm_client_version}. "
                f"Errors: {auth_errors}"
            )

        selected_qubits = _select_connected_qubits(
            architecture.qubits,
            architecture.qubit_connectivity,
            num_qubits,
        )
        coupling_map = _build_coupling_map(
            selected_qubits,
            architecture.qubit_connectivity,
        )
        transpile_kwargs = {
            "basis_gates": ["rx", "rz", "cz"],
            "optimization_level": int(optimization_level),
            "layout_method": layout_method,
        }
        if coupling_map is not None:
            transpile_kwargs["coupling_map"] = coupling_map
        transpiled_template = transpile(measured_template, **transpile_kwargs)

        def evaluate_batch(parameter_values_batch):
            return _evaluate_parameter_batch_iqm_client(
                client,
                transpiled_template,
                parameter_values_batch,
                hamiltonian,
                sparse_terms,
                shots,
                selected_qubits,
                Circuit,
                Instruction,
                timeout_secs=result_timeout_secs,
            )

        execution_mode = "iqm-client"

    rng = np.random.default_rng(int(random_seed))
    if init_params is None:
        best_params = rng.uniform(0.0, 2 * np.pi, size=n_params)
    else:
        best_params = np.asarray(init_params, dtype=float).reshape(-1)
        if best_params.size != n_params:
            raise ValueError(f"init_params must have length {n_params}, got {best_params.size}")

    optimization_label = str(optimization_method).strip() if optimization_method else ""
    optimization_key = optimization_label.lower()
    if optimize_on_hardware:
        if optimization_key == "spsa":
            best_params, best_energy, best_distribution, n_evals = (
                _optimize_parameters_spsa_hardware(
                    best_params,
                    evaluate_batch,
                    maxiter=max(1, int(n_iterations)),
                    learning_rate=spsa_learning_rate,
                    perturbation=spsa_perturbation,
                    alpha=spsa_alpha,
                    gamma=spsa_gamma,
                    stability_constant=spsa_stability_constant,
                    random_seed=random_seed,
                )
            )
            execution_mode = f"{execution_mode}+hw-spsa"
        elif optimization_key in {"random-search", ""}:
            # Baseline evaluation for random-search mode.
            energies, distributions = evaluate_batch([best_params])
            best_energy = float(energies[0])
            best_distribution = distributions[0]
            n_evals = 1

            sigma = float(search_scale)
            for _ in range(int(n_iterations)):
                batch_size = max(int(candidates_per_iteration), 1)
                candidates = [best_params]
                for _ in range(batch_size - 1):
                    candidate = best_params + rng.normal(0.0, sigma, size=n_params)
                    candidate = np.mod(candidate, 2 * np.pi)
                    candidates.append(candidate)

                energies, distributions = evaluate_batch(candidates)
                n_evals += len(candidates)

                best_idx = int(np.argmin(energies))
                if float(energies[best_idx]) < best_energy:
                    best_energy = float(energies[best_idx])
                    best_params = np.asarray(candidates[best_idx], dtype=float)
                    best_distribution = distributions[best_idx]

                sigma *= float(search_decay)
            execution_mode = f"{execution_mode}+hw-random-search"
        else:
            raise ValueError(
                "Hardware optimization currently supports optimization_method="
                "'SPSA' or 'random-search'. "
                f"Received: {optimization_method!r}"
            )
    elif optimization_label and optimization_key != "random-search":
        estimator = StatevectorEstimator()
        best_params, best_energy, n_evals = _optimize_parameters_local(
            best_params,
            ansatz,
            hamiltonian,
            estimator,
            method=optimization_label,
            maxiter=max(1, int(n_iterations)),
            ftol=ftol,
        )
        _, distributions = evaluate_batch([best_params])
        best_distribution = distributions[0]
        execution_mode = f"{execution_mode}+{optimization_key}"
    else:
        # Baseline evaluation for random-search mode.
        energies, distributions = evaluate_batch([best_params])
        best_energy = float(energies[0])
        best_distribution = distributions[0]
        n_evals = 1

        sigma = float(search_scale)
        for _ in range(int(n_iterations)):
            batch_size = max(int(candidates_per_iteration), 1)
            candidates = [best_params]
            for _ in range(batch_size - 1):
                candidate = best_params + rng.normal(0.0, sigma, size=n_params)
                candidate = np.mod(candidate, 2 * np.pi)
                candidates.append(candidate)

            energies, distributions = evaluate_batch(candidates)
            n_evals += len(candidates)

            best_idx = int(np.argmin(energies))
            if float(energies[best_idx]) < best_energy:
                best_energy = float(energies[best_idx])
                best_params = np.asarray(candidates[best_idx], dtype=float)
                best_distribution = distributions[best_idx]

            sigma *= float(search_decay)
        execution_mode = f"{execution_mode}+random-search"

    if not best_distribution:
        raise RuntimeError("IQM run returned an empty distribution.")

    best_bitstring, best_sample_energy = _extract_lowest_energy_bitstring(
        best_distribution,
        num_qubits,
        sparse_terms,
    )
    top_bitstrings = (
        _extract_top_k_bitstrings(best_distribution, num_qubits, top_k=top_k)
        if return_topk
        else None
    )

    elapsed = time.time() - t0
    print(
        f"  IQM QAOA [{num_qubits}q, {reps}p, mode={execution_mode}]: "
        f"expectation={best_energy:.6f}, sample_min={best_sample_energy:.6f}, "
        f"evals={n_evals}, selected={sum(best_bitstring)}/{num_qubits}, "
        f"url={server_url}, "
        f"time={elapsed:.1f}s"
    )

    if return_topk:
        return best_bitstring, float(best_sample_energy), top_bitstrings
    return best_bitstring, float(best_sample_energy)
