import numpy as np
from mpi4py import MPI
from qulacs import Observable, QuantumCircuit, QuantumState
from qulacs.gate import PauliRotation


def load_hamiltonian(npz_path):
    data = np.load(npz_path, allow_pickle=False)
    coeff_key = "coeffs" if "coeffs" in data else "sparse_coeffs"
    num_qubits = int(np.asarray(data["num_qubits"]).reshape(-1)[0])
    return (
        data["sparse_ops"],
        data["sparse_positions"],
        data[coeff_key],
        num_qubits,
    )


def _valid_positions(pos_row):
    return [int(x) for x in np.asarray(pos_row).reshape(-1) if int(x) >= 0]


def _wrap_periodic(params):
    return np.mod(np.asarray(params, dtype=float), 2.0 * np.pi)


def _pauli_string(op, positions):
    parts = []
    for pauli, qubit in zip(str(op), positions):
        parts.append(pauli)
        parts.append(str(int(qubit)))
    return " ".join(parts)


def build_observable(n_qubits, ops, pos, coeffs):
    observable = Observable(n_qubits)
    for op, pos_row, coeff in zip(ops, pos, coeffs):
        valid_pos = _valid_positions(pos_row)
        if not valid_pos:
            continue
        observable.add_operator(float(np.real(coeff)), _pauli_string(op, valid_pos))
    return observable


def build_qaoa_circuit(n_qubits, ops, pos, coeffs, gammas, betas):
    circuit = QuantumCircuit(n_qubits)

    for qubit in range(n_qubits):
        circuit.add_H_gate(qubit)

    for gamma, beta in zip(gammas, betas):
        for op, pos_row, coeff in zip(ops, pos, coeffs):
            op = str(op)
            valid_pos = _valid_positions(pos_row)
            if not valid_pos:
                continue

            pauli_ids = []
            for ch in op:
                if ch == "X":
                    pauli_ids.append(1)
                elif ch == "Y":
                    pauli_ids.append(2)
                elif ch == "Z":
                    pauli_ids.append(3)
                else:
                    raise ValueError(f"Unsupported Pauli character {ch!r} in op {op!r}.")

            circuit.add_gate(
                PauliRotation(valid_pos, pauli_ids, 2.0 * float(gamma) * float(np.real(coeff)))
            )

        for qubit in range(n_qubits):
            circuit.add_RX_gate(qubit, 2.0 * float(beta))

    return circuit


def _simulate_expectation(n_qubits, ops, pos, coeffs, params, observable):
    p = len(params) // 2
    gammas = params[:p]
    betas = params[p:]

    state = QuantumState(n_qubits)
    state.set_zero_state()

    circuit = build_qaoa_circuit(n_qubits, ops, pos, coeffs, gammas, betas)
    circuit.update_quantum_state(state)

    expectation = float(np.real(observable.get_expectation_value(state)))
    return expectation, state


def _evaluate_parameter_list(comm, n_qubits, ops, pos, coeffs, observable, params_list):
    params_list = comm.bcast(params_list, root=0)
    results = []
    for params in params_list:
        expectation, _ = _simulate_expectation(
            n_qubits,
            ops,
            pos,
            coeffs,
            _wrap_periodic(params),
            observable,
        )
        results.append(float(expectation))
    return results if comm.Get_rank() == 0 else None


def _index_to_bitstring(index, n_qubits):
    return "".join("1" if ((int(index) >> qubit) & 1) else "0" for qubit in range(n_qubits))


def _basis_index_energy(index, ops, pos, coeffs):
    energy = 0.0
    index = int(index)
    for op, pos_row, coeff in zip(ops, pos, coeffs):
        op = str(op)
        valid_pos = _valid_positions(pos_row)
        if not valid_pos:
            continue
        if set(op) != {"Z"} or len(valid_pos) != len(op):
            raise ValueError(
                "Sample energy evaluation supports only diagonal Z-string terms. "
                f"Received op={op!r}, positions={valid_pos!r}."
            )

        value = 1.0
        for qubit_idx in valid_pos:
            bit = (index >> int(qubit_idx)) & 1
            value *= 1.0 - 2.0 * bit
        energy += float(np.real(coeff)) * value
    return float(energy)


def _select_lowest_energy_sample(state, ops, pos, coeffs, shots, random_seed):
    sampled = np.asarray(state.sampling(max(1, int(shots)), int(random_seed)), dtype=np.int64)
    unique_sampled = np.unique(sampled)
    sampled_energies = np.asarray([
        _basis_index_energy(index, ops, pos, coeffs)
        for index in unique_sampled
    ], dtype=float)
    best_pos = int(np.argmin(sampled_energies))
    return int(unique_sampled[best_pos]), float(sampled_energies[best_pos]), int(len(unique_sampled))


def _optimize_parameters_spsa(
    comm,
    n_qubits,
    ops,
    pos,
    coeffs,
    observable,
    p=1,
    maxiter=20,
    learning_rate=0.4,
    perturbation=0.2,
    alpha=0.602,
    gamma=0.101,
    stability_constant=None,
    random_seed=42,
):
    rank = comm.Get_rank()
    n_params = 2 * int(p)

    if rank == 0:
        rng = np.random.default_rng(int(random_seed))
        theta = rng.uniform(0.0, 2.0 * np.pi, size=n_params)
        best_theta = np.asarray(theta, dtype=float).copy()
        best_expectation = np.inf
        stability = (
            float(stability_constant)
            if stability_constant is not None
            else max(1.0, 0.1 * float(maxiter))
        )
    else:
        rng = None
        theta = None
        best_theta = None
        best_expectation = None
        stability = None

    for iteration in range(max(1, int(maxiter))):
        if rank == 0:
            ak = float(learning_rate) / ((iteration + 1 + stability) ** float(alpha))
            ck = float(perturbation) / ((iteration + 1) ** float(gamma))
            delta = rng.choice([-1.0, 1.0], size=n_params)
            theta_plus = _wrap_periodic(theta + ck * delta)
            theta_minus = _wrap_periodic(theta - ck * delta)
            params_list = [theta_plus, theta_minus]
        else:
            ck = None
            delta = None
            params_list = None

        batch_energies = _evaluate_parameter_list(
            comm,
            n_qubits,
            ops,
            pos,
            coeffs,
            observable,
            params_list,
        )

        if rank == 0:
            y_plus = float(batch_energies[0])
            y_minus = float(batch_energies[1])

            if y_plus < best_expectation:
                best_expectation = y_plus
                best_theta = np.asarray(theta_plus, dtype=float)
            if y_minus < best_expectation:
                best_expectation = y_minus
                best_theta = np.asarray(theta_minus, dtype=float)

            gradient = ((y_plus - y_minus) / (2.0 * ck)) * delta
            theta = _wrap_periodic(theta - ak * gradient)

            print(
                f"  Multi-node SPSA iter {iteration + 1}/{maxiter}: "
                f"y_plus={y_plus:.6f}, y_minus={y_minus:.6f}, best={best_expectation:.6f}",
                flush=True,
            )

    if rank == 0:
        final_list = [theta]
    else:
        final_list = None

    final_energy = _evaluate_parameter_list(
        comm,
        n_qubits,
        ops,
        pos,
        coeffs,
        observable,
        final_list,
    )

    if rank != 0:
        return None, None

    final_expectation = float(final_energy[0])
    if final_expectation < best_expectation:
        best_expectation = final_expectation
        best_theta = np.asarray(theta, dtype=float)

    return np.asarray(best_theta, dtype=float), float(best_expectation)


def run_qaoa_qulacs_mpi_observable(
    npz_path,
    p=1,
    maxiter=20,
    shots=2048,
    learning_rate=0.4,
    perturbation=0.2,
    alpha=0.602,
    gamma=0.101,
    stability_constant=None,
    random_seed=42,
):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    ops, pos, coeffs, n_qubits = load_hamiltonian(npz_path)
    observable = build_observable(n_qubits, ops, pos, coeffs)

    if rank == 0:
        print(
            f"  Multi-node QAOA setup: n_qubits={n_qubits}, mpi_ranks={size}, "
            f"terms={len(coeffs)}, reps={p}, maxiter={maxiter}, shots={shots}",
            flush=True,
        )

    best_params, best_expectation = _optimize_parameters_spsa(
        comm,
        n_qubits,
        ops,
        pos,
        coeffs,
        observable,
        p=p,
        maxiter=maxiter,
        learning_rate=learning_rate,
        perturbation=perturbation,
        alpha=alpha,
        gamma=gamma,
        stability_constant=stability_constant,
        random_seed=random_seed,
    )

    best_params = comm.bcast(best_params, root=0)

    final_expectation, final_state = _simulate_expectation(
        n_qubits,
        ops,
        pos,
        coeffs,
        best_params,
        observable,
    )

    selected_index, selected_energy, unique_samples = _select_lowest_energy_sample(
        final_state,
        ops,
        pos,
        coeffs,
        shots=shots,
        random_seed=random_seed,
    )

    if rank != 0:
        return None, None, None

    bitstring = _index_to_bitstring(selected_index, n_qubits)

    metadata = {
        "optimizer": "SPSA",
        "runner": "qaoa_qulacs_mpi_observable",
        "reps": int(p),
        "maxiter": int(maxiter),
        "shots": int(shots),
        "mpi_ranks": int(size),
        "best_params": np.asarray(best_params, dtype=float).tolist(),
        "expectation": float(final_expectation),
        "best_expectation_seen": float(best_expectation),
        "selection_mode": "lowest-energy-sampled-bitstring",
        "unique_samples": int(unique_samples),
    }
    print(
        f"  Multi-node QAOA [{n_qubits}q, {p}p, {size} ranks]: "
        f"expectation={final_expectation:.6f}, selected_energy={selected_energy:.6f}",
        flush=True,
    )
    return bitstring, selected_energy, metadata
