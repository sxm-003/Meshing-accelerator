import numpy as np
from mpi4py import MPI
from qulacs import QuantumCircuit, QuantumState
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


def _basis_energies(n_qubits, ops, pos, coeffs):
    dim = 1 << n_qubits
    states = np.arange(dim, dtype=np.uint64)
    energies = np.zeros(dim, dtype=np.float64)

    for op, pos_row, coeff in zip(ops, pos, coeffs):
        op = str(op)
        valid_pos = _valid_positions(pos_row)
        if not valid_pos:
            continue

        if set(op) != {"Z"} or len(valid_pos) != len(op):
            raise ValueError(
                "HPC QAOA energy evaluation currently supports only diagonal Z-string terms. "
                f"Received op={op!r}, positions={valid_pos!r}."
            )

        term = np.ones(dim, dtype=np.float64)
        for qubit_idx in valid_pos:
            bits = ((states >> np.uint64(qubit_idx)) & np.uint64(1)).astype(np.float64)
            term *= 1.0 - 2.0 * bits

        energies += float(np.real(coeff)) * term

    return energies


def _simulate_expectation(n_qubits, ops, pos, coeffs, params, basis_energies):
    p = len(params) // 2
    gammas = params[:p]
    betas = params[p:]

    state = QuantumState(n_qubits)
    state.set_zero_state()

    circuit = build_qaoa_circuit(n_qubits, ops, pos, coeffs, gammas, betas)
    circuit.update_quantum_state(state)

    probs = np.abs(state.get_vector()) ** 2
    expectation = float(np.dot(probs, basis_energies))
    return expectation, probs


def _evaluate_parameter_batch(comm, n_qubits, ops, pos, coeffs, basis_energies, params_batch):
    batch = comm.bcast(params_batch, root=0)
    rank = comm.Get_rank()
    size = comm.Get_size()

    local_results = []
    for idx in range(rank, len(batch), size):
        params = _wrap_periodic(batch[idx])
        expectation, _ = _simulate_expectation(
            n_qubits,
            ops,
            pos,
            coeffs,
            params,
            basis_energies,
        )
        local_results.append((idx, expectation))

    gathered = comm.gather(local_results, root=0)
    if rank != 0:
        return None

    ordered = [None] * len(batch)
    for chunk in gathered:
        for idx, expectation in chunk:
            ordered[idx] = float(expectation)

    return ordered


def _index_to_bitstring(index, n_qubits):
    return "".join("1" if ((index >> qubit) & 1) else "0" for qubit in range(n_qubits))


def _select_lowest_energy_sample(probs, basis_energies, shots, random_seed):
    rng = np.random.default_rng(int(random_seed))
    state_indices = np.arange(probs.size, dtype=np.int64)
    sampled = rng.choice(state_indices, size=max(1, int(shots)), p=probs)
    unique_sampled = np.unique(sampled)
    best_sampled_idx = int(unique_sampled[np.argmin(basis_energies[unique_sampled])])
    most_probable_idx = int(np.argmax(probs))
    return best_sampled_idx, most_probable_idx


def _optimize_parameters_spsa(
    comm,
    n_qubits,
    ops,
    pos,
    coeffs,
    basis_energies,
    p=1,
    maxiter=40,
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
            params_batch = [theta_plus, theta_minus]
        else:
            ck = None
            delta = None
            params_batch = None

        batch_energies = _evaluate_parameter_batch(
            comm,
            n_qubits,
            ops,
            pos,
            coeffs,
            basis_energies,
            params_batch,
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
                f"  HPC SPSA iter {iteration + 1}/{maxiter}: "
                f"y_plus={y_plus:.6f}, y_minus={y_minus:.6f}, best={best_expectation:.6f}",
                flush=True,
            )

    if rank == 0:
        final_batch = [theta]
    else:
        final_batch = None

    final_energy = _evaluate_parameter_batch(
        comm,
        n_qubits,
        ops,
        pos,
        coeffs,
        basis_energies,
        final_batch,
    )

    if rank != 0:
        return None, None

    final_expectation = float(final_energy[0])
    if final_expectation < best_expectation:
        best_expectation = final_expectation
        best_theta = np.asarray(theta, dtype=float)

    return np.asarray(best_theta, dtype=float), float(best_expectation)


def run_qaoa_qulacs_mpi(
    npz_path,
    p=1,
    maxiter=40,
    shots=4096,
    learning_rate=0.4,
    perturbation=0.2,
    alpha=0.602,
    gamma=0.101,
    stability_constant=None,
    random_seed=42,
):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    ops, pos, coeffs, n_qubits = load_hamiltonian(npz_path)
    basis_energies = _basis_energies(n_qubits, ops, pos, coeffs)

    best_params, best_expectation = _optimize_parameters_spsa(
        comm,
        n_qubits,
        ops,
        pos,
        coeffs,
        basis_energies,
        p=p,
        maxiter=maxiter,
        learning_rate=learning_rate,
        perturbation=perturbation,
        alpha=alpha,
        gamma=gamma,
        stability_constant=stability_constant,
        random_seed=random_seed,
    )

    if rank != 0:
        return None, None, None

    final_expectation, probs = _simulate_expectation(
        n_qubits,
        ops,
        pos,
        coeffs,
        best_params,
        basis_energies,
    )
    selected_index, most_probable_index = _select_lowest_energy_sample(
        probs,
        basis_energies,
        shots=shots,
        random_seed=random_seed,
    )
    bitstring = _index_to_bitstring(selected_index, n_qubits)
    bit_energy = float(basis_energies[selected_index])

    metadata = {
        "optimizer": "SPSA",
        "reps": int(p),
        "maxiter": int(maxiter),
        "shots": int(shots),
        "best_params": np.asarray(best_params, dtype=float).tolist(),
        "expectation": float(final_expectation),
        "selection_mode": "lowest-energy-sampled-bitstring",
        "selected_probability": float(probs[selected_index]),
        "most_probable_bitstring": _index_to_bitstring(most_probable_index, n_qubits),
        "most_probable_probability": float(probs[most_probable_index]),
        "most_probable_energy": float(basis_energies[most_probable_index]),
    }
    print(
        f"  HPC QAOA [{n_qubits}q, {p}p]: "
        f"expectation={final_expectation:.6f}, "
        f"selected_prob={probs[selected_index]:.6f}, "
        f"selected_energy={bit_energy:.6f}, "
        f"most_probable_energy={basis_energies[most_probable_index]:.6f}",
        flush=True,
    )
    return bitstring, bit_energy, metadata
