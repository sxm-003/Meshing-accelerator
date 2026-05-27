# Meshing-accelerator (main)

This folder contains the end-to-end mesh pipeline, Prefect flows, QAOA/Aer backends, and the HPC workflow.

## 1) Environment creation

From the repo root:
`Kindly rename Meshing-accelerator as main`
```bash
cd main
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Notes:
- Python >= 3.10 is required.
- If you run HPC jobs locally (MPI/Qulacs), install system MPI libraries first.

## 2) pyproject.toml

The project is managed by [main/pyproject.toml](main/pyproject.toml). If you add new modules, update dependencies there and reinstall:

```bash
python -m pip install -e .
```

## 3) Prefect setup

The flows are standard Prefect flows and can be run locally without a server. If you want orchestration and UI:

```bash
prefect server start
```

Then run the pipeline in another terminal:

```bash
cd main
python -m orchestrator.flow
```

Prefect task concurrency is controlled inside [main/orchestrator/flow.py](main/orchestrator/flow.py) via:
- `hamiltonian_concurrency`
- `parallel_qaoa`
- `qaoa_concurrency`

If you run notebooks that invoke the flows, start Prefect first (or run in local-only mode without the server).

## 4) Hamiltonian penalty terms and tuning

Penalty construction lives in [main/quantum_processing/hamiltonian_builder.py](main/quantum_processing/hamiltonian_builder.py). The main entry point is `hamiltonian_builder(...)`.

Core penalty flags and weights:
- Domain penalty: `alpha`, `band`
- Spacing penalty: `gamma`
- Sparsity penalty: `use_sparsity`, `N`, `mu`
- Repulsion penalty: `use_repulsion`, `d_min`, `eta`
- Bend penalty: `use_bend`, `kappa`
- Max edge penalty: `use_max_edge`, `d_max`, `eta_max`
- Density penalty: `use_density_field`, `density_radius`, `gamma_density`
- Angular bins penalty: `use_angular_bins`, `num_angular_bins`, `eta_theta`
- Collinearity penalty: `use_collinearity_penalty`, `eta_col`
- Boundary alignment penalty: `use_boundary_alignment`, `boundary_nodes`, `beta`

Tuning multipliers:
- `tuning_factors` lets you scale each penalty family after normalization.
- Example:

```python
H = hamiltonian_builder(
    phi,
    r,
    alpha=1.0,
    gamma=0.5,
    use_bend=True,
    kappa=0.2,
    tuning_factors={
        "domain": 1.5,
        "spacing": 0.8,
        "bend": 1.2,
    },
)
```

## 5) Pipeline parameters (DXF input, QAOA knobs, adaptive nodes)

The main entry point is `mesh_hamiltonian_pipeline(...)` in [main/orchestrator/flow.py](main/orchestrator/flow.py).

DXF input:
- Set `dxf_path` to your DXF file.
- Example:

```python
mesh_hamiltonian_pipeline("data/sample.dxf")
```

Key parameters to tune:
- `Q_max`: max qubits per patch. Lower values produce smaller patch sizes.
- `adaptive_nodes`: enable adaptive node density (boundary/curvature aware).
- `L_nodes`: base node spacing (used for uniform mode and adaptive fallback).
- `L_fine`, `L_coarse`: fine/coarse spacing for adaptive mode (optional).
- `overlap_factor`: patch overlap (0.0 = none, 1.0 = standard, >1.0 = more overlap).
- `qaoa_backend`: choose backend (`"aer"` or `"hpc"`).
- `qaoa_backend_config`: backend-specific config (see below).

### Backend selection

Aer (local simulator):
```python
mesh_hamiltonian_pipeline(
    "data/sample.dxf",
    qaoa_backend="aer",
    qaoa_backend_config={
        "aer_max_parallel_threads": 0,
        "aer_max_parallel_experiments": 0,
        "aer_max_parallel_shots": 0,
        "log_backend_config": True,
    },
)
```

HPC (remote scheduler):
```python
mesh_hamiltonian_pipeline(
    "data/sample.dxf",
    qaoa_backend="hpc",
    qaoa_backend_config={
        "remote_run_dir": "~/hpc_runs/your_run",
        "remote_log_dir": "~/qaoa_logs/your_run",
        "poll_interval_seconds": 15,
    },
)
```

## 6) Fujitsu simulator / HPC backend (QSIM)

This workflow uses the scripts in [main/HPC_Jobs](main/HPC_Jobs) and the SSH host name `qsim` (see [main/orchestrator/flow.py](main/orchestrator/flow.py)).

Important notes:
- Copy all files in [main/HPC_Jobs](main/HPC_Jobs) into the MPI QARP-enabled environment.
- Do NOT include `qaoa_qulacs_mpi3.py` or `run_qaoa3.sh` in that environment.

### SSH config

Create a host entry that matches the name `qsim`:

```text
Host qsim
  HostName <your-hpc-host>
  User <your-username>
  IdentityFile ~/.ssh/<your-key>
  ServerAliveInterval 30
```

You can test the connection and pipeline with [main/2d_testHPC.ipynb](main/2d_testHPC.ipynb).

To force the HPC backend in the pipeline:

```python
mesh_hamiltonian_pipeline("data/hpc_test_sq.dxf", qaoa_backend="hpc")
```

## 7) Aer backend (local users)

If you are not using the HPC backend, default to Aer:

```python
mesh_hamiltonian_pipeline("data/sample.dxf", qaoa_backend="aer")
```

Warning: keep `Q_max` < 16 when using the Aer simulator to avoid large local simulation slowdowns.

You can also cap parallelism using `qaoa_concurrency` and toggle `parallel_qaoa`.

## 8) IQM hardware access (test_modules_random.ipynb)

The IQM hardware example lives in [main/test_modules_random.ipynb](main/test_modules_random.ipynb).

Minimum setup:
- Set the IQM endpoint and target system:
    - `IQM_SERVER_URL` (example in the notebook)
    - `IQM_QUANTUM_COMPUTER` (for example `garnet`)
- Provide an IQM token either via environment or a local token file.

Token handling options:
- Environment variable: `IQM_TOKEN`
- JSON token file passed in the notebook (for example via `token_json_path`)

Keep token files out of Git and store them outside the repo or under a locally ignored `tokens/` folder.

## 9) Annealing flow (work in progress)

The annealing flow is implemented in [main/orchestrator/flow_annealer.py](main/orchestrator/flow_annealer.py). It mirrors the QAOA pipeline but sends Hamiltonians to a D-Wave sampler. This is under active development and may change.

## 10) Outputs and artifacts

Every run writes to a run-scoped folder under `outputs/` (Prefect flow run id):
- `outputs/<run_id>/hamiltonians`: saved sparse-Pauli Hamiltonians
- `outputs/<run_id>/records`: patch records (including bitstrings)
- `outputs/<run_id>/mesh`: final mesh files (`.msh`, `.vtk`, `.obj`)
- `outputs/<run_id>/merged_indices.npy`: merged node indices
- `outputs/<run_id>/critical_merged_indices.npy`: merged critical nodes

If you need a custom output directory, update the path construction in [main/orchestrator/flow.py](main/orchestrator/flow.py).

Kindly contact me on any issues at Discord: sxm_dead . I have indeed used AI at places of commenting, some part of code and Readme.md creation as well, I have verified them but do let me know in cases of any error. Thank you. 
Hope you enjoy my tinkering :\)  
