import psutil
import math

def estimate_hamiltonian_concurrency(
    target_cpu_frac=0.8,
    cpu_per_job_frac=0.35,
    min_concurrency=1,
):
    total_cpus = psutil.cpu_count(logical=True)

    max_jobs = math.floor(
        (target_cpu_frac * total_cpus) / cpu_per_job_frac
    )

    return max(min_concurrency, max_jobs)

