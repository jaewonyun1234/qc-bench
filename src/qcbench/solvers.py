"""Solver wrappers (VQD)."""
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms import VQD
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms.state_fidelities import ComputeUncompute
from qiskit_algorithms.utils import algorithm_globals


def _build_optimizer(settings: Optional[Dict[str, Any]] = None) -> COBYLA:
    """Construct the COBYLA optimizer from settings.

    Args:
        settings: Optional dict, e.g. {"maxiter": 200}.

    Returns:
        COBYLA optimizer instance.
    """
    settings = settings or {}
    maxiter = settings.get("maxiter", 200)
    return COBYLA(maxiter=maxiter)


def run_vqd(
    qubit_op: SparsePauliOp,
    ansatz,
    estimator,
    sampler,
    k: int = 5,
    seed: Optional[int] = None,
    optimizer_settings: Optional[Dict[str, Any]] = None,
    shots: Optional[int] = None,
) -> Tuple[List[float], Dict[str, Any]]:
    """Run VQD and return energies plus metadata.

    Args:
        qubit_op: Hamiltonian as a SparsePauliOp.
        ansatz: Parameterized quantum circuit.
        estimator: Estimator primitive (BackendEstimatorV2).
        sampler: Sampler primitive (BackendSamplerV2).
        k: Number of eigenvalues to compute.
        seed: Random seed for reproducibility.
        optimizer_settings: Optimizer configuration dict.
        shots: Optional shots passed to fidelity estimator.

    Returns:
        Tuple of (energies, metadata dict).
    """
    if seed is not None:
        algorithm_globals.random_seed = seed

    optimizer = _build_optimizer(optimizer_settings)

    rng = np.random.default_rng(seed)
    if ansatz.num_parameters == 0:
        initial_points = None
    else:
        # One random start per target eigenstate.
        initial_points = [rng.random(ansatz.num_parameters) for _ in range(k)]

    fidelity = ComputeUncompute(sampler=sampler, shots=shots)
    vqd = VQD(
        estimator=estimator,
        fidelity=fidelity,
        ansatz=ansatz,
        optimizer=optimizer,
        k=k,
        initial_point=initial_points,
    )
    result = vqd.compute_eigenvalues(operator=qubit_op)

    energies = [float(np.real(ev)) for ev in result.eigenvalues]
    cost_evals = result.cost_function_evals
    eval_count = None
    if cost_evals is not None:
        try:
            eval_count = int(np.sum(cost_evals))
        except Exception:
            eval_count = None

    meta = {
        "cost_function_evals": cost_evals,
        "eval_count": eval_count,
        "optimizer_times": result.optimizer_times,
        "success": len(energies) >= k,
    }
    return energies, meta


__all__ = ["run_vqd"]
