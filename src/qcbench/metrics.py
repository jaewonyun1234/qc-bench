"""Metric helpers for benchmarking runs."""
from typing import Dict, Iterable, List, Optional

import numpy as np

_TWO_QUBIT_GATES = {"cx", "cz", "swap", "iswap", "ecr", "rxx", "ryy", "rzz"}


def count_two_qubit_gates(circuit) -> int:
    """Count two-qubit gates from a circuit's op counts.

    Args:
        circuit: QuantumCircuit instance.

    Returns:
        Total count of two-qubit gates (by name).
    """
    ops = circuit.count_ops()
    return int(sum(ops.get(name, 0) for name in _TWO_QUBIT_GATES))


def circuit_cost(circuit) -> Dict[str, int]:
    """Return depth, two-qubit count, and parameter count.

    Args:
        circuit: QuantumCircuit instance.

    Returns:
        Dict with keys: depth, two_qubit_count, num_parameters.
    """
    return {
        "depth": int(circuit.depth()),
        "two_qubit_count": count_two_qubit_gates(circuit),
        "num_parameters": int(circuit.num_parameters),
    }


def operator_cost(qubit_op) -> int:
    """Return number of Pauli terms in the operator.

    Args:
        qubit_op: SparsePauliOp or similar.

    Returns:
        Number of terms, or 0 if not available.
    """
    try:
        return int(len(qubit_op))
    except Exception:
        return 0


def _energy_columns(prefix: str, energies: Iterable[float], k: int) -> Dict[str, Optional[float]]:
    """Pack a list of energies into columns with a prefix.

    Args:
        prefix: Column prefix (e.g., "E").
        energies: Iterable of floats.
        k: Number of columns to produce.

    Returns:
        Dict mapping prefix+index -> value (or None if missing).
    """
    vals = list(energies)
    return {f"{prefix}{i}": (float(vals[i]) if i < len(vals) else None) for i in range(k)}


def _gap_list(energies: Iterable[float]) -> List[float]:
    """Compute energy gaps E[i+1] - E[i].

    Args:
        energies: Iterable of energies in ascending order.

    Returns:
        List of consecutive gaps.
    """
    vals = list(energies)
    return [vals[i + 1] - vals[i] for i in range(len(vals) - 1)]


def build_metrics_row(
    *,
    potential: str,
    ansatz_type: str,
    reps: int,
    entanglement: str,
    backend_type: str,
    noise_strength: Optional[float],
    shots: Optional[int],
    seed: int,
    grid_info: Dict[str, float],
    energies: List[float],
    exact_energies: List[float],
    circuit,
    qubit_op,
    runtime_sec: float,
    eval_count: Optional[int],
    success: bool,
    k_states: int,
    grid_energies: Optional[List[float]] = None,
    analytic_energies: Optional[List[float]] = None,
    padding_leakage: Optional[List[float]] = None,
) -> Dict[str, object]:
    """Assemble a single row of metrics for the results table.

    Args:
        potential: Potential name (isw/ho/doublewell).
        ansatz_type: Ansatz identifier.
        reps: Ansatz repetitions.
        entanglement: Entanglement pattern.
        backend_type: "statevector" or "noisy".
        noise_strength: Noise strength for noisy runs.
        shots: Shot count for noisy runs.
        seed: Random seed.
        grid_info: Dict from GridHamiltonian.get_grid_info().
        energies: VQD energies.
        exact_energies: Reference energies from padded Hamiltonian.
        circuit: QuantumCircuit used for the run.
        qubit_op: Hamiltonian as SparsePauliOp.
        runtime_sec: Wall-clock runtime for the run.
        eval_count: Estimated number of evaluations.
        success: Success flag from VQD.
        k_states: Number of states requested.
        grid_energies: Optional energies from unpadded H_phys.
        analytic_energies: Optional analytic energies (ISW/HO only).
        padding_leakage: Optional leakage values for padded eigenvectors.

    Returns:
        Flat dict representing one row in the results table.
    """
    energy_errors = [abs(e - e_ref) for e, e_ref in zip(energies, exact_energies)]
    gap_errors = [
        abs(e_gap - r_gap)
        for e_gap, r_gap in zip(_gap_list(energies), _gap_list(exact_energies))
    ]

    row: Dict[str, object] = {
        "potential": potential,
        "ansatz": ansatz_type,
        "reps": reps,
        "entanglement": entanglement,
        "backend": backend_type,
        "noise_strength": noise_strength,
        "shots": shots,
        "seed": seed,
        "k_states": k_states,
        "runtime_sec": float(runtime_sec),
        "eval_count": eval_count,
        "success": bool(success),
        "num_pauli_terms": operator_cost(qubit_op),
    }

    row.update(grid_info)
    row.update(circuit_cost(circuit))
    row.update(_energy_columns("E", energies, k_states))
    row.update(_energy_columns("E_exact", exact_energies, k_states))
    row.update(_energy_columns("E_err", energy_errors, k_states))

    if grid_energies is not None:
        row.update(_energy_columns("E_grid", grid_energies, k_states))
        pad_errors = [abs(e - g) for e, g in zip(exact_energies, grid_energies)]
        row.update(_energy_columns("E_pad_err", pad_errors, k_states))

    if analytic_energies is not None:
        row.update(_energy_columns("E_analytic", analytic_energies, k_states))
        ref = grid_energies if grid_energies is not None else exact_energies
        analytic_errors = [abs(e - a) for e, a in zip(ref, analytic_energies)]
        row.update(_energy_columns("E_analytic_err", analytic_errors, k_states))

    if padding_leakage is not None:
        leakage_vals = list(padding_leakage)
        row["padding_leakage_max"] = float(np.max(leakage_vals))
        row["padding_leakage_mean"] = float(np.mean(leakage_vals))

    # Gap errors are useful for excited-state spacing diagnostics.
    for i, gap_err in enumerate(gap_errors):
        row[f"gap_err_{i}"] = gap_err

    return row


__all__ = [
    "count_two_qubit_gates",
    "circuit_cost",
    "operator_cost",
    "build_metrics_row",
]
