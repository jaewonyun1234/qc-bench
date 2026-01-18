"""Run benchmark configs and write results table."""
from __future__ import annotations

import argparse
import itertools
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml
from qiskit import transpile

from qcbench.analytics import exactSolutions
from qcbench.ansatz import build_efficient_su2, build_hva
from qcbench.backends import make_noisy_primitives, make_statevector_primitives
from qcbench.hamiltonian import GridHamiltonian
from qcbench.metrics import build_metrics_row
from qcbench.potentials import V_doublewell, V_ho, V_isw
from qcbench.solvers import run_vqd


def load_config(path: Path) -> Dict[str, Any]:
    """Load a YAML config file.

    Args:
        path: Path to a YAML config.

    Returns:
        Parsed config dict.

    Raises:
        ValueError: If the config is empty.
    """
    cfg = yaml.safe_load(path.read_text())
    if not cfg:
        raise ValueError(f"Config is empty: {path}")
    return cfg


def _grid_from_config(cfg: Dict[str, Any]) -> GridHamiltonian:
    """Build GridHamiltonian from config.

    Args:
        cfg: Parsed config dict.

    Returns:
        GridHamiltonian instance.
    """
    grid_cfg = cfg.get("grid", {})
    L = grid_cfg.get("L")
    N = grid_cfg.get("N")
    if L is None or N is None:
        raise ValueError("grid.L and grid.N are required")

    hbar = grid_cfg.get("hbar", 1.0)
    m = grid_cfg.get("m", 1.0)

    if grid_cfg.get("centered", False):
        x_min = -L / 2
        x_max = L / 2
    else:
        x_min = grid_cfg.get("x_min", 0.0)
        x_max = grid_cfg.get("x_max", None)

    return GridHamiltonian(L=L, N=N, hbar=hbar, m=m, x_min=x_min, x_max=x_max)


def _potential_fn(cfg: Dict[str, Any]):
    """Select potential function based on config.

    Args:
        cfg: Parsed config dict.

    Returns:
        Callable mapping x_grid -> V(x).
    """
    pot_cfg = cfg.get("potential", {})
    name = pot_cfg.get("name")
    params = pot_cfg.get("params", {})

    if name == "isw":
        return lambda x: V_isw(x, L=params.get("L"))
    if name == "ho":
        return lambda x: V_ho(x, **params)
    if name in {"doublewell", "double_well"}:
        return lambda x: V_doublewell(x, **params)

    raise ValueError(f"Unknown potential name: {name}")




def _ansatz_configs(cfg: Dict[str, Any]) -> Tuple[List[str], List[int], str]:
    """Extract ansatz-related settings from config.

    Returns:
        Tuple of (types, reps_list, entanglement).
    """
    ans_cfg = cfg.get("ansatz", {})
    types = ans_cfg.get("types", ["efficient_su2", "hva"])
    reps_list = ans_cfg.get("reps", [1])
    entanglement = ans_cfg.get("entanglement", "linear")
    return types, reps_list, entanglement


def _backend_configs(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Extract backend configuration block."""
    return cfg.get("backend", {})


def _ensure_results_dir(path: Path) -> None:
    """Ensure results output directory exists."""
    path.parent.mkdir(parents=True, exist_ok=True)


def _load_existing(path: Path) -> pd.DataFrame:
    """Load existing results if present.

    Returns:
        Existing results DataFrame or empty DataFrame.
    """
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()


def _format_matrix(matrix: np.ndarray, precision: int = 4) -> str:
    """Return a nicely formatted matrix string for CLI output."""
    df = pd.DataFrame(matrix)
    with pd.option_context(
        "display.precision",
        precision,
        "display.max_columns",
        None,
        "display.width",
        160,
    ):
        return df.to_string()


def run_config(config_path: Path, output_path: Path, append: bool = True) -> pd.DataFrame:
    """Run a single config and write results.

    Args:
        config_path: Path to YAML config.
        output_path: Path to output parquet file.
        append: If True, append to existing results.

    Returns:
        DataFrame of results written to disk.
    """
    cfg = load_config(config_path)
    grid = _grid_from_config(cfg)
    pot_fn = _potential_fn(cfg)
    solutions = exactSolutions(grid)

    # Build physical Hamiltonian and its padded qubit representation.
    H_phys = grid.build_hamiltonian(pot_fn)
    qubit_op, exact_evals, n_qubits, padded = grid.build_qubit_hamiltonian(
        H_phys, penalty_factor=cfg.get("hamiltonian", {}).get("penalty_factor", 1e3)
    )

    k_states = int(cfg.get("vqd", {}).get("k_states", 5))
    # eigenvalues from the padded
    exact_energies = [float(e) for e in exact_evals[:k_states]]
    # eigenvalues from N*N physical Hamiltonian
    grid_evals = np.linalg.eigvalsh(H_phys).real
    grid_energies = [float(e) for e in grid_evals[:k_states]]
    analytic_energies = solutions.analytic_energies(cfg, k_states)
    leak = exactSolutions.padding_leakage(padded, n_phys=H_phys.shape[0], k=k_states)

    ansatz_types, reps_list, entanglement = _ansatz_configs(cfg)
    backend_cfg = _backend_configs(cfg)
    seeds = cfg.get("seeds", [0])

    # Precompute kinetic/potential operators for HVA.
    kinetic = grid.build_kinetic_energy()
    potential = grid.build_potential_energy(pot_fn)
    kinetic_padded, _ = grid.pad_to_qubit_dimension(kinetic, penalty_factor=0.0)
    potential_padded, _ = grid.pad_to_qubit_dimension(potential, penalty_factor=0.0)
    kinetic_op = grid.to_pauli_op(kinetic_padded)
    potential_op = grid.to_pauli_op(potential_padded)

    if cfg.get("debug", {}).get("print_padded_matrices", False):
        print("\nKinetic padded matrix:")
        print(_format_matrix(kinetic_padded))
        print("\nPotential padded matrix:")
        print(_format_matrix(potential_padded))

    rows: List[Dict[str, Any]] = []

    for ansatz_type, reps in itertools.product(ansatz_types, reps_list):
        if ansatz_type == "efficient_su2":
            base_ansatz = build_efficient_su2(n_qubits, reps=reps, entanglement=entanglement)
        elif ansatz_type == "hva":
            base_ansatz = build_hva(n_qubits, kinetic_op, potential_op, reps=reps)
        else:
            raise ValueError(f"Unknown ansatz type: {ansatz_type}")

        # Statevector backend
        if backend_cfg.get("statevector", True):
            for seed in seeds:
                estimator, sampler, backend = make_statevector_primitives(seed=seed)
                ansatz = transpile(base_ansatz, backend)
                start = time.perf_counter()
                energies, meta = run_vqd(
                    qubit_op,
                    ansatz,
                    estimator,
                    sampler,
                    k=k_states,
                    seed=seed,
                    optimizer_settings=cfg.get("vqd", {}).get("optimizer", {}),
                )
                runtime = time.perf_counter() - start

                row = build_metrics_row(
                    potential=cfg.get("potential", {}).get("name"),
                    ansatz_type=ansatz_type,
                    reps=reps,
                    entanglement=entanglement,
                    backend_type="statevector",
                    noise_strength=None,
                    shots=None,
                    seed=seed,
                    grid_info=grid.get_grid_info(),
                    energies=energies,
                    exact_energies=exact_energies,
                    circuit=ansatz,
                    qubit_op=qubit_op,
                    runtime_sec=runtime,
                    eval_count=meta.get("eval_count"),
                    success=meta.get("success", False),
                    k_states=k_states,
                    grid_energies=grid_energies,
                    analytic_energies=analytic_energies,
                    padding_leakage=leak,
                )
                rows.append(row)

        # Noisy backend
        noisy_cfg = backend_cfg.get("noisy", {})
        noise_strengths = noisy_cfg.get("noise_strengths", [])
        shots_list = noisy_cfg.get("shots", [])
        basis_gates = noisy_cfg.get("basis_gates", None)

        for noise_strength, shots, seed in itertools.product(noise_strengths, shots_list, seeds):
            estimator, sampler, backend = make_noisy_primitives(
                shots=shots,
                noise_strength=noise_strength,
                seed=seed,
                basis_gates=basis_gates,
            )
            ansatz = transpile(base_ansatz, backend)
            start = time.perf_counter()
            energies, meta = run_vqd(
                qubit_op,
                ansatz,
                estimator,
                sampler,
                k=k_states,
                seed=seed,
                optimizer_settings=cfg.get("vqd", {}).get("optimizer", {}),
                shots=shots,
            )
            runtime = time.perf_counter() - start

            row = build_metrics_row(
                potential=cfg.get("potential", {}).get("name"),
                ansatz_type=ansatz_type,
                reps=reps,
                entanglement=entanglement,
                backend_type="noisy",
                noise_strength=noise_strength,
                shots=shots,
                seed=seed,
                grid_info=grid.get_grid_info(),
                energies=energies,
                exact_energies=exact_energies,
                circuit=ansatz,
                qubit_op=qubit_op,
                runtime_sec=runtime,
                eval_count=meta.get("eval_count"),
                success=meta.get("success", False),
                k_states=k_states,
                grid_energies=grid_energies,
                analytic_energies=analytic_energies,
                padding_leakage=leak,
            )
            rows.append(row)

    df = pd.DataFrame(rows)
    _ensure_results_dir(output_path)

    if append:
        existing = _load_existing(output_path)
        if not existing.empty:
            df = pd.concat([existing, df], ignore_index=True)

    df.to_parquet(output_path, index=False)
    return df


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Run qc-bench config.")
    parser.add_argument("config", type=str, help="Path to YAML config")
    parser.add_argument("--output", type=str, default="results/runs.parquet")
    parser.add_argument("--no-append", action="store_true")
    args = parser.parse_args()

    run_config(Path(args.config), Path(args.output), append=not args.no_append)


if __name__ == "__main__":
    main()
