"""Exact/analytic solutions and diagnostics for 1D systems."""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


class exactSolutions:
    """Exact solutions and diagnostics for grid-based Hamiltonians."""

    def __init__(self, grid) -> None:
        self.grid = grid

    def analytical_infinite_well(self, n: int):
        """Analytical infinite square well eigenpair for mode n.

        Args:
            n: Mode index (n=1 is the ground state).

        Returns:
            Tuple of (energy, wavefunction) evaluated on the grid.
        """
        E_n = (n * np.pi * self.grid.hbar) ** 2 / (2 * self.grid.m * self.grid.L**2)
        x_rel = self.grid.x_grid - self.grid.x_min
        psi_n = np.sqrt(2 / self.grid.L) * np.sin(n * np.pi * x_rel / self.grid.L)
        psi_n_discrete = np.sqrt(self.grid.dx) * psi_n
        return E_n, psi_n_discrete

    def analytical_harmonic_oscillator(self, n: int, omega: float) -> float:
        """Analytical harmonic oscillator energy level (n=0 is ground state).

        Args:
            n: Quantum number (n=0,1,2,...).
            omega: Oscillator frequency.

        Returns:
            Energy level E_n.
        """
        return self.grid.hbar * omega * (n + 0.5)

    def analytic_energies(self, cfg: Dict[str, Any], k_states: int) -> Optional[list[float]]:
        """Compute analytic energies where closed-form solutions exist.

        Args:
            cfg: Parsed config dict.
            k_states: Number of energy levels to return.

        Returns:
            List of analytic energies for ISW/HO, or None for other potentials.
        """
        pot_cfg = cfg.get("potential", {})
        name = pot_cfg.get("name")
        params = pot_cfg.get("params", {})

        if name == "isw":
            return [self.analytical_infinite_well(n=i + 1)[0] for i in range(k_states)]
        if name == "ho":
            omega = params.get("omega")
            if omega is None:
                return None
            return [self.analytical_harmonic_oscillator(n=i, omega=omega) for i in range(k_states)]

        return None

    @staticmethod
    def exact_spectrum(padded_matrix: np.ndarray) -> np.ndarray:
        """Reference eigenvalues from the padded matrix (ground truth).

        Args:
            padded_matrix: Dense 2^n x 2^n matrix.

        Returns:
            Eigenvalues sorted in ascending order.
        """
        evals, _ = np.linalg.eigh(padded_matrix)
        return evals

    @staticmethod
    def padding_leakage(padded_matrix: np.ndarray, n_phys: int, k: int) -> np.ndarray:
        """Return leakage (weight outside physical subspace) for the lowest k states.

        Args:
            padded_matrix: Dense 2^n x 2^n matrix.
            n_phys: Dimension of the physical subspace (original grid size).
            k: Number of lowest-energy states to analyze.

        Returns:
            1D array of length k. Each entry is the leakage for one eigenvector.
        """
        evals, evecs = np.linalg.eigh(padded_matrix)
        _ = evals  # keep order consistent; values not needed here
        # Sum weight in the physical block of each eigenvector.
        weights = np.sum(np.abs(evecs[:n_phys, :k]) ** 2, axis=0)
        return 1.0 - weights


__all__ = ["exactSolutions"]
