"""Grid discretization helpers for 1D potentials.

Finite-difference discretization on a 1D grid. This builds the physical
Hamiltonian H_phys on the grid before any padding to 2^n.
"""
from typing import Callable, Optional, Tuple

import numpy as np


class GridDiscretization:
    """Implements position-space grid discretization for a 1D system.

    The grid uses N interior points with spacing dx = L / (N + 1), which is the
    standard finite-difference setup for Dirichlet boundaries.
    """

    def __init__(
        self,
        L: float,
        N: int,
        hbar: float = 1.0,
        m: float = 1.0,
        x_min: float = 0.0,
        x_max: Optional[float] = None,
    ) -> None:
        """Initialize the grid discretization.

        Args:
            L: Length of the domain.
            N: Number of interior grid points.
            hbar: Reduced Planck constant.
            m: Mass.
            x_min: Domain minimum (default 0.0).
            x_max: Domain maximum; if None, uses x_min + L.

        Raises:
            ValueError: If x_max is provided but inconsistent with L.
        """
        self.x_min = x_min
        if x_max is not None and not np.isclose(x_max - x_min, L):
            raise ValueError(
                f"Inconsistent grid: L={L} vs x_max-x_min={x_max - x_min}. "
                "Please ensure L matches the domain boundaries or set x_max=None."
            )
        self.x_max = x_max if x_max is not None else x_min + L
        self.L = L
        self.N = N
        self.hbar = hbar
        self.m = m

        self.dx = self.L / (N + 1)
        # Interior points only (exclude boundaries).
        self.x_grid = np.linspace(self.x_min + self.dx, self.x_max - self.dx, N)
        self.n_qubits = int(np.ceil(np.log2(N)))
        self.hilbert_space_dim = 2**self.n_qubits

    def build_kinetic_energy(self) -> np.ndarray:
        """Build the kinetic energy matrix using finite differences.

        Returns:
            NxN kinetic energy matrix with the standard second-derivative stencil.
        """
        coeff = -self.hbar**2 / (2 * self.m * self.dx**2)
        kinetic = np.zeros((self.N, self.N))
        # Tridiagonal stencil: [-2, 1, 1] on the diagonal and off-diagonals.
        for i in range(self.N):
            kinetic[i, i] = -2.0
            if i > 0:
                kinetic[i, i - 1] = 1.0
            if i < self.N - 1:
                kinetic[i, i + 1] = 1.0
        return coeff * kinetic

    def build_potential_energy(self, potential_func: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
        """Build the diagonal potential energy matrix V(x).

        Args:
            potential_func: Callable that maps x_grid -> V(x) values.

        Returns:
            NxN diagonal matrix with V(x) on the diagonal.
        """
        V = potential_func(self.x_grid)
        return np.diag(V)

    def build_hamiltonian(self, potential_func: Optional[Callable[[np.ndarray], np.ndarray]] = None) -> np.ndarray:
        """Build the full Hamiltonian matrix (kinetic + potential).

        Args:
            potential_func: Optional potential function. If None, uses V(x)=0.

        Returns:
            NxN physical Hamiltonian matrix H_phys.
        """
        H = self.build_kinetic_energy()
        # Add potential only if provided.
        if potential_func is not None:
            H += self.build_potential_energy(potential_func)
        return H

    def analytical_infinite_well(self, n: int) -> Tuple[float, np.ndarray]:
        """Analytical infinite square well eigenpair for mode n.

        Args:
            n: Mode index (n=1 is the ground state).

        Returns:
            Tuple of (energy, wavefunction) evaluated on the grid.
        """
        E_n = (n * np.pi * self.hbar) ** 2 / (2 * self.m * self.L**2)
        x_rel = self.x_grid - self.x_min
        psi_n = np.sqrt(2 / self.L) * np.sin(n * np.pi * x_rel / self.L)
        psi_n_discrete = np.sqrt(self.dx) * psi_n
        return E_n, psi_n_discrete

    def analytical_harmonic_oscillator(self, n: int, omega: float) -> float:
        """Analytical harmonic oscillator energy level (n=0 is ground state).

        Args:
            n: Quantum number (n=0,1,2,...).
            omega: Oscillator frequency.

        Returns:
            Energy level E_n.
        """
        return self.hbar * omega * (n + 0.5)

    def get_grid_info(self) -> dict:
        """Return grid parameters as a dictionary.

        Returns:
            Dictionary with L, N, dx, n_qubits, hilbert_dim, x_min, x_max, hbar, m.
        """
        return {
            "L": self.L,
            "N": self.N,
            "dx": self.dx,
            "n_qubits": self.n_qubits,
            "hilbert_dim": self.hilbert_space_dim,
            "x_min": self.x_grid[0],
            "x_max": self.x_grid[-1],
            "hbar": self.hbar,
            "m": self.m,
        }


__all__ = ["GridDiscretization"]
