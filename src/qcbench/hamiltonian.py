"""Grid-based Hamiltonian builder and qubit-space helpers.

This module merges grid discretization with Hamiltonian-to-qubit conversion.
"""
from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np
from qiskit.quantum_info import SparsePauliOp

from qcbench.analytics import exactSolutions


class GridHamiltonian:
    """Build grid-discretized Hamiltonians and qubit operators."""

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

    def build_qubit_hamiltonian(
        self, h_phys: np.ndarray, penalty_factor: float = 1e3, atol: float = 1e-10
    ) -> Tuple[SparsePauliOp, np.ndarray, int, np.ndarray]:
        """Convert H_phys to qubit operator and compute reference spectrum.

        Args:
            h_phys: Physical Hamiltonian (NxN) from discretization.
            penalty_factor: Padding penalty multiplier for unphysical states.
            atol: Simplification tolerance for Pauli conversion.

        Returns:
            Tuple of (qubit_op, exact_evals, n_qubits, padded_matrix).
        """
        padded, n_qubits = self.pad_to_qubit_dimension(h_phys, penalty_factor=penalty_factor)
        qubit_op = self.to_pauli_op(padded, atol=atol)
        exact_evals = exactSolutions.exact_spectrum(padded)
        return qubit_op, exact_evals, n_qubits, padded

    @staticmethod
    def pad_to_qubit_dimension(h_phys: np.ndarray, penalty_factor: float = 1e3) -> Tuple[np.ndarray, int]:
        """Pad matrix to size 2^n_qubits x 2^n_qubits for qubit encoding.

        The top-left N x N block contains the physical Hamiltonian. The remaining
        basis states (unphysical) are given a large diagonal energy penalty (Lambda)
        so that low-energy eigenstates stay inside the physical subspace.

        Args:
            h_phys: Original matrix of size N x N.
            penalty_factor: Multiplier that sets how large the energy penalty (Lambda) is
                relative to the physical energy scale.

        Returns:
            Tuple of (padded_matrix, n_qubits).
        """
        if h_phys.ndim != 2 or h_phys.shape[0] != h_phys.shape[1]:
            raise ValueError("h_phys must be a square matrix")

        n = h_phys.shape[0]
        n_qubits = int(np.ceil(np.log2(n)))
        dim = 2**n_qubits

        if dim == n:
            return h_phys, n_qubits

        padded = np.zeros((dim, dim), dtype=h_phys.dtype)
        padded[:n, :n] = h_phys

        max_entry = np.max(np.abs(h_phys))
        if max_entry == 0:
            max_entry = 1.0
        # Penalize the padded subspace to keep low-energy states physical.
        Lambda = penalty_factor * max_entry

        padded[n:, n:] = np.eye(dim - n) * Lambda
        return padded, n_qubits

    @staticmethod
    def to_pauli_op(padded_matrix: np.ndarray, atol: float = 1e-10) -> SparsePauliOp:
        """Convert dense Hermitian matrix to SparsePauliOp.

        Args:
            padded_matrix: Dense 2^n x 2^n matrix.
            atol: Simplification tolerance for dropping small coefficients.

        Returns:
            SparsePauliOp representation of the input matrix.
        """
        return SparsePauliOp.from_operator(padded_matrix).simplify(atol=atol)

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


__all__ = ["GridHamiltonian"]
