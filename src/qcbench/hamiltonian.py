"""Qubit-space Hamiltonian helpers.

This module pads the physical Hamiltonian to 2^n and converts it to a
SparsePauliOp for Qiskit. It also provides small diagnostics for padding.
"""
from typing import Tuple

import numpy as np
from qiskit.quantum_info import SparsePauliOp


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


def truncate_interaction(matrix: np.ndarray, distance: int = 1) -> np.ndarray:
    """Zero out entries beyond a given interaction distance.

    Args:
        matrix: Dense matrix to truncate.
        distance: Maximum allowed |i-j| for nonzero entries.

    Returns:
        A copy of the matrix with elements beyond the band set to 0.

    Note: for the finite-difference kinetic matrix used here, distance=1 is
    already the natural bandwidth, so this is effectively a no-op unless you
    set distance < 1 or use a wider-band discretization in the future.
    """
    trunc_matrix = matrix.copy()
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if abs(i - j) > distance:
                trunc_matrix[i, j] = 0.0
    return trunc_matrix


def to_sparse_pauli_op(padded_matrix: np.ndarray, atol: float = 1e-10) -> SparsePauliOp:
    """Convert dense Hermitian matrix to SparsePauliOp.

    Args:
        padded_matrix: Dense 2^n x 2^n matrix.
        atol: Simplification tolerance for dropping small coefficients.

    Returns:
        SparsePauliOp representation of the input matrix.

    Example:
        >>> import numpy as np
        >>> H = np.array([[1, 1], [1, 1]], dtype=float)
        >>> op = to_sparse_pauli_op(H)
        >>> op
        SparsePauliOp(['I', 'X'], coeffs=[1.+0.j, 1.+0.j])

    """
    return SparsePauliOp.from_operator(padded_matrix).simplify(atol=atol)


def exact_spectrum(padded_matrix: np.ndarray) -> np.ndarray:
    """Reference eigenvalues from the padded matrix (ground truth).

    Args:
        padded_matrix: Dense 2^n x 2^n matrix.

    Returns:
        Eigenvalues sorted in ascending order.
    """
    evals, _ = np.linalg.eigh(padded_matrix)
    return evals


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


def build_qubit_hamiltonian(
    h_phys: np.ndarray, penalty_factor: float = 1e3, atol: float = 1e-10
) -> Tuple[SparsePauliOp, np.ndarray, int, np.ndarray]:
    """Convert H_phys to qubit operator and compute reference spectrum.

    Args:
        h_phys: Physical Hamiltonian (NxN) from discretization.
        penalty_factor: Padding penalty multiplier for unphysical states.
        atol: Simplification tolerance for Pauli conversion.

    Returns:
        Tuple of (qubit_op, exact_evals, n_qubits, padded_matrix).
    """
    padded, n_qubits = pad_to_qubit_dimension(h_phys, penalty_factor=penalty_factor)
    qubit_op = to_sparse_pauli_op(padded, atol=atol)
    exact_evals = exact_spectrum(padded)
    return qubit_op, exact_evals, n_qubits, padded


__all__ = [
    "pad_to_qubit_dimension",
    "truncate_interaction",
    "to_sparse_pauli_op",
    "exact_spectrum",
    "padding_leakage",
    "build_qubit_hamiltonian",
]
