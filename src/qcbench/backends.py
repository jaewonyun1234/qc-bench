"""Backend/primitive factories for statevector and noisy simulations."""
from typing import Optional, Tuple

from qiskit.primitives import BackendEstimatorV2, BackendSamplerV2
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

_TWO_QUBIT_GATES = {"cx", "cz", "swap", "iswap", "ecr", "rxx", "ryy", "rzz"}
_DEFAULT_1Q_GATES = ["u1", "u2", "u3", "h", "rx", "ry", "rz"]
_DEFAULT_2Q_GATES = ["cx"]


def _split_basis_gates(basis_gates: Optional[list[str]]) -> tuple[list[str], list[str]]:
    """Split basis gates into 1-qubit and 2-qubit lists.

    Args:
        basis_gates: Optional list of gate names from config.

    Returns:
        Tuple of (one_qubit_gates, two_qubit_gates).
    """
    if not basis_gates:
        return list(_DEFAULT_1Q_GATES), list(_DEFAULT_2Q_GATES)

    one_q = [gate for gate in basis_gates if gate not in _TWO_QUBIT_GATES]
    two_q = [gate for gate in basis_gates if gate in _TWO_QUBIT_GATES]

    if not one_q:
        one_q = list(_DEFAULT_1Q_GATES)
    if not two_q:
        two_q = list(_DEFAULT_2Q_GATES)

    return one_q, two_q


def _build_noise_model(noise_strength: float, basis_gates: Optional[list[str]] = None) -> NoiseModel:
    """Create a simple depolarizing-noise model.

    Args:
        noise_strength: Depolarizing error probability.
        basis_gates: Optional gate list to attach errors to.

    Returns:
        Configured NoiseModel.
    """
    noise_model = NoiseModel()
    if noise_strength > 0:
        error_1q = depolarizing_error(noise_strength, 1)
        error_2q = depolarizing_error(noise_strength, 2)
        gates_1q, gates_2q = _split_basis_gates(basis_gates)
        # Apply 1-qubit errors to 1-qubit gates and 2-qubit errors to 2-qubit gates.
        noise_model.add_all_qubit_quantum_error(error_1q, gates_1q)
        noise_model.add_all_qubit_quantum_error(error_2q, gates_2q)
    return noise_model


def make_statevector_primitives(
    seed: Optional[int] = None,
) -> Tuple[BackendEstimatorV2, BackendSamplerV2, AerSimulator]:
    """Return estimator, sampler, backend for ideal statevector simulation.

    Args:
        seed: Optional simulator seed.

    Returns:
        Tuple of (estimator, sampler, backend).
    """
    backend = AerSimulator(method="statevector", seed_simulator=seed)
    estimator = BackendEstimatorV2(backend=backend)
    sampler = BackendSamplerV2(backend=backend)
    return estimator, sampler, backend


def make_noisy_primitives(
    shots: int,
    noise_strength: float,
    seed: Optional[int] = None,
    basis_gates: Optional[list[str]] = None,
) -> Tuple[BackendEstimatorV2, BackendSamplerV2, AerSimulator]:
    """Return estimator, sampler, backend with depolarizing noise on Aer.

    Args:
        shots: Number of shots for sampling.
        noise_strength: Depolarizing error probability.
        seed: Optional simulator seed.
        basis_gates: Optional gate list for noise attachment.

    Returns:
        Tuple of (estimator, sampler, backend).
    """
    noise_model = _build_noise_model(noise_strength, basis_gates=basis_gates)
    backend = AerSimulator(
        method="density_matrix",
        noise_model=noise_model,
        shots=shots,
        seed_simulator=seed,
    )
    estimator = BackendEstimatorV2(backend=backend)
    sampler = BackendSamplerV2(backend=backend)
    if shots is not None:
        estimator.options.default_shots = shots
        sampler.options.default_shots = shots
    return estimator, sampler, backend




__all__ = [
    "make_statevector_primitives",
    "make_noisy_primitives"
]
