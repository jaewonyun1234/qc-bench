"""Example unit tests using the built-in unittest framework.

Run (from repo root):
  ./qc-bench-venv/bin/python -m unittest discover -s tests -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
sys.path.insert(0, str(_SRC_ROOT))


from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp

from qcbench.ansatz import build_hva
from qcbench.analytics import exactSolutions
from qcbench.backends import _build_noise_model, _split_basis_gates
from qcbench.hamiltonian import GridHamiltonian


class TestBackends(unittest.TestCase):
    def test_split_basis_gates_defaults_when_none(self) -> None:
        one_q, two_q = _split_basis_gates(None)
        self.assertGreater(len(one_q), 0)
        self.assertGreater(len(two_q), 0)
        self.assertIn("cx", two_q)

    def test_split_basis_gates_fallback_when_missing_one_qubit(self) -> None:
        one_q, two_q = _split_basis_gates(["cx"])
        self.assertIn("cx", two_q)
        self.assertIn("u3", one_q)

    def test_build_noise_model_off_when_strength_zero(self) -> None:
        noise_model = _build_noise_model(0.0, basis_gates=["u3", "cx"])
        self.assertEqual(noise_model.to_dict().get("errors"), [])

    def test_build_noise_model_attaches_errors(self) -> None:
        noise_model = _build_noise_model(0.01, basis_gates=["u3", "cx"])
        errors = noise_model.to_dict().get("errors", [])
        self.assertGreater(len(errors), 0)
        operations = {op for err in errors for op in err.get("operations", [])}
        self.assertIn("u3", operations)
        self.assertIn("cx", operations)


class TestHamiltonian(unittest.TestCase):
    def test_pad_to_qubit_dimension_adds_penalty(self) -> None:
        h_phys = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, -1.0],
            ],
            dtype=float,
        )
        padded, n_qubits = GridHamiltonian.pad_to_qubit_dimension(h_phys, penalty_factor=10.0)
        self.assertEqual(n_qubits, 2)
        self.assertEqual(padded.shape, (4, 4))
        np.testing.assert_allclose(padded[:3, :3], h_phys)
        np.testing.assert_allclose(padded[:3, 3], 0.0)
        np.testing.assert_allclose(padded[3, :3], 0.0)
        expected_lambda = 10.0 * np.max(np.abs(h_phys))
        self.assertAlmostEqual(float(padded[3, 3]), float(expected_lambda))

    def test_to_pauli_op_round_trip(self) -> None:
        matrix = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=float)
        op = GridHamiltonian.to_pauli_op(matrix)
        self.assertIsInstance(op, SparsePauliOp)
        np.testing.assert_allclose(op.to_matrix(), matrix)


class TestAnalytics(unittest.TestCase):
    def test_analytic_infinite_well_energy(self) -> None:
        grid = GridHamiltonian(L=1.0, N=4)
        sol = exactSolutions(grid)
        energy, _ = sol.analytical_infinite_well(n=1)
        expected = (np.pi * grid.hbar) ** 2 / (2 * grid.m * grid.L**2)
        self.assertAlmostEqual(float(energy), float(expected))

    def test_padding_leakage_for_diagonal_padded_matrix(self) -> None:
        padded = np.diag([0.0, 1.0, 100.0, 100.0]).astype(float)
        leak = exactSolutions.padding_leakage(padded, n_phys=2, k=2)
        np.testing.assert_allclose(leak, [0.0, 0.0])


class TestAnsatz(unittest.TestCase):
    def test_build_hva_skips_identity_terms(self) -> None:
        potential_op = SparsePauliOp(["II", "ZI"], coeffs=[1.0, 0.5])
        kinetic_op = SparsePauliOp(["II", "XX"], coeffs=[1.0, 0.7])
        circuit = build_hva(n_qubits=2, kinetic_op=kinetic_op, potential_op=potential_op, reps=2)

        evo_gates = [ci.operation for ci in circuit.data if isinstance(ci.operation, PauliEvolutionGate)]
        self.assertEqual(len(evo_gates), 4)
        self.assertEqual(circuit.num_parameters, 4)


if __name__ == "__main__":
    unittest.main()
