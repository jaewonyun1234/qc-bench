"""Ansatz builders (efficient_su2 and physics-informed HVA)."""
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import PauliEvolutionGate, efficient_su2
from qiskit.quantum_info import SparsePauliOp


def build_efficient_su2(n_qubits: int, reps: int = 1, entanglement: str = "linear") -> QuantumCircuit:
    """Construct a hardware-efficient efficient_su2 ansatz.

    Args:
        n_qubits: Number of qubits.
        reps: Number of layers (repetitions).
        entanglement: Entanglement pattern (e.g., "linear").

    Returns:
        Parameterized QuantumCircuit implementing efficient_su2.
    """
    return efficient_su2(
        n_qubits,
        su2_gates=["ry", "rz"],
        entanglement=entanglement,
        reps=reps,
    )


def build_hva(
    n_qubits: int,
    kinetic_op: SparsePauliOp,
    potential_op: SparsePauliOp,
    reps: int = 1,
) -> QuantumCircuit:
    """Build a truncated Hamiltonian variational ansatz (HVA).

    Structure: [Potential Layer] -> [Kinetic Layer] repeated.

    Args:
        n_qubits: Number of qubits.
        kinetic_op: Truncated kinetic operator (SparsePauliOp).
        potential_op: Potential operator (SparsePauliOp).
        reps: Number of layers (repetitions).

    Returns:
        Parameterized QuantumCircuit implementing the HVA.
    """
    circuit = QuantumCircuit(n_qubits)
    params = ParameterVector("theta", 2 * reps)

    for layer in range(reps):
        theta_v = params[2 * layer]
        theta_t = params[2 * layer + 1]

        # Potential layer: diagonal terms (mostly Z strings).
        for pauli_string, coeff in potential_op.label_iter():
            if all(c=="I" for c in pauli_string):
                continue
            gate_op = SparsePauliOp([pauli_string], coeffs=[coeff.real])
            evo_gate = PauliEvolutionGate(gate_op, time=theta_v)
            circuit.append(evo_gate, range(n_qubits))

        # Kinetic layer: truncated off-diagonal terms.
        for pauli_string, coeff in kinetic_op.label_iter():
            if all(c == "I" for c in pauli_string):
                continue
            gate_op = SparsePauliOp([pauli_string], coeffs=[coeff.real])
            evo_gate = PauliEvolutionGate(gate_op, time=theta_t)
            circuit.append(evo_gate, range(n_qubits))

    return circuit



def draw_circuit(circuit, output: str = "mpl", fold: int = 120):
    """Render a circuit for inspection.

    Args:
        circuit: QuantumCircuit to visualize.
        output: Draw style ("mpl", "text", "latex", etc.).
        fold: Line wrap width for text output.

    Returns:
        Rendered circuit object (e.g., matplotlib Figure for output="mpl").
    """
    return circuit.draw(output=output, fold=fold)


def draw_decomposed_circuit(circuit, output: str = "mpl", fold: int = 120, reps: int = 1):
    """Render a decomposed version of the circuit.

    Args:
        circuit: QuantumCircuit to visualize.
        output: Draw style ("mpl", "text", "latex", etc.).
        fold: Line wrap width for text output.
        reps: Decomposition depth passed to QuantumCircuit.decompose().

    Returns:
        Rendered circuit object after decomposition.
    """
    return circuit.decompose(reps=reps).draw(output=output, fold=fold)


__all__ = ["build_efficient_su2", "build_hva", "draw_circuit", "draw_decomposed_circuit"]
