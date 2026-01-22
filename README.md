# qc-bench

Benchmark lab for 1D quantum systems on a grid (Infinite Square Well / Harmonic Oscillator / Double Well), comparing VQE/VQD ansatzes and the impact of shots + noise.

## Hamiltonian Variational Ansatz (HVA) in this project

This benchmark compares two VQE/VQD ansatz families:

* **`efficient_su2` (hardware-efficient):** generic layers of single-qubit rotations + entanglers.
* **`hva` (Hamiltonian Variational Ansatz):** physics-informed layers built from the Hamiltonian’s structure.

Goal:

> For simple 1D quantum systems (ISW / HO / Double Well), which ansatz gives the best first `k_states` energies for the least circuit cost, and how do shots + noise degrade performance?

---

## Quickstart

```bash
python -m venv qc-bench-venv
source qc-bench-venv/bin/activate
pip install -e .
qc-bench --help
qc-bench configs/isw.yaml

```

If `qc-bench` is not on your PATH, use:

```bash
python -m qcbench.runner --help
python -m qcbench.runner configs/isw.yaml

```

By default, results append to `results/runs.parquet`. Use `--no-append` to overwrite:

```bash
qc-bench configs/isw.yaml --no-append

```

Then open `notebooks/01_figures.ipynb` to plot from the results table.

---

## Results table: `results/runs.parquet`

Each run appends **one row** to `results/runs.parquet` with run settings, grid info, circuit cost, energies, and error metrics.

### Column reference (full schema)

#### A) Run identifiers / settings

* `potential`: string. Potential name (e.g., `isw`, `ho`, `doublewell`).
* `ansatz`: string. Ansatz family (e.g., `efficient_su2`, `hva`).
* `reps`: int. Ansatz repetition count / depth parameter.
* `entanglement`: string. Entanglement pattern (passed through to ansatz builder).
* `backend`: string. `"statevector"` or `"noisy"`.
* `noise_strength`: float or NaN. Noise parameter used for `"noisy"` runs.
* `shots`: int or NaN. Shot count used for `"noisy"` runs.
* `seed`: int. Random seed for reproducibility.
* `k_states`: int. Number of eigenstates requested (produces energy columns `0..k_states-1`).

#### B) Grid / physical constants (from the discretized Hamiltonian)

* `L`: float. Domain length.
* `N`: int. Number of grid points (physical Hilbert dimension).
* `dx`: float. Grid spacing.
* `n_qubits`: int. Smallest integer such that $2^{n_{\text{qubits}}} \ge N$.
* `hilbert_dim`: int. Embedded qubit dimension, defined as $\mathrm{hilbert_dim} = 2^{n_{\text{qubits}}}$.
* `x_min`, `x_max`: float. Min/max coordinates of the grid.
* `hbar`: float. The reduced Planck constant $\hbar$ used in the kinetic term.
* `m`: float. Mass $m$ used in the kinetic term.

#### C) Circuit / operator cost

* `depth`: int. Circuit depth after construction (and transpilation if applicable in your workflow).
* `two_qubit_count`: int. Count of two-qubit gates (CX/CZ/SWAP/iSWAP/ECR/RXX/RYY/RZZ).
* `num_parameters`: int. Number of trainable circuit parameters.
* `num_pauli_terms`: int. Number of Pauli terms in the qubit Hamiltonian operator.

#### D) Optimization / runtime metadata

* `runtime_sec`: float. Wall-clock runtime for the VQD call.
* `eval_count`: int or NaN. Optimizer evaluation count (if available from metadata).
* `success`: bool. Whether the optimizer reports success.

#### E) Energies (per-state columns)

For `i = 0..k_states-1` (ground state is `i=0`):

* `E{i}`: **VQD estimated energy** for state `i`. (This is what you are benchmarking.)
* `E_exact[i]`: Energies from exact diagonalization of the **padded qubit Hamiltonian**, which is a
  $$2^{n_{\text{qubits}}} \times 2^{n_{\text{qubits}}}$$
  matrix.
* `E_err{i}`: **VQD error vs padded reference**, defined as `abs(E{i} - E_exact{i})`.

Also always included (grid diagonalization):

* `E_grid[i]`: Energies from exact diagonalization of the **physical grid Hamiltonian**, which is an
  $$N \times N$$
  matrix.
* `E_pad_err{i}`: **Padding mismatch**, defined as `abs(E_exact{i} - E_grid{i})`.

Analytic (only when a closed-form exists and required params exist; e.g., ISW, HO with `omega`):

* `E_analytic{i}`: Analytic (continuous-space) textbook energy level.
* `E_analytic_err{i}`: Discretization error, defined as `abs(E_grid{i} - E_analytic{i})` (or `abs(E_exact{i} - E_analytic{i})` if grid energies were absent).

#### F) Gap error (energy spacing diagnostic)

For `i = 0..k_states-2`:

* `gap_err_{i}`: error in the energy gap between states `i` and `i+1`:

#### G) Padding leakage diagnostics (how “physical” the padded eigenvectors are)

* `padding_leakage_max`: max leakage across the lowest `k_states` padded eigenvectors.
* `padding_leakage_mean`: mean leakage across the lowest `k_states` padded eigenvectors.

* `leakage`: Probability mass “leaking” **outside the original physical $N$-dimensional subspace** (i.e., outside the first $N$ basis states after embedding into $2^{n_{\text{qubits}}}$).

---

## QC-Bench Metrics Reference (energies & errors)

QC-Bench tracks errors across three layers: **continuous physics**, **grid discretization**, and **quantum execution**.

### 1) `E_analytic{i}` — Theoretical (continuous) energy

* **Definition:** Closed-form energy in continuous space (no discretization).
* **Why it matters:** Physics “ground truth” (when it exists).
* **Example (Harmonic Oscillator):**
  $$E_n = \hbar\omega\left(n+\frac{1}{2}\right);;\Rightarrow;;E_0=\frac{1}{2}\hbar\omega.$$
  (So you need the parameter `omega` to define the analytic energies.)


### 2) `E_grid{i}` — Discretized grid energy

* **Definition:** Eigenvalue of the **physical**  grid Hamiltonian.
* **Why it matters:** Best possible answer your **grid model** can produce (even with a perfect quantum solver).

### 3) `E_exact{i}` — Padded qubit target energy

* **Definition:** Eigenvalue of the final **padded**  Hamiltonian used for qubits.
* **Why it matters:** The **actual mathematical target** that VQD is trying to match.

### 4) `E_pad_err{i}` — Padding error

* **Definition:** `abs(E_exact{i} - E_grid{i})`
* **Why it matters:** Detects whether padding/penalties are cleanly preserving the low-energy physical spectrum.

### 5) `E_analytic_err{i}` — Discretization error

* **Definition:** `abs(E_grid{i} - E_analytic{i})` (when analytic energies exist)
* **Why it matters:** Tells you whether your grid resolution (`dx`, `N`, domain size) is sufficient.

### 6) `E{i}` and `E_err{i}` — Quantum algorithm output and error

* **`E{i}` definition:** The VQD-estimated energy for state `i`.
* **`E_err{i}` definition:** `abs(E{i} - E_exact{i})`
* **Why it matters:** Captures optimization limits + sampling + noise + ansatz expressibility.

### 7) `gap_err_{i}` — Energy gap error

**Definition:**

$$
\texttt{gap\_err}_i
=
\left| (E_{i+1} - E_i) - (E^{\text{exact}}_{i+1} - E^{\text{exact}}_{i}) \right|
$$

* **Why it matters:** Excitation gaps are often more physically relevant than absolute energies.
