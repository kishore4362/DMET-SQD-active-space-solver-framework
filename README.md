# DMET–SQD Active-Space Solver Framework

A hybrid quantum–classical framework that integrates **Density Matrix Embedding Theory (DMET)** with **Sample-based Quantum Diagonalization (SQD)** for correlated electronic structure calculations on near-term quantum hardware.

This repository contains the code accompanying the paper:

> **"Beyond the Fragment Solver Bottleneck: DMET-Driven Quantum Diagonalization on a Superconducting Quantum Processor"**
> *(submitted to PRX Quantum)*

---

## Overview

The framework decomposes a large molecular system (iron-tetraphenylporphyrin + CO₂) into chemically meaningful fragments using DMET. The chemically active N–CO₂ binding fragment is solved using SQD on IBM quantum hardware, while all remaining fragments are solved classically with CCSD. A noise-aware chemical potential optimizer enforces self-consistency across fragments.

```
Full System (Fe-TPP·CO₂, 32 atoms, 142 electrons)
        │
        ▼  DMET fragmentation (type-C, 6 fragments)
┌───────────────────────────────────┐
│  F1–F5: CCSD (classical)          │
│  F6:    SQD  (quantum device)     │  ← N–CO₂ fragment, 8 qubits
└───────────────────────────────────┘
        │
        ▼  Self-consistent chemical potential μ
   E_DMET = E_core + Σ_I E_I
```

---

## Repository Structure

```
DMET-SQD-active-space-solver-framework/
│
├── tpp_CO2_relaxed_scan_1_.py          # Main script: full TPP·CO₂ complex relaxed scan
├── run_dmet_sqd_tpp_monomer.py         # TPP monomer reference calculation
├── run_dmet_sqd_co2_monomer.py         # CO₂ monomer reference calculation
│
├── tangelo/                            # Modified Tangelo source
│   ├── algorithms/
│   │   └── sqd_solver.py              # SQD solver with DMET active-space fix (Bug 1)
│   ├── problem_decomposition/
│   │   └── dmet/
│   │       ├── dmet_problem_decomposition.py   # DMET main loop
│   │       ├── fragment.py                     # SecondQuantizedDMETFragment
│   │       └── _helpers/
│   │           ├── dmet_orbitals.py            # Fragment Hamiltonian integrals
│   │           └── dmet_scf.py                 # Fragment SCF with chemical potential
│   └── toolboxes/
│       └── molecular_computation/
│           ├── frozen_orbitals.py              # Active-space orbital partitioning
│           └── rdms.py                         # RDM padding (frozen → full space)
│
├── relaxed_geoms/                      # Optimized geometries at each scan point
│   ├── type_C/3-21g/sqd/11_opt_geometries/    # TPP·CO₂ complex geometries
│   └── monomers/3-21g/sqd/11_opt_geometries/  # Monomer reference geometries
│
├── logs/                               # Calculation output logs
├── requirements.txt
└── README.md
```

---

## Requirements

- Python **3.12+**
- All dependencies listed in `requirements.txt`

Install all dependencies in a clean environment:

```bash
# Create and activate a new environment (recommended)
conda create -n dmet_sqd python=3.12
conda activate dmet_sqd

# Install dependencies
pip install -r requirements.txt
```

### Key dependencies

| Package | Purpose |
|---|---|
| `tangelo-gc` | DMET workflow framework |
| `pyscf` | Mean-field, integrals, geometry optimization |
| `geometric` | Constrained geometry optimization |
| `openfermion` | Active-space Hamiltonian construction |
| `qiskit` + `qiskit-ibm-runtime` | Quantum circuit execution |
| `qiskit-addon-sqd` | Sample-based quantum diagonalization |
| `ffsim` | LUCJ ansatz preparation |
| `qiskit-aer` | Local simulation backend |

---

## IBM Quantum Access

To run on **real quantum hardware** (IBM Kingston), you need an IBM Quantum account and API token.

**Never hardcode your token in the script.** Set it as an environment variable:

```bash
export IBM_QUANTUM_TOKEN="your_token_here"
export IBM_QUANTUM_INSTANCE="crn:v1:bluemix:public:quantum-computing:..."
```

Then in the script, replace the hardcoded token with:

```python
import os
service = QiskitRuntimeService(
    channel="ibm_quantum_platform",
    token=os.environ["IBM_QUANTUM_TOKEN"],
    instance=os.environ["IBM_QUANTUM_INSTANCE"],
)
```

Get your token at: https://quantum.ibm.com/account

---

## Running the Calculations

### 1. Full TPP·CO₂ Complex — Relaxed Potential Energy Scan

This is the main calculation. It runs a relaxed scan of the N–C bond distance
from **1.3 Å to 1.7 Å** in 5 steps (0.1 Å each), fully optimizing the geometry
at each point with the N–C distance constrained.

```bash
python tpp_CO2_relaxed_scan_1_.py
```

**What it does:**
- Optimizes the geometry at each N–C distance (RHF/3-21G + geomeTRIC)
- Partitions the molecule into 6 DMET fragments (type-C scheme)
- Solves fragments F1–F5 with CCSD
- Solves fragment F6 (N + CO₂) with SQD on quantum device
- Updates the chemical potential μ via noise-aware Brent optimizer
- Outputs DMET energies and optimized geometries

**Output files:**
```
relaxed_geoms/type_C/3-21g/sqd/11_opt_geometries/relaxed_geom_X.XX_A.xyz
tpp_co2_hardware_relaxed_scan_energies_new.csv
logs/type_C/3-21g/sqd/11_opt_geometries/DMET_SQD_TPP_CO2_*.out
```

---

### 2. TPP Monomer Reference Calculation

Calculates the isolated iron-tetraphenylporphyrin (TPP) energy as a reference
for computing the CO₂ binding energy.

```bash
python run_dmet_sqd_tpp_monomer.py
```

**Output files:**
```
relaxed_geoms/monomers/3-21g/sqd/11_opt_geometries/TPP_only_optimized.xyz
logs/monomers/3-21g/sqd/11_opt_geometries/DMET_SQD_TPP_monomer_*.out
```

---

### 3. CO₂ Monomer Reference Calculation

Calculates the isolated CO₂ energy as a reference.

```bash
python run_dmet_sqd_co2_monomer.py
```

**Output files:**
```
relaxed_geoms/monomers/3-21g/sqd/11_opt_geometries/CO2_only_opt.xyz
logs/monomers/3-21g/sqd/11_opt_geometries/DMET_SQD_CO2_monomer_*.out
```

---

### 4. Binding Energy

Once all three calculations are complete, the CO₂ binding energy at each
scan point is:

```
ΔE_bind(d) = E_DMET[TPP·CO₂](d) − E_DMET[TPP] − E_DMET[CO₂]
```

---

## Switching Between Simulator and Hardware

In `tpp_CO2_relaxed_scan_1_.py`, the `sqd_options` dictionary controls the backend.

**Simulator (default, no IBM account needed):**
```python
sqd_options = {
    "backend": "aer_simulator",
    "use_simulator": True,
    "sim_method": "matrix_product_state",
    "sampling_ansatz": "hf",
    ...
}
```

**Hardware (IBM Kingston):**
```python
sqd_options = {
    "service": service,           # your QiskitRuntimeService object
    "backend": "ibm_kingston",
    "use_simulator": False,
    "sampling_ansatz": "lucj",
    "lucj_parameter_source": "ccsd",
    "lucj_n_reps": 1,
    "dynamical_decoupling": {"enable": True, "sequence_type": "XpXm"},
    "gate_twirling": {"enable_gates": True, "num_randomizations": 8},
    ...
}
```

---

## Key Parameters

### Molecular System

| Parameter | Value |
|---|---|
| System | Fe-TPP · CO₂, charge = −1, singlet |
| Total electrons | 142 |
| Basis set | 3-21G |
| Scan range | 1.3 – 1.7 Å (5 points) |
| DMET fragmentation | Type-C, 6 fragments |

### Active Space (Fragment F6: N–CO₂)

| Parameter | Value |
|---|---|
| Fragment atoms | N(4), C(29), O(30), O(31) |
| Active-space window | HOMO−1 to LUMO+1 |
| Active spatial orbitals | 4 |
| Active electrons | 4 (α=2, β=2) |
| **Qubits (Jordan–Wigner)** | **8** |

### SQD Parameters

| Parameter | Value |
|---|---|
| Shots | 200,000 |
| Batches | 16 |
| Samples per batch | 1,000 |
| Max SQD iterations | 14 |
| Convergence tolerance | 1×10⁻⁴ |

### Optimizer Parameters

| Parameter | Value |
|---|---|
| Residual tolerance δᵣ | 0.03 |
| Bracket radius ρ₀ | 0.05 |
| Bracket growth α | 1.5 |
| Brent tolerance δ_μ | 1×10⁻⁸ |

---

## The DMET–SQD Active-Space Interface (Key Contribution)

The central technical contribution is the correct construction of the active-space
Hamiltonian passed to SQD. The frozen-core Coulomb–exchange correction:

```
h_pq^AS = h̃_pq^I + Σ_{i ∈ frozen_occ} [2·g_pqii^I − g_piiq^I]
```

is implemented in `tangelo/algorithms/sqd_solver.py` using:

```python
from pyscf import ao2mo
from openfermion.ops.representations import get_active_space_integrals

h1 = mo_coeff.T @ mol.mean_field.get_hcore() @ mo_coeff
h2 = ao2mo.kernel(mol.mean_field._eri, mo_coeff)
h2 = ao2mo.restore(1, h2, n_mos)
h2 = h2.transpose(0, 2, 3, 1)   # PySCF → OpenFermion PQRS convention

frozen_occ  = getattr(mol, "frozen_occupied", None) or []
active_mos  = getattr(mol, "active_mos", list(range(n_mos)))
_, hcore, eri = get_active_space_integrals(h1, h2, frozen_occ, active_mos)
eri = eri.transpose(0, 3, 1, 2)  # back to chemist (ij|kl) for solve_fermion
```

Without this correction, the DMET self-consistency loop diverges on hardware.
See Section II of the paper for the full mathematical derivation.

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{dmet_sqd_2025,
  title   = {Beyond the Fragment Solver Bottleneck: DMET-Driven Quantum
             Diagonalization on a Superconducting Quantum Processor},
  journal = {PRX Quantum},
  year    = {2025},
  note    = {submitted}
}
```

Also cite the core packages this work builds on:

- **Tangelo:** Choïsy et al., arXiv:2206.12424 (2023)
- **SQD:** Robledo-Moreno et al., arXiv:2405.05068 (2024)
- **PySCF:** Sun et al., *J. Chem. Phys.* 153, 024109 (2020)
- **Qiskit:** Qiskit contributors, doi:10.5281/zenodo.2573505 (2024)

---

## Acknowledgements

The authors acknowledge the use of the
[Tangelo](https://github.com/goodchemistryco/Tangelo) open-source package
for all quantum chemistry workflow implementations, and IBM Quantum for
access to the Kingston superconducting quantum processor.

---

## License

This project is licensed under the **Apache License 2.0** —
see the [LICENSE](LICENSE) file for details.

---

## Contact

For questions about the code or methodology, please open a GitHub issue or
contact the corresponding author.
