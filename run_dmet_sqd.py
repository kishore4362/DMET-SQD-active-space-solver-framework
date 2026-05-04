# -*- coding: utf-8 -*-
"""Simple DMET + SQD test on a small C2H6-like molecule."""

import os
import sys
import tempfile
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from pyscf import gto, scf, lib, geomopt
from qiskit_ibm_runtime import QiskitRuntimeService

from tangelo import SecondQuantizedMolecule
from tangelo.problem_decomposition import DMETProblemDecomposition

# ========== 0. Logging Configuration ==========
enable_logging = False  # Set to False to disable saving output to a file

if enable_logging:
    logdir = os.path.join("logs", "test_runs")
    os.makedirs(logdir, exist_ok=True)
    pid = os.getpid()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logfile = os.path.join(logdir, f"run_dmet_sqd_PID{pid}_{timestamp}.out")

    # Redirect stdout and stderr to the log file
    log_fh = open(logfile, "w", encoding="utf-8")
    sys.stdout = log_fh
    sys.stderr = log_fh
    print(f"Logging enabled. Output redirected to: {logfile}")


# ========== 1. Base geometry ==========
base_geom = np.array([
    [-0.000, -0.000, 0.772],   # C0
    [-0.512, 0.887, 1.162],    # H1
    [-0.512, -0.887, 1.162],   # H2
    [1.024, -0.000, 1.162],    # H3
    [0.000, -0.000, -0.772],   # C4
    [0.512, 0.887, -1.162],    # H5
    [0.512, -0.887, -1.162],   # H6
    [-1.024, -0.000, -1.162],  # H7
])

atom_labels = ["C", "H", "H", "H", "C", "H", "H", "H"]

# Move the second CH3 group relative to the first carbon.
idx_C0 = 0
idx_CH3 = [4, 5, 6, 7]

vec_CC = base_geom[idx_CH3[0]] - base_geom[idx_C0]
dist_CC = np.linalg.norm(vec_CC)
unit_vec = vec_CC / dist_CC

print(f"Initial C-C distance: {dist_CC:.3f} Å")


# ========== 2. Function to move CH3 group ==========
def move_ch3(distance):
    """Return a new geometry with the second CH3 translated to the target C-C distance."""
    shift = (distance - dist_CC) * unit_vec
    new_geom = base_geom.copy()
    new_geom[idx_CH3] += shift
    return new_geom


def adjust_group_distance(geom, idx_a, idx_b, moving_group, target_dist):
    """Translate the moving group so the selected bond length matches target_dist."""
    vec = geom[idx_b] - geom[idx_a]
    current_dist = np.linalg.norm(vec)
    direction = vec / current_dist
    shift = (target_dist - current_dist) * direction
    new_geom = geom.copy()
    new_geom[moving_group] += shift
    return new_geom


def geom_to_string(coords, labels):
    return "\n".join(
        f"{labels[i]} {coords[i, 0]} {coords[i, 1]} {coords[i, 2]}" for i in range(len(labels))
    )


def write_constraints_file(path, i, j, dist_ang):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("$freeze\n")
        fh.write("$end\n\n")
        fh.write("$set\n")
        fh.write(f"distance {i + 1} {j + 1} {dist_ang:.8f}\n")
        fh.write("$end\n")


# ========== 3. DMET fragment active-space helpers ==========
def define_dmet_frag_as(homo_minus_m=0, lumo_plus_n=0, occ_thresh=0.5):
    def callable_for_dmet_object(info_fragment):
        mf_fragment, _, _, _, _, _, _ = info_fragment
        mo_occ = list(mf_fragment.mo_occ)
        n_lumo = next((i for i, occ in enumerate(mo_occ) if occ < occ_thresh), len(mo_occ) - 1)
        n_homo = max(n_lumo - 1, 0)
        kept = [n for n in range(n_homo - homo_minus_m, n_lumo + lumo_plus_n + 1)]
        frozen_orbitals = [n for n in range(len(mo_occ)) if n not in kept]
        print(f"Fragment: n_homo={n_homo}, n_lumo={n_lumo}, kept={kept}")
        return frozen_orbitals

    return callable_for_dmet_object


def keep_all_orbitals(_info_fragment):
    return []


# ========== 4. Solver options ==========
use_hardware = True

service = QiskitRuntimeService(
    channel="ibm_quantum_platform",
    token="RpTh1KcyZaKYfjDBPRiMfSjhuASU7asjrqu_opW1NLPI",
    instance="crn:v1:bluemix:public:quantum-computing:us-east:a/4ac976a1185b40be98b2de321d0f2ed4:ce3bdfc3-bee8-4db4-bcc3-819c4ee4e636::",
)

sqd_options = {
    "backend": "ibm_kingston" if use_hardware else "aer_simulator",
    "sim_method": "matrix_product_state",
    "shots": 20000,
    "n_batches": 6,
    "samples_per_batch": 300,
    "diagnostics": True,
    "tol": 1e-4,
    "energy_tol": 1e-3,
    "occupancies_tol": 1e-3,
    "symmetrize_spin": True,
}

noisy_optimizer_options = {
    "residual_evaluations": 7,
    "residual_tol": 1e-2,
    "bracket_radius": 0.05,
    "bracket_growth": 1.5,
    "max_bracket_steps": 8,
    "mu_min": -0.5,
    "mu_max": 0.5,
    "brent_tol": 1e-4,
    "damped_step": 0.1,
    "max_mu_step": 0.1,
    "max_damped_steps": 10,
}

if use_hardware:
    sqd_options["service"] = service
ccsd_options = {}


# ========== 5. Scan settings ==========
#distances = np.linspace(1.1, 1.9, 9)
distances =[1.5]
energies = []
opt_geoms = []


# ========== 6. Main DMET loop ==========
geom_start = base_geom.copy()

for d in distances:
    print(f"\n=== Running simple DMET + SQD for C-C distance = {d:.2f} Å ===")
    geom_start = adjust_group_distance(geom_start, idx_C0, idx_CH3[0], idx_CH3, d)

    geom_str = geom_to_string(geom_start, atom_labels)
    mol = gto.M(atom=geom_str, basis="3-21g", charge=0, spin=0, unit="Angstrom")
    mf = scf.RHF(mol)

    with tempfile.TemporaryDirectory() as tmpdir:
        cfile = os.path.join(tmpdir, "constraints.txt")
        write_constraints_file(cfile, idx_C0, idx_CH3[0], d)
        try:
            opt_mol = geomopt.optimize(mf, constraints=cfile, maxsteps=100)
        except ImportError as exc:
            raise RuntimeError(
                "PySCF geometry optimization requires the geomeTRIC package. "
                "Install it with: pip install geometric"
            ) from exc

    geom_opt = opt_mol.atom_coords() * lib.param.BOHR
    opt_geoms.append(geom_opt)
    geom_start = geom_opt.copy()

    mol = SecondQuantizedMolecule(geom_to_string(geom_opt, atom_labels), q=0, spin=0, basis="3-21g")

    try:
        options_dmet = {
            "molecule": mol,
            "fragment_atoms": [[0, 1, 2, 3], [4, 5, 6, 7]],
            "fragment_frozen_orbitals": [keep_all_orbitals, define_dmet_frag_as(1,1)],
            "fragment_solvers": ["ccsd", "ccsd"],
            "solvers_options": [ccsd_options, ccsd_options],
            "noisy_optimizer_options": noisy_optimizer_options,
            "verbose": True,
        }

        dmet_calc = DMETProblemDecomposition(options_dmet)
        dmet_calc.build()
        energy = dmet_calc.simulate()
        print(f"Simple DMET + SQD energy (Hartree): {energy:.6f}")
        print(f"Converged DMET chemical potential: {dmet_calc.chemical_potential:.10f}")
        print(f"Final DMET electron residual: {dmet_calc.electron_residual:.10f}")
        energies.append(energy)
    except Exception as exc:
        print(f"Error at distance {d:.2f}: {exc}")
        energies.append(np.nan)


# ========== 7. Plot ==========
plt.figure(figsize=(7, 5))
plt.plot(distances, energies, "o-", lw=2)
plt.xlabel("C-C bond distance (Å)", fontsize=12)
plt.ylabel("Total DMET Energy (Hartree)", fontsize=12)
plt.title("Simple C2H6-like Test: DMET + SQD", fontsize=13)
plt.grid(True)
plt.tight_layout()
plt.show()
