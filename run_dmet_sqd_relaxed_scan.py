# -*- coding: utf-8 -*-
"""Relaxed scan: optimize geometry with PySCF at each N–C distance, then run DMET."""

import sys
import os
from datetime import datetime

enable_logging = True

import tempfile
import numpy as np
import matplotlib.pyplot as plt
import json

from pyscf import gto, scf, lib, geomopt

from tangelo import SecondQuantizedMolecule
from tangelo.problem_decomposition import DMETProblemDecomposition
from tangelo.algorithms import BuiltInAnsatze as Ansatze

# ========== 1. Base geometry ==========
base_geom = np.array([
    [-1.168,   1.564,  -0.449],   #C0
    [-0.609,   2.110,  -1.594],   #C1
    [0.811,   2.074,  -1.445],    #C2
    [1.097,   1.487,  -0.222],    #C3
    [-0.121,   1.180,   0.393],   #N4
    [-1.164,   2.478,  -2.446],   #H
    [1.546,   2.435,  -2.152],    #H
    [2.447,   1.240,   0.407],    #C
    [2.345,   1.262,   1.497],    #H
    [3.108,   2.070,   0.125],    #H
    [3.117,  -0.056,   0.008],    #C
    [4.257,  -0.237,  -0.821],    #C
    [2.742,  -1.340,   0.381],    #C
    [4.847,   0.520,  -1.316],    #H
    [3.632,  -2.194,  -0.213],    #N
    [1.915,  -1.630,   1.012],    #H
    [3.658,  -3.197,  -0.164],    #H
    [4.588,  -1.539,  -0.968],    #N
    [-2.616,   1.295,  -0.134],   #C
    [-3.228,   1.913,  -0.804],   #H
    [-2.812,   1.614,   0.896],   #H
    [-3.042,  -0.149,  -0.292],   #C
    [-2.407,  -1.287,   0.194],   #C
    [-4.201,  -0.641,  -0.952],   #C
    [-3.180,  -2.356,  -0.181],   #N
    [-1.489,  -1.400,   0.750],   #H
    [-4.961,  -0.073,  -1.467],   #H
    [-3.017,  -3.330,   0.002],   #H
    [-4.301,  -1.987,  -0.894],   #N
    [-0.300,   0.585,   1.770],   #C
    [0.506,  -0.365,  2.045],     #O
    [-1.237,   1.082,   2.454],   #O
])

atom_labels = ["C","C","C","C","N","H","H","C","H","H","C","C","C","H","N","H","H","N","C","H","H","C","C","C","N","H","H","H","N","C","O","O"]

# N and CO2 indices (0-based)
idx_N = 4
idx_C = 29
idx_CO2 = [29, 30, 31]

# ========== 2. Helpers ==========

def geom_to_string(geom, labels):
    return "\n".join([f"{labels[i]} {geom[i,0]} {geom[i,1]} {geom[i,2]}" for i in range(len(labels))])

def adjust_group_distance(geom, idx_a, idx_b, group_indices, target_dist):
    """Translate group_indices so that distance between idx_a and idx_b equals target_dist."""
    g = geom.copy()
    vec = g[idx_b] - g[idx_a]
    dist = np.linalg.norm(vec)
    if dist == 0:
        raise ValueError("Zero distance between constrained atoms.")
    unit = vec / dist
    shift = (target_dist - dist) * unit
    g[group_indices] += shift
    return g

def write_constraints_file(path, i, j, dist_ang):
    # geomeTRIC uses 1-based atom indexing
    i1 = i + 1
    j1 = j + 1
    with open(path, "w", encoding="utf-8") as f:
        f.write("$set\n")
        f.write(f"distance {i1} {j1} {dist_ang:.8f}\n")

# ========== 3. Active space helpers ==========

def define_dmet_frag_as(homo_minus_m=0, lumo_plus_n=0, occ_thresh=0.5):
    def callable_for_dmet_object(info_fragment):
        mf_fragment, _, _, _, _, _, _ = info_fragment
        mo_occ = list(mf_fragment.mo_occ)
        n_lumo = next((i for i, o in enumerate(mo_occ) if o < occ_thresh), len(mo_occ) - 1)
        n_homo = max(n_lumo - 1, 0)
        kept = [n for n in range(n_homo - homo_minus_m, n_lumo + lumo_plus_n + 1)]
        frozen_orbitals = [n for n in range(len(mo_occ)) if n not in kept]
        print(f"Fragment: n_homo={n_homo}, n_lumo={n_lumo}, kept={kept}")
        return frozen_orbitals
    return callable_for_dmet_object

def keep_all_orbitals(_info_fragment):
    return []

# Fix chemical potential for tiny/isolated systems (avoid Newton failures)
def fixed_mu_optimizer(_func, mu0):
    return mu0

# ========== 4. Solver options ==========

sqd_options = {
    "backend": "aer_simulator",
    "sim_method": "matrix_product_state",
    "shots": 100000,
    "n_batches": 8,
    "samples_per_batch": 500,
    "max_iterations": 14,
    "diagnostics": True,
    "tol": 1e-4
}

vqe_options = {"qubit_mapping": "jw", "ansatz": Ansatze.UCCSD}
ccsd_options = {}

# ========== 5. Scan settings ==========

distances = np.linspace(0.7, 2.7, 21)
#distances = [2.3]

# ========== 5b. Basis set and output folders ==========
basis_set = "3-21g"

# Fragmentation type: "type_C" or "type_D"
fragmentation_type = "type_C"

# Compute frozen-monomer binding energy at each distance
compute_binding_energy = True

# Output folders based on basis set / solver / active space
frag6_solver = "sqd"
frag6_homo_minus_m = 1
frag6_lumo_plus_n = 1
active_space_folder = f"{frag6_homo_minus_m}{frag6_lumo_plus_n}_opt_geometries"
solver_geom_dir = os.path.join("relaxed_geoms", fragmentation_type, basis_set, frag6_solver, active_space_folder)
os.makedirs(solver_geom_dir, exist_ok=True)

# Log files go under logs/{frag_type}/{basis_set}/{solver}/{active_space}
if enable_logging:
    logdir = os.path.join("logs", fragmentation_type, basis_set, frag6_solver, active_space_folder)
    os.makedirs(logdir, exist_ok=True)
    pid = os.getpid()
    job_pid = os.environ.get("JOB_PID") or pid
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    logfile = os.path.join(
        logdir,
        f"DMET_SQD_TPP_CO2_JOB{job_pid}_PID{pid}_{timestamp}.out",
    )
    sys.stdout = open(logfile, "w")
    sys.stderr = sys.stdout

solver_options_map = {
    "sqd": sqd_options,
    "ccsd": ccsd_options,
    "vqe": vqe_options,
}
frag6_solver_options = solver_options_map.get(frag6_solver, sqd_options)

fragment_atoms_map = {
    "type_C": [
        [0, 1, 2, 3, 5, 6],
        [7, 8, 9],
        [10, 11, 12, 13, 14, 15, 16, 17],
        [18, 19, 20],
        [21, 22, 23, 24, 25, 26, 27, 28],
        [4, 29, 30, 31],
    ],
    "type_D": [
        [0, 1, 2, 3, 5, 6],
        [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
        [18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28],
        [4, 29, 30, 31],
    ],
}

fragment_frozen_orbitals_map = {
    "type_C": [
        keep_all_orbitals,
        keep_all_orbitals,
        keep_all_orbitals,
        keep_all_orbitals,
        keep_all_orbitals,
        define_dmet_frag_as(frag6_homo_minus_m, frag6_lumo_plus_n),
    ],
    "type_D": [
        keep_all_orbitals,
        keep_all_orbitals,
        keep_all_orbitals,
        define_dmet_frag_as(frag6_homo_minus_m, frag6_lumo_plus_n),
    ],
}

fragment_solvers_map = {
    "type_C": ["ccsd", "ccsd", "ccsd", "ccsd", "ccsd", frag6_solver],
    "type_D": ["ccsd", "ccsd", "ccsd", frag6_solver],
}

solvers_options_map = {
    "type_C": [ccsd_options, ccsd_options, ccsd_options, ccsd_options, ccsd_options, frag6_solver_options],
    "type_D": [ccsd_options, ccsd_options, ccsd_options, frag6_solver_options],
}

# Frozen-monomer fragmentations (derived from complex geometry)
idx_CO2 = [29, 30, 31]
tpp_fragment_atoms_frozen = [
    [0, 1, 2, 3, 5, 6],
    [7, 8, 9],
    [10, 11, 12, 13, 14, 15, 16, 17],
    [18, 19, 20],
    [21, 22, 23, 24, 25, 26, 27, 28],
    [4]
]
# ========== 6. Run relaxed scan ==========

energies = []
opt_geoms = []
csv_rows = []
prev_mu = None

geom_start = base_geom.copy()

for d in distances:
    print(f"\n=== Relaxed scan: N–C distance = {d:.2f} Å ===")

    # Start from previous optimized geometry, then enforce target distance
    geom_start = adjust_group_distance(geom_start, idx_N, idx_C, idx_CO2, d)

    # Build PySCF molecule (HF/3-21g, q=-1, spin=0)
    geom_str = geom_to_string(geom_start, atom_labels)
    mol = gto.M(atom=geom_str, basis=basis_set, charge=-1, spin=0, unit="Angstrom")
    mf = scf.RHF(mol)

    # Constrained optimization with geomeTRIC
    with tempfile.TemporaryDirectory() as tmpdir:
        cfile = os.path.join(tmpdir, "constraints.txt")
        write_constraints_file(cfile, idx_N, idx_C, d)
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

    # Save optimized geometry to XYZ (overwrites on rerun)
    xyz_path = os.path.join(solver_geom_dir, f"relaxed_geom_{d:.2f}_A.xyz")
    with open(xyz_path, "w", encoding="utf-8") as xyz:
        xyz.write(f"{len(atom_labels)}\n")
        xyz.write(f"N-C distance = {d:.8f} Angstrom\n")
        for i, label in enumerate(atom_labels):
            x, y, z = geom_opt[i]
            xyz.write(f"{label} {x:.10f} {y:.10f} {z:.10f}\n")

    # Run DMET on optimized geometry
    geom_opt_str = geom_to_string(geom_opt, atom_labels)
    mol_dmet = SecondQuantizedMolecule(geom_opt_str, q=-1, spin=0, basis=basis_set)

    options_dmet = {
        "molecule": mol_dmet,
        "fragment_atoms": fragment_atoms_map[fragmentation_type],
        "fragment_frozen_orbitals": fragment_frozen_orbitals_map[fragmentation_type],
        "fragment_solvers": fragment_solvers_map[fragmentation_type],
        "solvers_options": solvers_options_map[fragmentation_type],
        "verbose": True,
    }
    if prev_mu is not None:
        options_dmet["initial_chemical_potential"] = prev_mu

    dmet_calc = DMETProblemDecomposition(options_dmet)
    dmet_calc.build()
    energy = dmet_calc.simulate()
    prev_mu = dmet_calc.chemical_potential

    print(f"DMET energy (Hartree): {energy:.6f}")
    energies.append(energy)

    tpp_energy = None
    co2_energy = None
    bind_energy = None

    if compute_binding_energy:
        # TPP-only frozen monomer from complex geometry
        tpp_mask = [i for i in range(len(atom_labels)) if i not in idx_CO2]
        tpp_geom = geom_opt[tpp_mask]
        tpp_labels = [atom_labels[i] for i in tpp_mask]
        tpp_mol = SecondQuantizedMolecule(geom_to_string(tpp_geom, tpp_labels), q=-1, spin=0, basis=basis_set)
        tpp_options = {
            "molecule": tpp_mol,
            "fragment_atoms": tpp_fragment_atoms_frozen,
            "fragment_frozen_orbitals": [
                keep_all_orbitals,
                keep_all_orbitals,
                keep_all_orbitals,
                keep_all_orbitals,
                keep_all_orbitals,
                define_dmet_frag_as(frag6_homo_minus_m, frag6_lumo_plus_n),
            ],
            "fragment_solvers": ["ccsd", "ccsd", "ccsd", "ccsd", "ccsd","sqd"],
            "solvers_options": [ccsd_options, ccsd_options, ccsd_options, ccsd_options, ccsd_options, sqd_options],
            "verbose": True,
        }
        tpp_dmet = DMETProblemDecomposition(tpp_options)
        tpp_dmet.build()
        tpp_energy = tpp_dmet.simulate()
        print(f"TPP (frozen) DMET energy (Hartree): {tpp_energy:.6f}")

        # CO2-only frozen monomer from complex geometry
        co2_geom = geom_opt[idx_CO2]
        co2_labels = [atom_labels[i] for i in idx_CO2]
        co2_mol = SecondQuantizedMolecule(geom_to_string(co2_geom, co2_labels), q=0, spin=0, basis=basis_set)
        co2_options = {
            "molecule": co2_mol,
            "fragment_atoms": [[0, 1, 2]],
            "fragment_frozen_orbitals": [define_dmet_frag_as(frag6_homo_minus_m, frag6_lumo_plus_n)],
            "fragment_solvers": [frag6_solver],
            "solvers_options": [frag6_solver_options],
            "optimizer": fixed_mu_optimizer,
            "initial_chemical_potential": 0.0,
            "verbose": True,
        }
        co2_dmet = DMETProblemDecomposition(co2_options)
        co2_dmet.build()
        co2_energy = co2_dmet.simulate()
        print(f"CO2 (frozen) DMET energy (Hartree): {co2_energy:.6f}")

        bind_energy = energy - (tpp_energy + co2_energy)
        print(f"Binding energy (Hartree): {bind_energy:.6f}")

    csv_rows.append((d, energy, tpp_energy, co2_energy, bind_energy, xyz_path))

# ========== 7. Plot ==========

plt.figure(figsize=(7, 5))
plt.plot(distances, energies, "o-", lw=2)
plt.xlabel("N–C bond distance (Å)", fontsize=12)
plt.ylabel("Total DMET Energy (Hartree)", fontsize=12)
plt.title("Relaxed Scan: TPP–CO₂ (DMET)", fontsize=13)
plt.grid(True)
plt.tight_layout()
plt.show()

# ========== 8. Save CSV ==========
csv_path = "relaxed_scan_energies.csv"
with open(csv_path, "w", encoding="utf-8") as f:
    f.write("distance_angstrom,energy_hartree,tpp_energy_hartree,co2_energy_hartree,bind_energy_hartree,xyz_file\n")
    for d, e, tpp_e, co2_e, bind_e, xyz in csv_rows:
        tpp_s = "" if tpp_e is None else f"{tpp_e:.10f}"
        co2_s = "" if co2_e is None else f"{co2_e:.10f}"
        bind_s = "" if bind_e is None else f"{bind_e:.10f}"
        f.write(f"{d:.8f},{e:.10f},{tpp_s},{co2_s},{bind_s},{xyz}\n")
