# -*- coding: utf-8 -*-
"""Relaxed monomer calculations for binding energy: TPP-only and CO2-only."""

import sys
import os
from datetime import datetime
import tempfile
import numpy as np

from pyscf import gto, scf, lib, geomopt

from tangelo import SecondQuantizedMolecule
from tangelo.problem_decomposition import DMETProblemDecomposition

enable_logging = True
basis_set = "3-21g"
run_tpp = False
run_co2 = True

# Output folders
geom_root = os.path.join("relaxed_geoms", "monomers", basis_set)
os.makedirs(geom_root, exist_ok=True)

if enable_logging:
    logdir = os.path.join("logs", "monomers", basis_set)
    os.makedirs(logdir, exist_ok=True)
    pid = os.getpid()
    job_pid = os.environ.get("JOB_PID") or pid
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logfile = os.path.join(
        logdir,
        f"DMET_SQD_TPP_CO2_monomers_JOB{job_pid}_PID{pid}_{timestamp}.out",
    )
    sys.stdout = open(logfile, "w")
    sys.stderr = sys.stdout

# ========== Geometry ==========
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

atom_labels = [
    "C","C","C","C","N","H","H","C","H","H","C","C","C","H","N","H","H","N",
    "C","H","H","C","C","C","N","H","H","H","N","C","O","O"
]

idx_CO2 = [29, 30, 31]

def geom_to_string(geom, labels):
    return "\n".join(
        [f"{labels[i]} {geom[i,0]} {geom[i,1]} {geom[i,2]}" for i in range(len(labels))]
    )

def optimize_geom(geom, labels, charge, spin):
    geom_str = geom_to_string(geom, labels)
    mol = gto.M(atom=geom_str, basis=basis_set, charge=charge, spin=spin, unit="Angstrom")
    mf = scf.RHF(mol)
    with tempfile.TemporaryDirectory() as tmpdir:
        opt_mol = geomopt.optimize(mf, maxsteps=100, tmpdir=tmpdir)
    return opt_mol.atom_coords() * lib.param.BOHR

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

def fixed_mu_optimizer(_func, mu0):
    return mu0

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

#vqe_options = {"qubit_mapping": "jw", "ansatz": Ansatze.UCCSD}
ccsd_options = {}

tpp_energy = None
if run_tpp:
    # ========== TPP-only ==========
    tpp_mask = [i for i in range(len(atom_labels)) if i not in idx_CO2]
    tpp_geom = base_geom[tpp_mask]
    tpp_labels = [atom_labels[i] for i in tpp_mask]

    print("Optimizing TPP-only geometry...")
    tpp_opt = optimize_geom(tpp_geom, tpp_labels, charge=-1, spin=0)
    tpp_xyz = os.path.join(geom_root, "TPP_opt.xyz")
    with open(tpp_xyz, "w", encoding="utf-8") as f:
        f.write(f"{len(tpp_labels)}\n")
        f.write("TPP optimized geometry\n")
        for i, label in enumerate(tpp_labels):
            x, y, z = tpp_opt[i]
            f.write(f"{label} {x:.10f} {y:.10f} {z:.10f}\n")

    tpp_fragment_atoms = [
        [0, 1, 2, 3, 4, 5, 6],
        [7, 8, 9],
        [10, 11, 12, 13, 14, 15, 16, 17],
        [18, 19, 20],
        [21, 22, 23, 24, 25, 26, 27, 28],
    ]

    tpp_mol = SecondQuantizedMolecule(geom_to_string(tpp_opt, tpp_labels), q=-1, spin=0, basis=basis_set)
    tpp_options = {
        "molecule": tpp_mol,
        "fragment_atoms": tpp_fragment_atoms,
        "fragment_frozen_orbitals": [keep_all_orbitals,
                                     keep_all_orbitals,
                                     keep_all_orbitals,
                                     keep_all_orbitals,
                                     keep_all_orbitals],
        "fragment_solvers": ["ccsd","ccsd","ccsd","ccsd","ccsd"],
        "solvers_options": [ccsd_options,ccsd_options,ccsd_options,ccsd_options,ccsd_options],
        "verbose": True,
    }
    tpp_dmet = DMETProblemDecomposition(tpp_options)
    tpp_dmet.build()
    tpp_energy = tpp_dmet.simulate()
    print(f"TPP DMET energy (Hartree): {tpp_energy:.6f}")

co2_energy = None
if run_co2:
    # ========== CO2-only ==========
    co2_geom = base_geom[idx_CO2]
    co2_labels = [atom_labels[i] for i in idx_CO2]

    print("Optimizing CO2-only geometry...")
    co2_opt = optimize_geom(co2_geom, co2_labels, charge=0, spin=0)
    co2_xyz = os.path.join(geom_root, "CO2_opt.xyz")
    with open(co2_xyz, "w", encoding="utf-8") as f:
        f.write(f"{len(co2_labels)}\n")
        f.write("CO2 optimized geometry\n")
        for i, label in enumerate(co2_labels):
            x, y, z = co2_opt[i]
            f.write(f"{label} {x:.10f} {y:.10f} {z:.10f}\n")

    co2_mol = SecondQuantizedMolecule(geom_to_string(co2_opt, co2_labels), q=0, spin=0, basis=basis_set)
    co2_options = {
        "molecule": co2_mol,
        "fragment_atoms": [[0, 1, 2]],
        "fragment_frozen_orbitals": [define_dmet_frag_as(1,1)],
        "fragment_solvers": ["sqd"],
        "solvers_options": [sqd_options],
        "optimizer": fixed_mu_optimizer,
        "initial_chemical_potential": 0.0,
        "verbose": True,
    }
    co2_dmet = DMETProblemDecomposition(co2_options)
    co2_dmet.build()
    co2_energy = co2_dmet.simulate()
    print(f"CO2 DMET energy (Hartree): {co2_energy:.6f}")

# ========== Summary ==========
print("\nMonomer energies (Hartree):")
if tpp_energy is None:
    print("TPP: skipped")
else:
    print(f"TPP: {tpp_energy:.10f}")
if co2_energy is None:
    print("CO2: skipped")
else:
    print(f"CO2: {co2_energy:.10f}")
