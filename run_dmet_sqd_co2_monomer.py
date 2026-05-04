# -*- coding: utf-8 -*-
"""CO2-only monomer: geometry optimization + DMET(SQD) with active space [1,1]."""

import os
import sys
from datetime import datetime
import tempfile

import numpy as np
from pyscf import gto, scf, lib, geomopt

from tangelo import SecondQuantizedMolecule
from tangelo.problem_decomposition import DMETProblemDecomposition

enable_logging = True
basis_set = "3-21g"

# Coordinates from only_CO2_opt_DFT_B3LYP_321g.gjf
base_geom = np.array([
    [-0.23130303, 0.14312432, -0.04876380],  # C
    [-1.48970303, 0.14312432, -0.04876380],  # O
    [1.02709697, 0.14312432, -0.04876380],   # O
])
atom_labels = ["C", "O", "O"]


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


# Avoid Newton failures for tiny isolated systems.
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
    "tol": 1e-4,
}

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
        f"DMET_SQD_CO2_monomer_JOB{job_pid}_PID{pid}_{timestamp}.out",
    )
    sys.stdout = open(logfile, "w")
    sys.stderr = sys.stdout

print("Optimizing CO2-only geometry...")
co2_opt = optimize_geom(base_geom, atom_labels, charge=0, spin=0)

co2_xyz = os.path.join(geom_root, "CO2_only_opt.xyz")
with open(co2_xyz, "w", encoding="utf-8") as f:
    f.write(f"{len(atom_labels)}\n")
    f.write("CO2 optimized geometry\n")
    for i, label in enumerate(atom_labels):
        x, y, z = co2_opt[i]
        f.write(f"{label} {x:.10f} {y:.10f} {z:.10f}\n")

co2_mol = SecondQuantizedMolecule(geom_to_string(co2_opt, atom_labels), q=0, spin=0, basis=basis_set)
co2_options = {
    "molecule": co2_mol,
    "fragment_atoms": [[0, 1, 2]],
    "fragment_frozen_orbitals": [define_dmet_frag_as(1, 1)],
    "fragment_solvers": ["sqd"],
    "solvers_options": [sqd_options],
    "optimizer": fixed_mu_optimizer,
    "initial_chemical_potential": 0.0,
    "verbose": True,
}

co2_dmet = DMETProblemDecomposition(co2_options)
co2_dmet.build()
co2_energy = co2_dmet.simulate()

print(f"CO2 DMET energy (Hartree): {co2_energy:.10f}")
print(f"Optimized geometry saved to: {co2_xyz}")
