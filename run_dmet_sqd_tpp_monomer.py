# -*- coding: utf-8 -*-
"""TPP-only DMET monomer run using coordinates from only_tpp_opt_DFT_B3LYP_321g.gjf."""

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

# Geometry from only_tpp_opt_DFT_B3LYP_321g.gjf (charge=-1, spin=0)
base_geom = np.array([
    [0.94983041, 0.51954626, -0.01740200],
    [2.35158041, 0.51954626, -0.01740200],
    [2.77557841, 1.89034226, -0.01740200],
    [1.61852141, 2.68179526, -0.01750600],
    [0.51010341, 1.84005426, -0.01741100],
    [2.99595341, -0.35315974, -0.01737700],
    [3.80014141, 2.24683426, -0.01742700],
    [1.46452307, 4.21407607, -0.01771384],
    [0.92689118, 4.51862424, 0.85584347],
    [2.43260577, 4.66983463, -0.01766987],
    [0.69099750, 4.65208300, -1.27525435],
    [1.21839391, 4.75556604, -2.60566792],
    [-0.65922332, 5.02428422, -1.33482556],
    [2.23619670, 4.53976667, -2.91281683],
    [-0.96802659, 5.34898740, -2.65252309],
    [-1.40746967, 5.07942877, -0.54519764],
    [-1.84834650, 5.64462040, -2.97865474],
    [-0.04207089, -0.65847402, -0.01726345],
    [0.47274606, -1.55818436, 0.24803792],
    [-0.81975098, -0.46984115, 0.69304024],
    [-0.65564544, -0.81307387, -1.42126671],
    [-1.99166909, -0.58084470, -1.77625051],
    [0.03426135, -1.22202407, -2.61107524],
    [-2.13547452, -0.83428389, -3.13720231],
    [-2.83405504, -0.25863727, -1.16545493],
    [1.08510136, -1.47987280, -2.68895460],
    [-2.97115266, -0.75053457, -3.65038060],
    [0.17272994, 5.18738230, -3.43333007],
    [-0.90355989, -1.22599057, -3.65302263],
])

atom_labels = [
    "C", "C", "C", "C", "N", "H", "H", "C", "H", "H",
    "C", "C", "C", "H", "N", "H", "H", "C", "H", "H",
    "C", "C", "C", "N", "H", "H", "H", "N", "N",
]


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
        n_orb = len(mo_occ)
        n_lumo = next((i for i, o in enumerate(mo_occ) if o < occ_thresh), len(mo_occ) - 1)
        n_homo = max(n_lumo - 1, 0)
        k_max = min(n_homo, n_orb - 1 - n_lumo)
        kept = [n for n in range(n_homo - homo_minus_m, n_lumo + lumo_plus_n + 1)]
        frozen_orbitals = [n for n in range(len(mo_occ)) if n not in kept]
        print(f"Fragment: n_homo={n_homo}, n_lumo={n_lumo}, kept={kept}")
        print(f"Fragment active-space limits: n_orb={n_orb}, max symmetric [k,k]={k_max}")
        return frozen_orbitals

    return callable_for_dmet_object


def keep_all_orbitals(_info_fragment):
    return []


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
ccsd_options = {}

# TPP fragmentation with N4 as separate SQD fragment
# N4 coordinate: [0.51010341, 1.84005426, -0.01741100] (index 4)
fragment_atoms = [
    [0, 1, 2, 3, 5, 6],
    [7, 8, 9],
    [10, 11, 12, 13, 14, 15, 16, 17],
    [18, 19, 20],
    [21, 22, 23, 24, 25, 26, 27, 28],
    [4],
]

fragment_frozen_orbitals = [
    keep_all_orbitals,
    keep_all_orbitals,
    keep_all_orbitals,
    keep_all_orbitals,
    keep_all_orbitals,
    define_dmet_frag_as(1, 1),
]

fragment_solvers = ["ccsd", "ccsd", "ccsd", "ccsd", "ccsd", "sqd"]
solvers_options = [ccsd_options, ccsd_options, ccsd_options, ccsd_options, ccsd_options, sqd_options]

if enable_logging:
    logdir = os.path.join("logs", "monomers", basis_set)
    os.makedirs(logdir, exist_ok=True)
    pid = os.getpid()
    job_pid = os.environ.get("JOB_PID") or pid
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logfile = os.path.join(logdir, f"DMET_SQD_TPP_monomer_JOB{job_pid}_PID{pid}_{timestamp}.out")
    sys.stdout = open(logfile, "w")
    sys.stderr = sys.stdout

geom_root = os.path.join("relaxed_geoms", "monomers", basis_set)
os.makedirs(geom_root, exist_ok=True)

print("Optimizing TPP-only geometry...")
opt_geom = optimize_geom(base_geom, atom_labels, charge=-1, spin=0)

xyz_path = os.path.join(geom_root, "TPP_only_optimized.xyz")
with open(xyz_path, "w", encoding="utf-8") as f:
    f.write(f"{len(atom_labels)}\n")
    f.write("TPP optimized geometry\n")
    for i, label in enumerate(atom_labels):
        x, y, z = opt_geom[i]
        f.write(f"{label} {x:.10f} {y:.10f} {z:.10f}\n")

mol = SecondQuantizedMolecule(geom_to_string(opt_geom, atom_labels), q=-1, spin=0, basis=basis_set)
options = {
    "molecule": mol,
    "fragment_atoms": fragment_atoms,
    "fragment_frozen_orbitals": fragment_frozen_orbitals,
    "fragment_solvers": fragment_solvers,
    "solvers_options": solvers_options,
    "verbose": True,
}

dmet = DMETProblemDecomposition(options)
dmet.build()
energy = dmet.simulate()

print("TPP-only DMET completed")
print(f"TPP DMET energy (Hartree): {energy:.10f}")
print(f"XYZ written to: {xyz_path}")
