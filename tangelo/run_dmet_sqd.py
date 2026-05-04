import numpy as np
import matplotlib.pyplot as plt

from tangelo import SecondQuantizedMolecule
from tangelo.problem_decomposition import DMETProblemDecomposition
#from tangelo.problem_decomposition.dmet.dmet_edited_version import DMETProblemDecomposition
#from tangelo.toolboxes.molecular_computation.molecule import SecondQuantizedMolecule
from tangelo.algorithms import BuiltInAnsatze as Ansatze

# ========== 1. Base geometry ==========
base_geom = np.array([
    [-0.796, -1.145, -0.001],   # C0
    [-2.099, -0.732, -0.001],   # C1
    [-2.104,  0.722,  0.001],   # C2
    [-0.797,  1.148,  0.000],   # C3
    [-0.348, -2.125, -0.001],   # H4
    [-2.970, -1.370, -0.001],   # H5
    [-2.978,  1.355,  0.001],   # H6
    [-0.366,  2.136,  0.001],   # H7
    [ 0.014,  0.012,  0.000],   # N8
    [ 1.455,  0.003,  0.000],   # C9 (CO2 carbon)
    [ 2.038, -0.005,  1.123],   # O10
    [ 2.038, -0.002, -1.123]    # O11
])

atom_labels = ["C","C","C","C","H","H","H","H","N","C","O","O"]

# Indices for N and CO₂ atoms
idx_N  = 8   # N
idx_CO2 = [9,10,11]  # C, O, O of CO2

# Compute initial N–C distance and bond vector
vec_NC = base_geom[idx_CO2[0]] - base_geom[idx_N]
dist_NC = np.linalg.norm(vec_NC)
unit_vec = vec_NC / dist_NC

print(f"Initial N–C distance: {dist_NC:.3f} Å")

# ========== 2. Function to move CO₂ group ==========
def move_CO2(distance):
    """Return a new geometry with CO₂ translated so that N–C distance = distance (Å)."""
    shift = (distance - dist_NC) * unit_vec
    new_geom = base_geom.copy()
    new_geom[idx_CO2] += shift  # translate the CO2 group
    return new_geom

# ========== 3. Helper to define DMET fragment active space ==========
def define_dmet_frag_as(homo_minus_m=0, lumo_plus_n=0, occ_thresh=0.5):
    def callable_for_dmet_object(info_fragment):
        mf_fragment, _, _, _, _, _, _ = info_fragment
        mo_occ = list(mf_fragment.mo_occ)
        # find first orbital with occupation < occ_thresh (interpreted as LUMO)
        n_lumo = next((i for i,o in enumerate(mo_occ) if o < occ_thresh), len(mo_occ)-1)
        n_homo = max(n_lumo - 1, 0)
        frozen_orbitals = [n for n in range(len(mo_occ))
                           if n not in range(n_homo-homo_minus_m, n_lumo+lumo_plus_n+1)]
        # debug print
        print(f"Fragment: n_homo={n_homo}, n_lumo={n_lumo}, kept={[n for n in range(n_homo-homo_minus_m, n_lumo+lumo_plus_n+1)]}")
        return frozen_orbitals
    return callable_for_dmet_object


def keep_all_orbitals(info_fragment):
    """Return empty list → no frozen orbitals (full treatment)."""
    return []


# ========== 4. DMET + VQE setup (options fixed) ==========
vqe_options = {"qubit_mapping": "jw", "ansatz": Ansatze.UCCSD}
ccsd_options = {}
# Range of distances (Å)
distances = np.linspace(1.1, 2.6, 16)
energies = []

# ========== 5. Main DMET loop ==========
for d in distances:
    print(f"\n=== Running DMET + VQE for N–CO₂ distance = {d:.2f} Å ===")
    geom = move_CO2(d)

    # Convert geometry to Tangelo string format
    geom_str = "\n".join([f"{atom_labels[i]} {geom[i,0]} {geom[i,1]} {geom[i,2]}" for i in range(len(atom_labels))])

    mol = SecondQuantizedMolecule(geom_str, q=-1, spin=0, basis="3-21g")

    options_dmet = {
        "molecule": mol,
        "fragment_atoms": [[0],[1],[2],[3],[4],[5],[6],[7],[8],[9],[10],[11]],
        "fragment_frozen_orbitals": [keep_all_orbitals,keep_all_orbitals,keep_all_orbitals,
                                     keep_all_orbitals,keep_all_orbitals,keep_all_orbitals,
                                     keep_all_orbitals,keep_all_orbitals,
                                     keep_all_orbitals,
                                     define_dmet_frag_as(0,0),
                                     define_dmet_frag_as(0,0),
                                     define_dmet_frag_as(0,0)],

        #"fragment_solvers": [["ccsd"]*8,["vqe"]*3],
        "fragment_solvers": ["ccsd","ccsd","ccsd","ccsd","ccsd","ccsd","ccsd","ccsd","ccsd","sqd","sqd","sqd"],
        "solvers_options": [ccsd_options,ccsd_options,ccsd_options,ccsd_options,ccsd_options,ccsd_options,ccsd_options,ccsd_options,ccsd_options, sqd_options,sqd_options,sqd_options],
        #"initial_chemical_potential":0.5,
        "verbose": False
    }

    try:
        dmet_calc = DMETProblemDecomposition(options_dmet)
        dmet_calc.build()
        energy = dmet_calc.simulate()
        print(dmet_calc.get_resources())
        print(f"DMET + VQE energy (Hartree): {energy:.6f}")
        energies.append(energy)
    except Exception as e:
        print(f"Error at distance {d:.2f}: {e}")
        energies.append(np.nan)

# ========== 6. Plot Potential Energy Curve ==========
plt.figure(figsize=(7,5))
plt.plot(distances, energies, "o-", lw=2)
plt.xlabel("N–C bond distance (Å)", fontsize=12)
plt.ylabel("Total DMET Energy (Hartree)", fontsize=12)
plt.title("Potential Energy Curve: Pyrrole–CO₂ (DMET + VQE)", fontsize=13)
plt.grid(True)
plt.tight_layout()
plt.show()
