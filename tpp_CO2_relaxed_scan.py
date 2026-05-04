# -*- coding: utf-8 -*-
"""Relaxed TPP-CO2 scan with DMET(CCSD + hardware SQD)."""

import json
import os
import sys
import tempfile
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from pyscf import geomopt, gto, lib, scf
from qiskit_ibm_runtime import QiskitRuntimeService

from tangelo import SecondQuantizedMolecule
from tangelo.problem_decomposition import DMETProblemDecomposition

enable_logging = True

# ========== 1. Base geometry: TPP-CO2 ==========
base_geom = np.array([
    [0.60348246, 1.85634787, 0.00000900],   # C0
    [2.00523246, 1.85634787, 0.00000900],   # C1
    [2.42923046, 3.22714387, 0.00000900],   # C2
    [1.27217346, 4.01859687, -0.00009500],  # C3
    [0.16375546, 3.17685587, 0.00000000],   # N4
    [2.64960546, 0.98364187, 0.00003400],   # H5
    [3.45379346, 3.58363587, -0.00001600],  # H6
    [1.11817512, 5.55087768, -0.00030283],  # C7
    [0.58234684, 5.85536999, -0.87498706],  # H8
    [2.08625588, 6.00663630, 0.00162156],   # H9
    [0.41794696, 6.43481046, 1.04847902],   # C10
    [0.94555934, 6.81916660, 2.32627067],   # C11
    [-0.85341290, 7.01422678, 0.93402548],  # C12
    [1.91195594, 6.54119224, 2.73325985],   # H13
    [-1.11419619, 7.73513882, 2.09566065],  # N14
    [-1.57688110, 6.96413620, 0.12130753],  # H15
    [-1.93660749, 8.24242321, 2.28281318],  # H16
    [-0.02105623, 7.62051961, 2.94948071],  # N17
    [-0.38841884, 0.67832759, 0.00014755],  # C18
    [0.12608675, -0.22118313, -0.26643162], # H19
    [-1.16692993, 0.86749320, -0.70910336], # H20
    [-1.29101695, 0.04297686, 1.07404052],  # C21
    [-2.69163711, 0.08241227, 1.11417873],  # C22
    [-0.85473614, -0.70045872, 2.22113188], # C23
    [-3.11884790, -0.60988101, 2.24348035], # N24
    [-3.40033723, 0.54671102, 0.42958121],  # H25
    [0.17228519, -0.91949887, 2.49327980],  # H26
    [-4.05573224, -0.73837213, 2.51635694], # H27
    [-2.00356745, -1.09016783, 2.92362186], # N28
    [-0.69609292, 3.44267672, -0.00019843], # C29
    [-1.35272396, 3.73209590, 1.23672203],  # O30
    [-1.32249999, 3.56053861, -1.08522991], # O31
])

atom_labels = ["C","C","C","C","N","H","H","C","H","H","C","C","C","H","N","H","H","N","C","H","H","C","C","C","N","H","H","H","N","C","O","O"]

# N and CO2 indices (0-based)
idx_N = 4
idx_C = 29
idx_CO2 = [29, 30, 31]


# ========== 2. Helpers ==========
def geom_to_string(geom, labels):
    return "\n".join(
        f"{labels[i]} {geom[i, 0]} {geom[i, 1]} {geom[i, 2]}" for i in range(len(labels))
    )


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
    # geomeTRIC uses 1-based atom indexing.
    with open(path, "w", encoding="utf-8") as f:
        f.write("$set\n")
        f.write(f"distance {i + 1} {j + 1} {dist_ang:.8f}\n")


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


def newton_optimizer(func, mu0):
    """Force Tangelo's deterministic Newton optimizer even when an SQD fragment is present."""
    import scipy.optimize

    result = scipy.optimize.newton(func, mu0, tol=1e-4)
    return result.real


def noisy_optimizer(dmet_instance, mu0, options, verbose=True):
    """
    Root finder for noisy hardware residuals with in-memory caching.
    """
    dist_cache = {}

    original_oneshot = dmet_instance._oneshot_loop
    dmet_instance._final_bypass = False

    def patched_oneshot(mu, save_results=False, **kwargs):
        mu_val = float(np.real(mu))
        mu_str = f"{mu_val:.8f}"
        if dmet_instance._final_bypass and mu_str in dist_cache:
            res, eng = dist_cache[mu_str]
            dmet_instance.dmet_energy = eng
            if verbose:
                print(f"\n\t[Cache] Skipping final redundant call for converged mu={mu_val:.6f}")
            return res
        return original_oneshot(mu, save_results=save_results, **kwargs)

    dmet_instance._oneshot_loop = patched_oneshot

    def evaluate(mu):
        mu_val = float(np.real(mu))
        mu_str = f"{mu_val:.8f}"
        if mu_str in dist_cache:
            res, _eng = dist_cache[mu_str]
            if verbose:
                print(f"\t[OptimizerCache] Found mu={mu_val:.6f} in local memory. Skipping hardware calls.")
            return res

        n_eval = max(1, int(options.get("residual_evaluations", 1)))
        if verbose:
            print(f"\t[Optimizer] Evaluating mu={mu_val:.6f} with {n_eval} hardware jobs...")

        residuals = []
        energies = []
        for _ in range(n_eval):
            res = dmet_instance._oneshot_loop(mu)
            residuals.append(float(res))
            energies.append(float(dmet_instance.dmet_energy))

        median_res = float(np.median(residuals))
        median_eng = float(np.median(energies))

        dist_cache[mu_str] = [median_res, median_eng]

        if verbose and n_eval > 1:
            spread = float(np.max(np.abs(np.array(residuals) - median_res)))
            print(
                f"\t[Optimizer] Finished mu={mu_val:.6f}. "
                f"Median residual: {median_res:.6f}, spread: {spread:.6f}"
            )
        return median_res

    def accept_converged_mu(mu, residual=None):
        mu_val = float(np.real(mu))
        mu_str = f"{mu_val:.8f}"
        if residual is None:
            residual = evaluate(mu_val)
        
        res, eng = dist_cache[mu_str]
        dmet_instance._final_bypass = True
        dmet_instance._skip_final_oneshot = True
        dmet_instance._final_cached_mu = mu_val
        dmet_instance._final_cached_residual = float(res)
        dmet_instance._final_cached_energy = float(eng)
        dmet_instance.electron_residual = float(res)
        dmet_instance.dmet_energy = float(eng)
        return mu_val

    tol = float(options.get("residual_tol", 1e-2))
    mu_min = float(options.get("mu_min", -0.5))
    mu_max = float(options.get("mu_max", 0.5))
    mu = float(np.clip(mu0, mu_min, mu_max))

    f0 = evaluate(mu)
    if abs(f0) <= tol:
        return accept_converged_mu(mu, f0)

    radius = float(options.get("bracket_radius", 0.05))
    growth = float(options.get("bracket_growth", 1.5))
    bracket = None

    while bracket is None:
        left = float(np.clip(mu - radius, mu_min, mu_max))
        right = float(np.clip(mu + radius, mu_min, mu_max))
        f_left = evaluate(left)
        f_right = evaluate(right)

        if abs(f_left) <= tol:
            return accept_converged_mu(left, f_left)
        if abs(f_right) <= tol:
            return accept_converged_mu(right, f_right)
        if f_left * f0 < 0:
            bracket = (left, mu)
            break
        if f0 * f_right < 0:
            bracket = (mu, right)
            break
        if f_left * f_right < 0:
            bracket = (left, right)
            break
        if left <= mu_min and right >= mu_max:
            print("[Optimizer] Warning: mu range exhausted without bracket. Falling back to damping.")
            break
        radius *= growth

    if bracket:
        import scipy.optimize

        print(f"[Optimizer] Bracket found: {bracket}. Starting Brent search...")
        mu = scipy.optimize.brentq(
            lambda m: evaluate(m),
            bracket[0],
            bracket[1],
            xtol=float(options.get("brent_tol", 1e-4)),
        )
        res = evaluate(mu)
        if abs(res) <= tol:
            print(f"[Optimizer] Brent converged successfully at mu={mu:.6f}")
            return accept_converged_mu(mu, res)
        print(
            f"[Optimizer] Brent residual ({res:.4f}) > tol ({tol}). "
            "Continuing with persistent damping..."
        )

    damped_step = float(options.get("damped_step", 0.1))
    max_mu_step = float(options.get("max_mu_step", 0.1))
    print("[Optimizer] Persistent damping mode activated...")

    best_mu = mu
    best_res = evaluate(mu)
    while True:
        res = evaluate(mu)
        if abs(res) < abs(best_res):
            best_mu = mu
            best_res = res
            print(f"[Optimizer] New best observed point: mu={best_mu:.8f}, residual={best_res:.6f}")

        if abs(res) <= tol:
            return accept_converged_mu(mu, res)

        step = float(np.clip(damped_step * res, -max_mu_step, max_mu_step))
        new_mu = float(np.clip(mu - step, mu_min, mu_max))
        mu_str = f"{new_mu:.8f}"
        if mu_str in dist_cache and abs(dist_cache[mu_str][0]) > tol:
            new_mu += np.random.uniform(-1e-5, 1e-5)
        mu = new_mu


# ========== 4. Solver options ==========
class IBMAccountManager:
    """Manage one or more IBM Quantum accounts and rotate when usage limits are hit."""

    def __init__(self, accounts, iterations_per_account=20):
        if not accounts:
            raise ValueError("At least one IBM account must be provided to IBMAccountManager.")
        self.accounts = accounts
        self.iterations_per_account = iterations_per_account
        self.current_account_idx = 0
        self.account_iterations = 0
        self.total_iterations = 0
        self._current_service = None
        self._load_service()

    def _load_service(self):
        acc = self.accounts[self.current_account_idx]
        print(f"\n[AccountManager] Initializing IBM account {self.current_account_idx + 1}/{len(self.accounts)}")
        print(f"[AccountManager] Token: ...{acc['token'][-5:]}")
        self._current_service = QiskitRuntimeService(
            channel="ibm_quantum_platform",
            token=acc["token"],
            instance=acc["instance"],
        )
        self.account_iterations = 0

    def backend(self, name):
        for _ in range(len(self.accounts)):
            if self.account_iterations >= self.iterations_per_account:
                print(
                    f"[AccountManager] Soft limit of {self.iterations_per_account} reached for "
                    f"account {self.current_account_idx + 1}. Rotating..."
                )
                self.current_account_idx = (self.current_account_idx + 1) % len(self.accounts)
                self._load_service()

            try:
                backend_obj = self._current_service.backend(name)
                self.account_iterations += 1
                self.total_iterations += 1
                print(
                    f"[AccountManager] SQD call {self.total_iterations}: using account "
                    f"{self.current_account_idx + 1} ({self.account_iterations}/"
                    f"{self.iterations_per_account})"
                )
                return backend_obj
            except Exception as exc:
                error_msg = str(exc).lower()
                if any(kw in error_msg for kw in ["limit", "usage", "time", "quota", "access"]):
                    print(f"\n[AccountManager] Account {self.current_account_idx + 1} limit/access error: {exc}")
                    self.current_account_idx = (self.current_account_idx + 1) % len(self.accounts)
                    self._load_service()
                else:
                    raise

        raise RuntimeError("All IBM Quantum accounts reached usage limits or failed.")

    def __getattr__(self, name):
        return getattr(self._current_service, name)


# Define your IBM Quantum accounts here.
ibm_accounts = [
    {
        "token": "kWEZHN3iTHzoqRDqKk7oWyLe1WI1dpgPj9EBZxH_u_pZ",
        "instance": "crn:v1:bluemix:public:quantum-computing:us-east:a/457fbf8ffcdb4f8eaa005e51f1310c72:13af0048-a1ea-474a-b286-7b343383816e::",
    },

   {
        "token": "QpukZw3Tr7FJQil6ZnJw2PamzS0Old33BP83n_JDkB75",
        "instance": "crn:v1:bluemix:public:quantum-computing:us-east:a/0726ecf18f7f49f5b9ddb084a08a6b02:cb535801-7d53-4ce4-8bfa-33e2f0c50e6e::",
    },
    {
        "token": "YcWSqUmKsaAT_Ig5LQJ3N40PaG492mj27Do1YmyxV9yY",
        "instance": "crn:v1:bluemix:public:quantum-computing:us-east:a/46541683cbe04712ba1286872f0c74e9:fd0ca640-5da0-4cec-b94d-bf5e37ec8d7b::",
    },
    {
        "token": "_mpUflwlvZJPe6GCdHt_WUjhAZ9yKhdDSCzR6hBryYss",
        "instance": "crn:v1:bluemix:public:quantum-computing:us-east:a/4fdc283da539404e97f2634a099643dd:e50a4a56-e9e3-4304-83ec-bb438721d71a::",
    },
    {
        "token": "pvMLYVXcwUF4WWnZwtopLmFP_yPpmQwata63dmzinOW-",
        "instance": "crn:v1:bluemix:public:quantum-computing:us-east:a/0ad335e8eeb94c0f9e16f896f1613763:bc166bbb-efa7-4826-9576-0152a5e3bf5e::",
    },
    {
        "token": "BbQOHAZCWuDgVVbqxhMWhJJCRenZJZs3GeC-dasWSVqx",
        "instance": "crn:v1:bluemix:public:quantum-computing:us-east:a/1f3400031e9644918728fca1cd67ad26:a60fff20-afd9-40cf-9128-9194cfe59d6a::",
    },
    {
        "token": "lCOlKoRXSOj7ZfT0Osk7ZoGS1hvL33rA36e7mM1xf3eH",
        "instance": "crn:v1:bluemix:public:quantum-computing:us-east:a/eb64916b69664a28b8a31f60c6df5fe3:52ac9b17-8f0d-4fa7-bb1c-e8b16b9f2dae::",
    },
    {
        "token": "MQhbO0nfvKL156EBFz5PIFibN_QXsGNfMr0mQcvHHN6T",
        "instance": "crn:v1:bluemix:public:quantum-computing:us-east:a/7e6d614bc92045cf849801695aa631d4:7b0f013c-1263-4f3b-9834-7a7fd952960a::",
    },
    {
        "token": "QRBK-w8D5--0sXFvRFMokmRcE7EBdtiWwG__yJsCWpxm",
        "instance": "crn:v1:bluemix:public:quantum-computing:us-east:a/d01b336c5d3c4707b9e6dac2718e47cf:54e8be8a-ba5a-4fc0-bda2-6173688cb80f::",
    },

]

# Initialize the manager with 10 iterations per account limit for hardware runs.
# service = IBMAccountManager(ibm_accounts, iterations_per_account=10)
service = None

sqd_options = {
    # Hardware settings kept here for later restoration.
    # "service": service,
    # "backend": "ibm_kingston",
    # "use_simulator": False,
    "backend": "aer_simulator",
    "use_simulator": True,
    "sim_method": "matrix_product_state",
    "sampling_ansatz": "hf",
    # LUCJ sampling settings kept here for later restoration after HF baseline checks.
    # "sampling_ansatz": "lucj",
    # "lucj_parameter_source": "ccsd",
    # "lucj_optimize_t_amplitudes": False,
    # "lucj_n_reps": 1,
    # "lucj_seed": 123,
    # "lucj_diag_coulomb_scale": 0.25,
    # "lucj_interaction_pairs": "zigzag",
    # Hardware mitigation settings.
    # "dynamical_decoupling": {
    #     "enable": True,
    #     "sequence_type": "XpXm",
    #     "scheduling_method": "alap",
    # },
    # "gate_twirling": {
    #     "enable_gates": True,
    #     "num_randomizations": 8,
    # },
    # "measurement_twirling": {
    #     "enable_measure": True,
    #     "num_randomizations": 8,
    # },
    "shots": 200000,
    "n_batches": 16,
    "samples_per_batch": 1000,
    "max_iterations": 14,
    "diagnostics": True,
    "tol": 1e-4,
}

noisy_optimizer_options = {
    "residual_evaluations": 1,
    "residual_tol": 0.03,
    "bracket_radius": 0.05,
    "bracket_growth": 1.5,
    "max_bracket_steps": 8,
    "mu_min": -0.5,
    "mu_max": 0.5,
    "brent_tol": 1e-8,
    "damped_step": 0.1,
    "max_mu_step": 0.1,
    "max_damped_steps": 10,
}

ccsd_options = {}


# ========== 5. Scan settings ==========
#distances = [1.4]
distances = np.linspace(1.3, 1.7, 5)
basis_set = "3-21g"
fragmentation_type = "type_C"
sqd_homo_minus_m = 1
sqd_lumo_plus_n = 1
active_space_folder = f"{sqd_homo_minus_m}{sqd_lumo_plus_n}_opt_geometries"

solver_geom_dir = os.path.join("relaxed_geoms", fragmentation_type, basis_set, "sqd", active_space_folder)
os.makedirs(solver_geom_dir, exist_ok=True)

if enable_logging:
    logdir = os.path.join("logs", fragmentation_type, basis_set, "sqd", active_space_folder)
    os.makedirs(logdir, exist_ok=True)
    pid = os.getpid()
    job_pid = os.environ.get("JOB_PID") or pid
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logfile = os.path.join(logdir, f"DMET_SQD_TPP_CO2_JOB{job_pid}_PID{pid}_{timestamp}.out")
    sys.stdout = open(logfile, "w", encoding="utf-8")
    sys.stderr = sys.stdout

fragment_atoms = [
    [0, 1, 2, 3, 5, 6],
    [7, 8, 9],
    [10, 11, 12, 13, 14, 15, 16, 17],
    [18, 19, 20],
    [21, 22, 23, 24, 25, 26, 27, 28],
    [4, 29, 30, 31],
]
fragment_frozen_orbitals = [
    keep_all_orbitals,
    keep_all_orbitals,
    keep_all_orbitals,
    keep_all_orbitals,
    keep_all_orbitals,
    define_dmet_frag_as(sqd_homo_minus_m, sqd_lumo_plus_n),
]
fragment_solvers = ["ccsd", "ccsd", "ccsd", "ccsd", "ccsd", "sqd"]
solvers_options = [ccsd_options, ccsd_options, ccsd_options, ccsd_options, ccsd_options, sqd_options]


# ========== 6. Run relaxed scan ==========
energies = []
opt_geoms = []
csv_rows = []

prev_mu = None
geom_start = base_geom.copy()

for d in distances:
    print(f"\n=== Relaxed scan: N-C distance = {d:.2f} A ===")

    geom_start = adjust_group_distance(geom_start, idx_N, idx_C, idx_CO2, d)

    geom_str = geom_to_string(geom_start, atom_labels)
    mol = gto.M(atom=geom_str, basis=basis_set, charge=-1, spin=0, unit="Angstrom")
    mf = scf.RHF(mol)

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

    xyz_path = os.path.join(solver_geom_dir, f"relaxed_geom_{d:.2f}_A.xyz")
    with open(xyz_path, "w", encoding="utf-8") as xyz:
        xyz.write(f"{len(atom_labels)}\n")
        xyz.write(f"N-C distance = {d:.8f} Angstrom\n")
        for i, label in enumerate(atom_labels):
            x, y, z = geom_opt[i]
            xyz.write(f"{label} {x:.10f} {y:.10f} {z:.10f}\n")

    mol_dmet = SecondQuantizedMolecule(geom_to_string(geom_opt, atom_labels), q=-1, spin=0, basis=basis_set)

    options_dmet = {
        "molecule": mol_dmet,
        "fragment_atoms": fragment_atoms,
        "fragment_frozen_orbitals": fragment_frozen_orbitals,
        "fragment_solvers": fragment_solvers,
        "solvers_options": solvers_options,
        "noisy_optimizer_options": noisy_optimizer_options,
        "optimizer": lambda func, mu0: noisy_optimizer(dmet_calc, mu0, noisy_optimizer_options),
        "verbose": True,
    }

    dmet_calc = DMETProblemDecomposition(options_dmet)
    dmet_calc.build()
    energy = dmet_calc.simulate()
    prev_mu = dmet_calc.chemical_potential

    print(f"DMET energy (Hartree): {energy:.6f}")
    print(f"Converged DMET chemical potential: {dmet_calc.chemical_potential:.10f}")
    print(f"Final DMET electron residual: {dmet_calc.electron_residual:.10f}")
    energies.append(energy)

    csv_rows.append((d, energy, xyz_path))


# ========== 7. Plot ==========
plt.figure(figsize=(7, 5))
plt.plot(distances, energies, "o-", lw=2)
plt.xlabel("N-C bond distance (Angstrom)", fontsize=12)
plt.ylabel("Total DMET Energy (Hartree)", fontsize=12)
plt.title("Relaxed Scan: TPP-CO2 DMET(CCSD + hardware SQD)", fontsize=13)
plt.grid(True)
plt.tight_layout()
plt.show()


# ========== 8. Save CSV ==========
csv_path = "tpp_co2_hardware_relaxed_scan_energies_new.csv"
with open(csv_path, "w", encoding="utf-8") as f:
    f.write("distance_angstrom,energy_hartree,xyz_file\n")
    for d, e, xyz in csv_rows:
        f.write(f"{d:.8f},{e:.10f},{xyz}\n")
