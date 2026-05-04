# -*- coding: utf-8 -*-
"""Relaxed CH3-CH3 scan with DMET(CCSD + hardware SQD)."""

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

# ========== 1. Base geometry: CH3-CH3 ==========
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

# C-C scan indices and moving CH3 group.
idx_C0 = 0
idx_C1 = 4
idx_CH3 = [4, 5, 6, 7]


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


def cached_noisy_optimizer(dmet_instance, mu0, options, verbose=True, dist_key="none"):
    """
    Root finder for noisy hardware residuals.

    Each chemical potential is evaluated with repeated real SQD calls and the
    median residual/energy is cached so restarts do not repeat completed jobs.
    """
    cache_file = "ch3_ch3_mid_iteration_cache.json"

    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                full_cache = json.load(f)
        except (json.JSONDecodeError, ValueError):
            print(f"[Warning] Cache file {cache_file} was empty or invalid. Re-initializing.")
            full_cache = {}
    else:
        full_cache = {}
    dist_cache = full_cache.get(dist_key, {})

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
                print(f"\t[OptimizerCache] Found mu={mu_val:.6f} in cache. Skipping hardware calls.")
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
        full_cache[dist_key] = dist_cache
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(full_cache, f, indent=4)

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
        if mu_str not in dist_cache:
            raise RuntimeError(
                f"Converged mu={mu_val:.8f} was not cached; refusing final hardware rerun."
            )

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


# Define your IBM Quantum accounts here. Replace the placeholder token/instance
# strings with your real account records.
ibm_accounts = [
    {
        "token": "lACZLMEAn7vpNL-5d1noHsLpN6WAMXSoqDIwO5e5_2Wh",
        "instance": "crn:v1:bluemix:public:quantum-computing:us-east:a/b4e4bb23aa6e44e1be61fa915376a9c9:4f20789a-8bf4-4796-9e6d-e8ed2bcfd1ca::",
    },
    {
        "token": "WdxBQYRjvS3BdrdZKaBY3hDvB52cxPCNDC9L4cXxKHGA",
        "instance": "crn:v1:bluemix:public:quantum-computing:us-east:a/14148440cede487aa30dbe7298d0bcb5:f8b6b57a-5cac-4e2d-86b8-e4dce6f4a96b::",
    },
    {
        "token": "oZ3YCOfzaAuGnwBlXVBwkXxQMl9jAq_u1JbPWRDFG-rt",
        "instance": "crn:v1:bluemix:public:quantum-computing:us-east:a/ecac468acb41455887924684d9eea836:836d9777-5a29-449f-8ae4-accaae2655e7::",
    },

]

# Initialize the manager with 20 iterations per account limit.
service = IBMAccountManager(ibm_accounts, iterations_per_account=10)

sqd_options = {
    "service": service,
    "backend": "ibm_kingston",
    "use_simulator": False,
    "sampling_ansatz": "lucj",
    "lucj_parameter_source": "ccsd",
    "lucj_optimize_t_amplitudes": True,
    "lucj_n_reps": 1,
    "lucj_seed": 123,
    "lucj_diag_coulomb_scale": 0.25,
    "lucj_interaction_pairs": "zigzag",
    "dynamical_decoupling": {
        "enable": True,
        "sequence_type": "XpXm",
        "scheduling_method": "alap",
    },
    "gate_twirling": {
        "enable_gates": True,
        "num_randomizations": 8,
    },
    "measurement_twirling": {
        "enable_measure": True,
        "num_randomizations": 8,
    },
    "shots": 200000,
    "n_batches": 16,
    "samples_per_batch": 1000,
    "max_iterations": 14,
    "diagnostics": True,
    "tol": 1e-4,
}

noisy_optimizer_options = {
    "residual_evaluations": 1,
    "residual_tol": 0.003,
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
# distances = np.linspace(1.1, 1.9, 9)
distances = [1.3]
basis_set = "3-21g"
fragmentation_type = "ch3_ch3"
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
    logfile = os.path.join(logdir, f"DMET_SQD_CH3_CH3_JOB{job_pid}_PID{pid}_{timestamp}.out")
    sys.stdout = open(logfile, "w", encoding="utf-8")
    sys.stderr = sys.stdout

fragment_atoms = [[0, 1, 2, 3], [4, 5, 6, 7]]
fragment_frozen_orbitals = [
    keep_all_orbitals,
    define_dmet_frag_as(sqd_homo_minus_m, sqd_lumo_plus_n),
]
fragment_solvers = ["ccsd", "sqd"]
solvers_options = [ccsd_options, sqd_options]


# ========== 6. Run relaxed scan ==========
energies = []
opt_geoms = []
csv_rows = []
checkpoint_file = "ch3_ch3_hardware_scan_checkpoint.json"

if os.path.exists(checkpoint_file):
    try:
        with open(checkpoint_file, "r", encoding="utf-8") as f:
            checkpoint_data = json.load(f)
        print(f"Resuming from checkpoint. {len(checkpoint_data)} points already completed.")
    except json.JSONDecodeError as exc:
        print(f"[Warning] Could not parse {checkpoint_file}: {exc}. Starting from an empty checkpoint.")
        checkpoint_data = {}
else:
    checkpoint_data = {}

prev_mu = None
geom_start = base_geom.copy()

for d in distances:
    dist_key = f"ch3_ch3_{d:.2f}"
    print(f"\n=== Relaxed scan: C-C distance = {d:.2f} A ===")

    if dist_key in checkpoint_data:
        print(f"Distance {dist_key} found in checkpoint. Skipping...")
        checkpoint_energy = checkpoint_data[dist_key]["energy"]
        checkpoint_xyz = checkpoint_data[dist_key]["xyz_file"]
        energies.append(checkpoint_energy)
        csv_rows.append((d, checkpoint_energy, checkpoint_xyz))
        prev_mu = checkpoint_data[dist_key]["mu"]
        continue

    geom_start = adjust_group_distance(geom_start, idx_C0, idx_C1, idx_CH3, d)

    geom_str = geom_to_string(geom_start, atom_labels)
    mol = gto.M(atom=geom_str, basis=basis_set, charge=0, spin=0, unit="Angstrom")
    mf = scf.RHF(mol)

    with tempfile.TemporaryDirectory() as tmpdir:
        cfile = os.path.join(tmpdir, "constraints.txt")
        write_constraints_file(cfile, idx_C0, idx_C1, d)
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
        xyz.write(f"C-C distance = {d:.8f} Angstrom\n")
        for i, label in enumerate(atom_labels):
            x, y, z = geom_opt[i]
            xyz.write(f"{label} {x:.10f} {y:.10f} {z:.10f}\n")

    mol_dmet = SecondQuantizedMolecule(geom_to_string(geom_opt, atom_labels), q=0, spin=0, basis=basis_set)

    options_dmet = {
        "molecule": mol_dmet,
        "fragment_atoms": fragment_atoms,
        "fragment_frozen_orbitals": fragment_frozen_orbitals,
        "fragment_solvers": fragment_solvers,
        "solvers_options": solvers_options,
        "noisy_optimizer_options": noisy_optimizer_options,
        "optimizer": lambda func, mu0: cached_noisy_optimizer(dmet_calc, mu0, noisy_optimizer_options, dist_key=dist_key),
        "verbose": True,
    }
    if prev_mu is not None:
        options_dmet["initial_chemical_potential"] = prev_mu

    dmet_calc = DMETProblemDecomposition(options_dmet)
    dmet_calc.build()
    energy = dmet_calc.simulate()
    prev_mu = dmet_calc.chemical_potential

    print(f"DMET energy (Hartree): {energy:.6f}")
    print(f"Converged DMET chemical potential: {dmet_calc.chemical_potential:.10f}")
    print(f"Final DMET electron residual: {dmet_calc.electron_residual:.10f}")
    energies.append(energy)

    checkpoint_data[dist_key] = {
        "energy": energy,
        "mu": dmet_calc.chemical_potential,
        "xyz_file": xyz_path,
    }
    with open(checkpoint_file, "w", encoding="utf-8") as f:
        json.dump(checkpoint_data, f, indent=4)

    csv_rows.append((d, energy, xyz_path))


# ========== 7. Plot ==========
plt.figure(figsize=(7, 5))
plt.plot(distances, energies, "o-", lw=2)
plt.xlabel("C-C bond distance (Angstrom)", fontsize=12)
plt.ylabel("Total DMET Energy (Hartree)", fontsize=12)
plt.title("Relaxed Scan: CH3-CH3 DMET(CCSD + hardware SQD)", fontsize=13)
plt.grid(True)
plt.tight_layout()
plt.show()


# ========== 8. Save CSV ==========
csv_path = "ch3_ch3_hardware_relaxed_scan_energies.csv"
with open(csv_path, "w", encoding="utf-8") as f:
    f.write("distance_angstrom,energy_hartree,xyz_file\n")
    for d, e, xyz in csv_rows:
        f.write(f"{d:.8f},{e:.10f},{xyz}\n")
