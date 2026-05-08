# -*- coding: utf-8 -*-
"""CO2-only monomer: geometry optimization + DMET(SQD) with active space [1,1]."""

import os
import sys
from datetime import datetime

import numpy as np
from pyscf import gto, scf, lib, geomopt
from qiskit_ibm_runtime import QiskitRuntimeService

from tangelo import SecondQuantizedMolecule
from tangelo.problem_decomposition import DMETProblemDecomposition

enable_logging = True
basis_set = "3-21g"
sqd_homo_minus_m = 1
sqd_lumo_plus_n = 1
active_space_folder = f"{sqd_homo_minus_m}{sqd_lumo_plus_n}_opt_geometries"


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


ibm_accounts = [
    {
        "token": "rbGyRKObakMFhGfY778gHAb7NfdpE78l9WYoEA8y1_zv",
        "instance": "crn:v1:bluemix:public:quantum-computing:us-east:a/ccd2a1d13f904154a274171c9d5031fe:dae8523b-38d4-41bd-b0e1-a0fd4a9e58af::",
    },
]

service = IBMAccountManager(ibm_accounts, iterations_per_account=10)

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
    try:
        opt_mol = geomopt.optimize(mf, maxsteps=100)
    except ImportError as exc:
        raise RuntimeError(
            "PySCF geometry optimization requires the geomeTRIC package. "
            "Install it with: pip install geometric"
        ) from exc
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


def noisy_optimizer(dmet_instance, mu0, options, verbose=True):
    """Root finder for noisy hardware residuals with in-memory caching."""
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
            res = dmet_instance._oneshot_loop(mu_val)
            residuals.append(float(res))
            energies.append(float(dmet_instance.dmet_energy))

        median_res = float(np.median(residuals))
        median_eng = float(np.median(energies))
        dist_cache[mu_str] = [median_res, median_eng]
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
        elif f0 * f_right < 0:
            bracket = (mu, right)
        elif f_left * f_right < 0:
            bracket = (left, right)
        elif left <= mu_min and right >= mu_max:
            print("[Optimizer] Warning: mu range exhausted without bracket. Falling back to damping.")
            break
        else:
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

    damped_step = float(options.get("damped_step", 0.1))
    max_mu_step = float(options.get("max_mu_step", 0.1))
    max_damped_steps = int(options.get("max_damped_steps", 10))
    print("[Optimizer] Persistent damping mode activated...")

    best_mu = mu
    best_res = evaluate(mu)
    for _ in range(max_damped_steps):
        res = evaluate(mu)
        if abs(res) < abs(best_res):
            best_mu = mu
            best_res = res
            print(f"[Optimizer] New best observed point: mu={best_mu:.8f}, residual={best_res:.6f}")
        if abs(res) <= tol:
            return accept_converged_mu(mu, res)
        step = float(np.clip(damped_step * res, -max_mu_step, max_mu_step))
        mu = float(np.clip(mu - step, mu_min, mu_max))

    print(f"[Optimizer] Returning best observed mu={best_mu:.6f}, residual={best_res:.6f}")
    return accept_converged_mu(best_mu, best_res)


sqd_options = {
    # Match the SQD settings used by tpp_CO2_relaxed_scan.py.
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

geom_root = os.path.join("relaxed_geoms", "monomers", basis_set, "sqd", active_space_folder)
os.makedirs(geom_root, exist_ok=True)

if enable_logging:
    logdir = os.path.join("logs", "monomers", basis_set, "sqd", active_space_folder)
    os.makedirs(logdir, exist_ok=True)
    pid = os.getpid()
    job_pid = os.environ.get("JOB_PID") or pid
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logfile = os.path.join(
        logdir,
        f"DMET_SQD_CO2_monomer_JOB{job_pid}_PID{pid}_{timestamp}.out",
    )
    sys.stdout = open(logfile, "w", encoding="utf-8")
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
    "fragment_frozen_orbitals": [define_dmet_frag_as(sqd_homo_minus_m, sqd_lumo_plus_n)],
    "fragment_solvers": ["sqd"],
    "solvers_options": [sqd_options],
    "noisy_optimizer_options": noisy_optimizer_options,
    "optimizer": lambda func, mu0: noisy_optimizer(co2_dmet, mu0, noisy_optimizer_options),
    "verbose": True,
}

co2_dmet = DMETProblemDecomposition(co2_options)
co2_dmet.build()
co2_energy = co2_dmet.simulate()

print(f"CO2 DMET energy (Hartree): {co2_energy:.10f}")
print(f"Converged DMET chemical potential: {co2_dmet.chemical_potential:.10f}")
print(f"Final DMET electron residual: {co2_dmet.electron_residual:.10f}")
print(f"Optimized geometry saved to: {co2_xyz}")
