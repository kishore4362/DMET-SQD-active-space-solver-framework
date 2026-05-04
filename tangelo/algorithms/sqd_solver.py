import numpy as np
import ffsim
import ffsim.random

from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

try:
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as RuntimeSampler
except Exception:
    QiskitRuntimeService = None
    RuntimeSampler = None

try:
    from qiskit_aer import AerSimulator
except Exception:
    AerSimulator = None

from qiskit_addon_sqd.fermion import solve_fermion
from qiskit_addon_sqd.counts import counts_to_arrays
from qiskit_addon_sqd.configuration_recovery import recover_configurations
from qiskit_addon_sqd.subsampling import postselect_and_subsample

from tangelo.toolboxes.quantum_layouts.zigzag_layout import (
    get_zigzag_physical_layout
)

class SQDSolver:
    """Sample-based Quantum Diagonalization solver compatible with DMET."""

    def __init__(self, options):
        self.molecule = options["molecule"]
        backend_opt = options.get("backend")
        self.service = options.get("service", None)
        self.diagnostics = options.get("diagnostics", False)

        self.backend_name = (
            backend_opt if isinstance(backend_opt, str) else getattr(backend_opt, "name", "custom")
        )
        self.backend_obj = backend_opt if hasattr(backend_opt, "run") else None

        self.use_simulator = options.get("use_simulator", False)
        self.sim_method = options.get("sim_method", "matrix_product_state")
        if isinstance(backend_opt, str) and backend_opt in {"aer_simulator", "aer"}:
            self.use_simulator = True
        if self.service is None and self.backend_obj is None:
            self.use_simulator = True
        self.shots = options.get("shots", 100_000)
        self.sampler_options = options.get("sampler_options", {})
        self.dynamical_decoupling = options.get(
            "dynamical_decoupling",
            options.get("dynamic_decoupling"),
        )
        self.twirling = options.get("twirling")
        self.gate_twirling = options.get("gate_twirling")
        self.measurement_twirling = options.get("measurement_twirling")

        # SQD hyperparameters
        self.n_batches = options.get("n_batches", 8)
        self.samples_per_batch = options.get("samples_per_batch", 500)
        self.max_iterations = options.get("max_iterations", 100)
        self.tol = options.get("tol", 1e-8)

        # Sampling ansatz. HF is the historical default. LUCJ intentionally
        # broadens the determinant pool before SQD postselection.
        self.sampling_ansatz = options.get("sampling_ansatz", "hf").lower()
        self.lucj_n_reps = options.get("lucj_n_reps", 1)
        self.lucj_seed = options.get("lucj_seed", 123)
        self.lucj_diag_coulomb_scale = options.get("lucj_diag_coulomb_scale", 0.25)
        self.lucj_interaction_pairs = options.get("lucj_interaction_pairs", "nearest")
        self.lucj_with_final_orbital_rotation = options.get("lucj_with_final_orbital_rotation", False)
        self.lucj_parameter_source = options.get("lucj_parameter_source", "random").lower()
        self.lucj_t_amplitudes_tol = options.get("lucj_t_amplitudes_tol", 1e-8)
        self.lucj_optimize_t_amplitudes = options.get("lucj_optimize_t_amplitudes", False)
        self.lucj_ccsd_conv_tol = options.get("lucj_ccsd_conv_tol", 1e-9)
        self.lucj_ccsd_conv_tol_normt = options.get("lucj_ccsd_conv_tol_normt", 1e-7)
        self.lucj_ccsd_max_cycle = options.get("lucj_ccsd_max_cycle", 50)

        self.open_shell = False
        self.spin_sq = 0



        self._built = False
        self._simulated = False

    def _lucj_interactions(self, norb):
        """Return sparse interaction pairs for a low-depth LUCJ sampling circuit."""

        if self.lucj_interaction_pairs in (None, "all"):
            return None
        if self.lucj_interaction_pairs == "nearest":
            pairs = [(i, i) for i in range(norb)]
            pairs.extend((i, i + 1) for i in range(norb - 1))
            return (pairs, pairs)
        if self.lucj_interaction_pairs == "zigzag":
            same_spin_pairs = [(i, i + 1) for i in range(norb - 1)]
            opposite_spin_pairs = [(i, i) for i in range(0, norb, 4)]
            return (same_spin_pairs, opposite_spin_pairs)
        if isinstance(self.lucj_interaction_pairs, tuple):
            return self.lucj_interaction_pairs
        raise ValueError(
            "lucj_interaction_pairs must be one of None, 'all', 'nearest', 'zigzag', "
            "or a tuple of alpha-alpha and alpha-beta interaction-pair lists."
        )

    def _append_sampling_ansatz(self, qc, qr, norb, nelec_a, nelec_b):
        """Prepare the selected SQD sampling state."""

        qc.append(ffsim.qiskit.PrepareHartreeFockJW(norb, (nelec_a, nelec_b)), qr)

        if self.sampling_ansatz in {"hf", "hartree_fock"}:
            return
        if self.sampling_ansatz not in {"lucj", "ucj"}:
            raise ValueError(f"Unsupported SQD sampling_ansatz: {self.sampling_ansatz}")

        if self.lucj_parameter_source == "random":
            ucj_op = ffsim.random.random_ucj_op_spin_balanced(
                norb,
                n_reps=self.lucj_n_reps,
                interaction_pairs=self._lucj_interactions(norb),
                with_final_orbital_rotation=self.lucj_with_final_orbital_rotation,
                diag_coulomb_scale=self.lucj_diag_coulomb_scale,
                seed=self.lucj_seed,
            )
        elif self.lucj_parameter_source in {"ccsd", "t_amplitudes", "t-amplitudes"}:
            ucj_op = self._ccsd_initialized_ucj_op(norb, nelec_a, nelec_b)
        else:
            raise ValueError(
                "lucj_parameter_source must be one of 'random' or 'ccsd'. "
                f"Got {self.lucj_parameter_source!r}."
            )
        qc.append(ffsim.qiskit.UCJOpSpinBalancedJW(ucj_op), qr)

    def _ccsd_initialized_ucj_op(self, norb, nelec_a, nelec_b):
        """Build a spin-balanced UCJ/LUCJ operator from restricted CCSD amplitudes."""

        if nelec_a != nelec_b:
            raise ValueError("CCSD-initialized LUCJ currently requires a closed-shell fragment.")
        if getattr(self.molecule, "uhf", False) or getattr(self.molecule, "spin", 0) != 0:
            raise ValueError("CCSD-initialized LUCJ currently requires an RHF reference.")

        from pyscf import cc

        cc_fragment = cc.CCSD(self.molecule.mean_field, frozen=self.molecule.frozen_mos)
        cc_fragment.verbose = 0
        cc_fragment.conv_tol = self.lucj_ccsd_conv_tol
        cc_fragment.conv_tol_normt = self.lucj_ccsd_conv_tol_normt
        cc_fragment.max_cycle = self.lucj_ccsd_max_cycle
        _, t1, t2 = cc_fragment.ccsd()

        if t2.shape[0] + t2.shape[2] != norb:
            raise ValueError(
                "CCSD amplitudes are not aligned with the SQD active space: "
                f"t2 shape={t2.shape}, expected nocc+nvirt={norb}."
            )

        ucj_op = ffsim.UCJOpSpinBalanced.from_t_amplitudes(
            t2,
            t1=t1,
            n_reps=self.lucj_n_reps,
            interaction_pairs=self._lucj_interactions(norb),
            tol=self.lucj_t_amplitudes_tol,
            optimize=self.lucj_optimize_t_amplitudes,
        )
        if self.diagnostics:
            print(
                "LUCJ CCSD init diag: "
                f"norb={norb}, n_reps={ucj_op.n_reps}, "
                f"t1_shape={t1.shape}, t2_shape={t2.shape}, "
                f"t1_norm={np.linalg.norm(t1):.6e}, t2_norm={np.linalg.norm(t2):.6e}"
            )
        return ucj_op

    def _normalize_option_block(self, value, default_key):
        """Normalize convenience option values into runtime option dictionaries."""

        if value is None:
            return {}
        if isinstance(value, bool):
            return {default_key: value}
        if isinstance(value, dict):
            return value.copy()
        raise TypeError(
            f"Expected bool or dict for runtime option block '{default_key}', got {type(value).__name__}."
        )

    def _get_runtime_sampler_options(self):
        """Build runtime sampler options for hardware execution."""

        sampler_options = self.sampler_options.copy() if self.sampler_options else {}

        dd_options = self._normalize_option_block(self.dynamical_decoupling, "enable")
        if dd_options:
            merged_dd = sampler_options.get("dynamical_decoupling", {}).copy()
            merged_dd.update(dd_options)
            sampler_options["dynamical_decoupling"] = merged_dd

        twirling_options = self._normalize_option_block(self.twirling, "enable_gates")
        if self.gate_twirling is not None:
            twirling_options.update(self._normalize_option_block(self.gate_twirling, "enable_gates"))
        if self.measurement_twirling is not None:
            twirling_options.update(self._normalize_option_block(self.measurement_twirling, "enable_measure"))
        if twirling_options:
            merged_twirling = sampler_options.get("twirling", {}).copy()
            merged_twirling.update(twirling_options)
            sampler_options["twirling"] = merged_twirling

        return sampler_options or None

    # --------------------------------------------------
    # Build quantum circuit
    # --------------------------------------------------
    def build(self):
        mol = self.molecule

        norb = mol.n_active_mos
        nelec_a, nelec_b=mol.n_active_ab_electrons
        self.nelec_a = nelec_a
        self.nelec_b = nelec_b

        qr = QuantumRegister(2 * norb)
        qc = QuantumCircuit(qr)

        self._append_sampling_ansatz(qc, qr, norb, nelec_a, nelec_b)

        qc.measure_all()

        if self.use_simulator:
            if self.backend_obj is not None:
                backend = self.backend_obj
            else:
                if AerSimulator is None:
                    raise RuntimeError("qiskit_aer is required for aer_simulator backend.")
                backend = AerSimulator(method=self.sim_method)

            # For AerSimulator, decompose ffsim custom instructions and skip
            # full transpilation to avoid HLS issues with measurement.
            qc_pre = ffsim.qiskit.PRE_INIT.run(qc)
            # Decompose ffsim gates (e.g., slater_jw) into standard gates.
            qc_decomp = qc_pre.decompose(reps=10)
            self.circuit = qc_decomp
            self.backend = backend
        else:
            if self.service is None:
                raise RuntimeError("QiskitRuntimeService is required for hardware backends.")

            backend = self.service.backend(self.backend_name)

            initial_layout, _ = get_zigzag_physical_layout(norb, backend)

            pm = generate_preset_pass_manager(
                backend=backend,
                optimization_level=3,
                initial_layout=initial_layout
            )
            pm.pre_init = ffsim.qiskit.PRE_INIT

            self.circuit = pm.run(qc)
            self.backend = backend
        self._built = True


    # --------------------------------------------------
    # Run SQD
    # --------------------------------------------------
    def simulate(self):
        if not self._built:
            raise RuntimeError("Call build() before simulate().")

        if self.use_simulator:
            job = self.backend.run(self.circuit, shots=self.shots)
            result = job.result()
            counts = result.get_counts()
        else:
            if RuntimeSampler is None:
                raise RuntimeError("qiskit_ibm_runtime is required for hardware backends.")
            
            # Retry logic for transient IBM hardware/service failures
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    # Refresh backend ONLY if we are retrying after a failure (to save counts)
                    if attempt > 0 and hasattr(self.service, "backend"):
                        try:
                            self.backend = self.service.backend(self.backend_name)
                        except Exception as be:
                            print(f"[SQDSolver] Warning: Could not refresh backend from service: {be}")

                    sampler = RuntimeSampler(mode=self.backend, options=self._get_runtime_sampler_options())
                    job = sampler.run([self.circuit], shots=self.shots)
                    result = job.result()
                    counts = result[0].data.meas.get_counts()
                    break # Success!
                except Exception as e:
                    if attempt < max_retries - 1:
                        import time
                        print(f"\n[Warning] SQD Job failed with error: {e}")
                        print(f"Waiting 30 seconds and retrying (Attempt {attempt+2}/{max_retries})...")
                        time.sleep(30)
                    else:
                        print("\n[Error] SQD Job failed after maximum retries.")
                        raise e

        bs_mat, probs = counts_to_arrays(counts)
        if bs_mat.size == 0:
            raise RuntimeError("SQD: No samples returned from backend (empty counts).")
        if self.diagnostics:
            first_bs = bs_mat[0]
            n_bits = first_bs.shape[0]
            half = n_bits // 2
            ham_left = int(first_bs[:half].sum())
            ham_right = int(first_bs[half:].sum())
            print(
                "SQD counts diag: "
                f"n_bits={n_bits}, first_bitstring={first_bs.astype(int).tolist()}, "
                f"hamming_left={ham_left}, hamming_right={ham_right}"
            )

        hcore = self.molecule.one_ele
        eri = self.molecule.two_ele
        # Ensure SQD operates on the active-space integrals aligned with the sampled bitstrings.
        active = getattr(self.molecule, "active_mos", None)
        if active is not None and hcore.shape[0] != len(active):
            hcore = hcore[np.ix_(active, active)]
            eri = eri[np.ix_(active, active, active, active)]
        e_nuc =float(self.molecule.mean_field.mol.energy_nuc())

        avg_occ = None
        rng = np.random.default_rng(123)

        best_states = None

        for it in range(self.max_iterations):

            if avg_occ is None:
                bs_tmp, p_tmp = bs_mat, probs
            else:
                bs_tmp, p_tmp = recover_configurations(
                    bs_mat, probs, avg_occ,
                    self.nelec_a,
                    self.nelec_b,
                    rand_seed=rng
                )

            batches = postselect_and_subsample(
                bs_tmp,
                p_tmp,
                hamming_right=self.nelec_a,
                hamming_left=self.nelec_b,
                samples_per_batch=self.samples_per_batch,
                num_batches=self.n_batches,
                rand_seed=rng
            )
            # Diagnostics: catch empty batches early (likely electron-count mismatch after postselection)
            batch_sizes = [len(b) for b in batches]
            if self.diagnostics:
                total_samples = len(bs_tmp)
                total_postselected = sum(batch_sizes)
                print(
                    "SQD postselect diag: "
                    f"total_samples={total_samples}, total_postselected={total_postselected}, "
                    f"batch_sizes={batch_sizes}"
                )
                if batch_sizes and batch_sizes[0] > 0:
                    b0 = batches[0]
                    b0_first = b0[0]
                    n_bits = b0_first.shape[0]
                    half = n_bits // 2
                    ham_left = int(b0_first[:half].sum())
                    ham_right = int(b0_first[half:].sum())
                    print(
                        "SQD batch diag: "
                        f"batch0_shape={b0.shape}, first_bitstring={b0_first.astype(int).tolist()}, "
                        f"hamming_left={ham_left}, hamming_right={ham_right}"
                    )
            if any(sz == 0 for sz in batch_sizes):
                raise RuntimeError(
                    "SQD: postselection yielded empty batch(es). "
                    f"nelec_a={self.nelec_a}, nelec_b={self.nelec_b}, "
                    f"batch_sizes={batch_sizes}"
                )

            energies = []
            occs = []

            for b in range(self.n_batches):
                e, sci_states, avg_o, _ = solve_fermion(
                    batches[b],
                    hcore,
                    eri,
                    open_shell=self.open_shell,
                    spin_sq=self.spin_sq
                )
                if self.diagnostics:
                    try:
                        dm1 = sci_states.rdm(rank=1, spin_summed=True)
                        dm1_trace = float(np.trace(dm1))
                    except Exception as exc:
                        dm1_trace = f"rdm_error: {exc}"
                    amp_norm = float(np.linalg.norm(sci_states.amplitudes))
                    print(
                        "SQD sci diag: "
                        f"batch={b}, norb={sci_states.norb}, nelec={sci_states.nelec}, "
                        f"amp_norm={amp_norm}, trace(rdm1)={dm1_trace}"
                    )
                energies.append(e + e_nuc)
                occs.append(avg_o)

            avg_occ = tuple(np.mean(occs, axis=0))
            best_states = sci_states

            if it > 0 and abs(np.mean(energies) - self.energy) < self.tol:
                break

            self.energy = np.mean(energies)

        self.sci_states = best_states
        self._simulated = True

    # --------------------------------------------------
    # RDMs for DMET
    # --------------------------------------------------
    #def get_rdm(self, resample=False):
    #    if not self._simulated:
    #        raise RuntimeError("Call simulate() before get_rdm().")

    #    norb = self.molecule.n_active_mos

    #    # Build CI vector
    #    coeffs = np.array([c for _, c in self.sci_states])
    #    dets = [d for d, _ in self.sci_states]

    #    onerdm = ffsim.rdms.one_rdm_from_ci(dets, coeffs, norb)
    #    twordm = ffsim.rdms.two_rdm_from_ci(dets, coeffs, norb)

    #    return onerdm, twordm

    def get_rdm(self, resample =False):
        if not self._simulated:
            raise RuntimeError("call simulate() before get_rdm().")
        sci_state = self.sci_states
        onerdm = sci_state.rdm(rank=1, spin_summed = True)
        twordm = sci_state.rdm(rank=2, spin_summed = True)

        # Normalize RDM shapes to the active spatial-orbital space expected by DMET.
        n_active = self.molecule.n_active_mos
        n_mos = getattr(self.molecule, "n_mos", n_active)

        def _spinorb_to_spatial(d1, d2, norb):
            d1_sp = d1[0::2, 0::2] + d1[1::2, 1::2]
            d2_sp = np.zeros((norb, norb, norb, norb), dtype=d2.dtype)
            for sigma in (0, 1):
                for tau in (0, 1):
                    d2_sp += d2[sigma::2, tau::2, sigma::2, tau::2]
            return d1_sp, d2_sp

        if onerdm.shape[0] == 2 * n_mos:
            onerdm, twordm = _spinorb_to_spatial(onerdm, twordm, n_mos)
        elif onerdm.shape[0] == 2 * n_active:
            onerdm, twordm = _spinorb_to_spatial(onerdm, twordm, n_active)

        if onerdm.shape[0] != n_active and hasattr(self.molecule, "active_mos"):
            idx = self.molecule.active_mos
            onerdm = onerdm[np.ix_(idx, idx)]
            twordm = twordm[np.ix_(idx, idx, idx, idx)]

        return onerdm, twordm 
