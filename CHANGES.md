# Changes from upstream Tangelo

This repository is a modified copy of Tangelo
(https://github.com/goodchemistryco/Tangelo, Apache License 2.0;
Senicourt et al., arXiv:2206.12424), copied on <date>.
Upstream copyright notices are retained. Only the files below differ from
upstream (paths relative to `tangelo/` unless noted).

## Added
| File | Purpose |
|---|---|
| `algorithms/sqd_solver.py` | New SQD solver: LUCJ sampling circuit (ffsim), simulator or IBM Runtime execution with dynamical decoupling and twirling options, zigzag hardware layout, configuration recovery and subsampling, subspace diagonalization via qiskit-addon-sqd, frozen-orbital active-space integrals via OpenFermion, active-space RDMs returned to the DMET loop |
| `algorithms/tests/` | Tests for the SQD solver |
| `toolboxes/quantum_layouts/` | Hardware qubit-layout data (zigzag layouts) for the IBM device |

## Modified
| File | Change |
|---|---|
| `problem_decomposition/dmet/dmet_problem_decomposition.py` | Added `"sqd"` fragment solver; added a noise-robust chemical-potential root finder (median of repeated residuals, geometric bracketing, Brent, damped fallback), used by default when any fragment solver is SQD; store and report the final electron residual; support for reusing a cached converged result in the final loop; added diagnostics; changed the Newton tolerance for non-SQD runs from 1e-5 to 1e-4 |
| `problem_decomposition/tests/dmet/test_dmet.py` | Added tests for SQD solver wiring (mock solver), Newton use for non-SQD runs, and the noisy root finder |

## Not changed
All other Tangelo modules, including `frozen_orbitals.py` and `rdms.py`
(frozen-orbital handling and RDM padding are upstream code), are unchanged.

## Study scripts (repository root)
- `tpp_CO2_relaxed_scan.py`: constrained N–C scan; CCSD fragments plus SQD on IBM hardware. It uses its own noisy chemical-potential optimizer passed through the `optimizer` option. IBM credentials are read from environment variables.
- `run_dmet_sqd.py`: <one-line purpose>
- `run_dmet_sqd_tpp_monomer.py`, `run_dmet_sqd_co2_monomer.py`: monomer reference energies.
