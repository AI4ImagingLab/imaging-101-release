# Provenance and validation

Status: implementation and validation in progress; not accepted for release until all gates below are passed.

## Binding and distinctness

The bound observations are the complete author-distributed TCV discharge 85270:190 native time samples,120 detector channels,113 independently designated good channels. The inverse unknown is a120×41 asymmetric poloidal emissivity field at every native time. The closest old binding, abel_plasma_emission, was read in full: it instead uses a single camera projection with cylindrical symmetry and independent radial Abel inversions. The new multiview finite-collection-volume acquisition does not impose that poloidal symmetry. Covercrop electromagnetic conductivity and hydraulic injection conductivity README files were also read when screening the separate neutron-water candidate; they do not duplicate this task.

Data classification: **author_real_processed**. Primary paper arXiv2603.11856v1, dated12March2026, section4 identifies the50 measured TCV discharges from the first half of2025; section2 describes the120 carbon-coated foil bolometers in five cameras. The repository supplies shot85270 as shot_data_85270.mat. The native data are processed RADCAM signals, not generated observations. The numeric conversion to NPZ preserves every stored observation and calibration number exactly; it does not claim access to raw ADC voltages.

There is **no physical ground truth**. evaluation/baseline_reference.npz contains reconstruction shape(1,190,120,41), float64, independently produced by unchanged author offline MAP code. It is not renamed truth. Magnetic-equilibrium arrays come from separate magnetic measurements or pre-discharge plans; no radiation reconstruction is included in public input.

## Sources, revisions and licenses

- Paper: https://arxiv.org/abs/2603.11856v1; private tcv_paper_v1.html. Preprint status is retained; no unsupported journal-publication claim.
- Main author source: https://github.com/dhamm97/real-time-tomo-prad, remote HEAD04fde81ce85cdbdf327652fdeb0c20b0903b226e verified with git ls-remote on2026-10-27. Selected files downloaded by this exact commit. MIT, copyright2026dhamm97; LICENSE retained.
- Diffusion source: https://github.com/dhamm97/pyxu-diffops, remote treeSHA b46f8ba783918d0d6d172b24ab694b919e5ae089 verified from GitHub API on2026-10-27 after a git transport failure. Exact-commit files downloaded; MIT copyright2024DanieleHamm preserved in NOTICE. This is a verified remote tree/revision, not an assertion that the repository will remain unchanged.
- General original optimization framework: Pyxu2.0.2. The local solver explicitly implements its Chambolle-Dossal update and stopping logic. Cleaned runtime imports neither framework nor author packages.
- Measurements are openly distributed with the author repository; no separate per-file measurement license was found. This limitation is retained in NOTICE rather than inventing an additional license.

## Private archive index

All private paths below are relative to the maintainer batch's astro_earth_physics directory, outside the final task. tcv_archive/ holds selectively downloaded unchanged author source, shot85270 and geometric calibration; diffops_archive/ holds unchanged diffusion source. tcv_tree.json and diffops_tree.json retain remote tree metadata. tcv_full_manifest.json records source SHA256s and exact execution environment. tcv_oracle_environment.txt records installed oracle versions. tcv_oracle_probe.py and tcv_oracle_full.py are private capture scripts; tcv_probe.log and tcv_full.log retain execution outputs. tcv_probe.npz and tcv_full_oracle.npz are private independent numerical captures. tcv_frozen_acceptance.json fixes tolerances before extraction. package_tcv_data.py records lossless conversion. build_tcv_notebook.py authors the review notebook. No archive script or oracle package is imported by the final task.

## Original execution and test-first sequence

Private command: oracle-env/Scripts/python.exe tcv_oracle_full.py. Original main and diffusion files are unchanged. The script invokes the same offline functions and original optimizer as compute_baseline_tomographic_inversion, using the same per-frame normalization, measured nearest-time magnetic fields, parameters, support, initialization and stopping. All 190native frames completed in103.48285939998459s with exit0. The preliminary independent first-frame run completed in8.158438499900512s including setup. A capture-script shape mistake in optional objective recording was corrected before the successful full run; no original numerical source was patched.

Windows, Python3.11.11, NumPy2.0.2, SciPy1.17.1, Pyxu2.0.2; single-thread OMP/MKL/OpenBLAS/Numba. Cleaned runtime uses Python3.11.11, NumPy2.4.6, SciPy1.17.1, Matplotlib3.11.1. Domain dependencies exist only in the isolated private oracle environment. Local runtime uses explicit numerical mathematics. Resource probe before execution:13.87GiB physical RAM free,27.67GiB virtual free and41.81GiB disk free; acquisitions were inspected for file sizes before selective download.

Before local implementation, original full results and intermediate fixtures were saved and acceptance was frozen. First-frame independent rerun maximum absolute difference was1.1641532182693481e-9W/m³ against global reference dynamic range3758557.138069409W/m³. Original maximum frame-normalized forward RMS was0.030576372875822475. Frozen Judge bounds: NCC≥0.999999, dynamic-range NRMSE≤1e-6, maximum frame forward RMS≤0.035, outside-support maximum exactly0. No boundary was selected or relaxed after examining local results. Parity requires full-field relative L2 error<1e-7, tighter than Judge acceptance. Tensor absolute tolerance1e-10; computed gradients rtol1e-9/atol1e-7; first3 updates rtol1e-9/atol1e-10; unchanged acquisition arrays exact equality.

## Equation/function/test traceability

| Paper stage | Original function | Local function | Independent validation |
|---|---|---|---|
| Eq1 acquisition and appendix normalization | RtPradROI.preprocess_bolo_data | preprocessing.prepare; physics_model.forward/adjoint | Captured geometry and observations; adjoint dot-product identity |
| Magnetic field interpolation | RtPradROI.preprocess_magnetic_data | interpolate_equilibria | All 190interpolated grids compared exactly |
| Eq4 Gaussian likelihood | _DataFidelityFunctional | loss_gradient | Original first-frame loss/gradient; independent finite differences |
| Eq5 diffusion tensor | AnisDiffusionOp, DiffusionCoeffAnisotropic | diffusion_tensor | All 190tensor fields vs upstream; eigenvalues 0.01/1 independently checked |
| Eq5 finite differences | _Diffusion._assemble_matrix_based | gradient_matrix; diffusion_matrix | Original gradient fixture; constants/linear coordinates/PSD checks |
| Eq3/Eq6 constrained offline MAP | compute_MAP; PyxuPGD.m_step | reconstruct_frame | Original initialization and3 updates; full 190-frame parity |
| Eq9 toroidal integration | compute_baseline_tomographic_inversion | integrate_power | Analytic constant-emissivity cylindrical-shell volume |
| Native full pipeline | compute_MAP for each native frame | reconstruct; main | Complete full-pipeline test and active notebook |
| Eq12 realtime implementation discrepancy | RtPradROI.assemble_gradient_matrix | No validated replacement claimed | Archived original constant derivative vs fresh offline derivative in notebook |

## Investigated discrepancies and scientific limitations

The realtime reduced-vessel code stores horizontal derivatives at row 2*counter while vertical derivatives use row counter. These overlap instead of occupying independent component blocks. Direct execution gives nonzero derivatives 80m^-1 at two rows for an exactly constant field; a no-flux gradient must annihilate constants. The defect is independently established, not inferred from agreement between copied formulas. The root approved selecting the separately published offline MAP comparison before local extraction. No realtime estimate is represented as a validated fresh reconstruction. The notebook preserves the failure diagnostic and fresh planned-versus-measured equilibrium comparison without fabricated method images.

The selected offline gradient uses a rectangular full grid and support projection, penalizing vessel-edge gradients; it is not the separate realtime no-flux vessel operator. Its mean-spacing convention h=(dR+dZ)/2 is preserved. Negative emissivity can occur under the Gaussian estimator and is not silently clipped. Auxiliary magnetic flux is uncertain, and default noise scaling is a method convention rather than independently measured detector variance. Sparse projections and method agreement cannot certify local physical accuracy.

## Fixture and result consumers

test/fixtures/upstream_stages.npz contains original constructor-independent numeric fixtures named output_*, including preprocessing, all diffusion tensors, first-frame losses/gradients, initial backprojection,3 updates and per-frame objective/residual measurements; input_mask is the original vessel support. test/comparing_test/test_parity.py consumes them. The full original array appears once in evaluation/baseline_reference.npz and is consumed by the Judge, parity test and review notebook. test/fixtures/upstream_rt_gradient_diagnostic.npz contains the original constant-field derivative and is consumed only by the explicit notebook diagnostic. notebooks/reference_outputs/upstream_measurements.npz contains independently reproduced upstream times/objectives/residuals; notebooks/reference_outputs/fresh_measurements.npz contains fresh notebook runtime, metrics, objective trace and selected-frame comparison powers. Neither file is truth; neither supplies a fresh image.

Historical paper claims are context only. Machine-readable metrics retain full numeric precision. Current upstream, fresh local and independent mathematical checks are distinguished. The Judge, visualization metric implementation and notebook use the same known analytic metric case in tests.

## Gates

- Passed: primary paper, exact remotely verified revisions, measured data identity, equation/function/test mapping and source license notices.
- Passed: unchanged original offline method on all exact bundled native frames; frozen numerical intermediates and acceptance before local extraction.
- Passed: initial stage parity and full 190-frame parity under the clean shared environment; original first-frame local relativeL2 error4.75e-15 and3-update maximum difference2.22e-16.
- Passed: `python -m src.main` and standalone Judge CLI; all 15 tests passed in 115.05 s (`tcv_tests_retry.log`). After adding independent visualization metric agreement, all 13 unit/contract tests passed in 1.42 s (`tcv_unit_final.log`). Complete notebook execution passed (`tcv_nbconvert_final.log`), including all 190 frames and three fresh selected-frame ablations. Seven figures were visually inspected; revised raw/good-channel, common-scale signed reconstruction and common-axis ablation figures were re-inspected in `figures_final/`.
- Passed: public README/NPZ key, shape, dtype and acquisition-constant audit; GitHub URL and hidden-answer leak scan; local import/call-role audit; plans/signatures and objective factors, units, gradient adjoints, normalization and execution flow checked against source. Final task inventory excludes upstream code, PDFs, capture scripts, caches and temporary executed notebooks.
- Explicitly failed excluded method: original realtime reduced-vessel gradient physics check. This does not alter the selected offline method; do not claim realtime reproduction.

## SHA256 identities

| Asset | SHA256 |
|---|---|
| tcv_archive\src\data\shot_data_85270.mat | 0ab6dc9dea0c0e30fd042418aa75d907a156b976c3f7c0642b2b8f12430d990a |
| tcv_archive\src\tcv_geometry\geometry_matrix_NINO.npy | 24502d6ff155f9557d08edf3497b39d4d09f700a100d699d5559ab5d832150e1 |
| tcv_archive\src\tcv_geometry\etendues.npy | 239e02c8495e3c03207f20b13e063b97aab0914a925560e42ebd522e1e74c14c |
| staging\tcv_bolometric_emissivity\data\raw_data.npz | 8e9c155d93b05c79f6b8f7a8fe667ada825c3d7cd8b578e3752f8f9a730f48fd |
| staging\tcv_bolometric_emissivity\evaluation\baseline_reference.npz | 744f264753b56777d688c45057cd0bd48ee73b452754b42a15e2e1fc2c08af7c |
| staging\tcv_bolometric_emissivity\test\fixtures\upstream_stages.npz | 9be7c7633471b50f5057329639c343b35834ecf261fcf5ec61d7cfe94eb67b6b |
| tcv_full_oracle.npz | 6026e3b4c1a0b365794a5c798f558d86d464a2966efe919e366b5c9e18581776 |
| tcv_frozen_acceptance.json | 154d6d04c8b1491fafeda9feb5ee1a11297e9995b71bb341feca59051632eaf0 |

## Final execution and review evidence

The final standalone Judge reports NCC 1.0000000000000004 (floating-point roundoff), dynamic-range NRMSE 2.531985758753277e-16, maximum normalized observation RMS 0.03057637287582246, and exactly zero exterior emission. All original boundaries remain frozen. These values quantify reproduction and consistency, not physical ground-truth accuracy. The notebook preserves negative estimates and explicitly labels each fresh ablation and original reference.

Commands were run with the shared Python, single-thread numerical backend, and task cwd: `python -m src.main`; `python evaluation/judge.py --prediction output/reconstruction.npz --report output/judge_report.json`; `python -m pytest test/ -v`; and `python -m nbconvert --to notebook --execute notebooks/tcv_bolometric_emissivity.ipynb --output tcv_bolometric_emissivity_final.ipynb --output-dir output --ExecutePreprocessor.timeout=600`. The notebook uses a private kernel pointing to the shared interpreter. Initial pytest failed because the Windows global temporary directory denied access; rerunning with writable private TEMP/TMP passed without algorithm changes. Initial notebook kernel launch failed on a UTF-8 BOM in private kernel configuration; correcting that private configuration passed. Windows event-loop and local kernel transport warnings remain recorded in logs and did not affect numerical execution.

The reviewed canonical notebook retains executed outputs. Temporary executed copies and Python caches were moved to the private archive. No original archive was deleted. Source imports are NumPy, SciPy general interpolation/sparse primitives, Matplotlib and standard library only. The Judge is independent of solver imports. No additional reconstruction/domain package is required.


## Independent parent acceptance

Canonical kernel metadata is portable python3; all executed cells, counts and scientific outputs are preserved from the private shared-runtime execution.

CLI, standalone Judge, pytest, active nbconvert and dual-engine math rendering were independently rerun: all exit0. Parent notebook: 6 executed code cells, 7 computed PNG outputs, 22 formulas. Private evidence: research_batch11_20261027/acceptance/tcv_bolometric_emissivity/execution.json and semantic_review.json. All panels actually inspected by parent; source/physics/GT roles and limitations semantically checked. Canonical notebook is this parent-executed artifact. No new source or notebook execution is inferred from cached upstream outputs.
