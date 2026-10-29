# Provenance and validation

Status: original and local mathematical gates passed; complete notebook/visual/final inventory pending. Do not count as complete until final records exist.

## Paper, source, exact observations

Larson,K.M. (2024), Gnssrefl: an open source software package in python for GNSS interferometric reflectometry applications, GPS Solutions28,165. DOI10.1007/s10291-024-01694-8, version of record27July2024; primary article https://link.springer.com/article/10.1007/s10291-024-01694-8 . Archived published PDF from German National Library https://d-nb.info/1344734308/34 . Author source https://github.com/kristinemlarson/gnssrefl remotely verified Git commit047f9f53ff5cf762a2cab9abd908e347c38dd632; private git checkout and original numerical source retained. Version4.2.3, GPL3 code/LICENSE preserved. Data are author-supplied measured receiver SNR examples, not synthesized; no separate data-license grant is invented beyond repository provenance.

Complete native MCHL station2025DOY10,11,12 test/data/refl_code receiver files are archived privately. Real data binding is supported by station acquisition timestamps, satellite/elevation/azimuth/SNR channels, author use_mchl documentation and exact repository regression workflow. Independent antenna location/meteorological refraction parameters are author acquisition calibration. Apriori_rh is not a target field; selected standard Bennett refraction does not infer this benchmark's heights from it. No water-content calibration or reconstructed phaseRH is bundled. Visible arrays are explicitly author_real_processed:324 full arcs after the original fixed quality control, polynomial direct-signal removal and independent refraction correction. Two of326 candidate spectra fail original post-spectrum QC and remain private diagnostics, never removed according to local results.

## Original execution and capture before extraction

Private run_gnss_original.py executes author gnssir on every day and all configured frequencies1,20,5. Author default fast approximate Astropy NFFT branch ran successfully with full outputs in gnss_oracle; then the supported original scipy branch was selected before any local implementation and run completely in gnss_oracle_scipy. Independent repeated scipy outputs are bit exact:103,111,110 accepted rows, total324. Evidence gnss_first_execution.json,gnss_scipy_execution.json,gnss_scipy_repeat_agreement.json,matching stdout/stderr; no cached regression outputs substitute for execution. The source was run under private oracle-env Python3.11,NumPy2.0.2,SciPy1.17.1,Astropy8.0.2,one numerical thread on Windows. Shared runtime is Python3.11,NumPy2.4.6,Matplotlib3.11.1; no package upgrade was made. Only missing tqdm4.67.1 was added in the isolated original environment.

Windows lacks a standalone mkdir executable. A private subprocess-call compatibility shim handles mkdir through pathlib; no numerical or acquisition functions were modified. Missing neighboring days9/13 are original warnings: no artificial boundary records were generated. A read-only Python trace captures all326 bound LSP inputs/summaries, six full spectral stages and each day's refraction arrays. No capture script or upstream import is shipped in runtime. package_gnss.py matches each of324 author accepted arc files exactly to its original spectral-call inputs and evaluates both original supported branches on these exact observations. gnss_binding.json records original pickle hashes, native call indices and public NPZ hash.

gnss_frozen_acceptance.json predates all local mathematical source files: full-height max absolute discrepancy1e-7m; spectral amplitude rtol1e-8/atol1e-9. These are extraction tolerances, not physical resolution. The grid is about0.005m and no independently measured height truth exists. Float64 is explicitly the output dtype. No threshold was relaxed after local results.

## Traceability

| Stage | Original function | Local | Independent case |
|---|---|---|---|
| Refraction/direct trend/QC preprocessing | extract_arcs.apply_refraction and original arc extraction | Already-applied author observation processing, explicitly declared public input | Complete original native-file run and frozen refraction/arc captures |
| Carrier/elevation coordinate | gps.strip_compute | solvers.reconstruct | Six captured coordinate/response spectra |
| Trial-height discretization | gps.get_ofac_hifac, freq_out | physics_model.spectrum | Exact original linspace including floating1599/1600point distinction |
| Harmonic inverse | gps.strip_compute scipy branch | Explicit cosine/sine Gram inverse in spectrum | Six full original spectra plus analytic irregular-sample sinusoid |
| Dominant peak | gps.strip_compute argmax | solvers.reconstruct | Complete324 original outputs |
| Physical forward | Harmonic GNSS interference equation | physics_model.forward and notebook fitted amplitudes | Known sine/cosine phase conventions |
| Evaluation | Independent numerical contract | evaluation/judge.py | Poor/malformed/GTpriority/reportwrite tests |

The default fast NFFT branch is an approximation and can choose different grid peaks. Its independent outputs remain identified, not relabeled as local exact measurements. Generic SciPy LombScargle is used only in the original oracle; the cleaned inverse explicitly evaluates normal equations with NumPy. No cross-task import, hidden target calibration, domain reconstruction package or backend mutation occurs.

## Reference consumers and gates

evaluation/baseline_reference.npz is the original full324 height result, not truth. notebooks/reference_outputs/upstream.npz contains original direct/fast comparisons consumed only by the review notebook. test/fixtures/original_stages.npz contains independently captured six complete original numerical stages, consumed by comparing_test/test_original.py. No fixture is generated by local code. Tests12 passed, including all324 fresh heights, six original spectra, synthetic analytic harmonic (test only), strict missing/key/shape/dtype/NaN/Inf failures, poor output, truth precedence/no corrupted-truth fallback, arbitrary cwd and report-write behavior. Shared source main writes required NPZ. Notebook and final source/document consistency review pending.

## Archived hashes

Private root: research_batch11_20261027/astro_earth_physics relative to maintainer directory. Complete checkouts, source paper, receiver archives, original processed pickles, capture code and execution logs stay outside this task. Additional archive consumers/index are in gnss_binding.json and gnss_archive_index.json.

| Archive path | SHA256 |
|---|---|
| `gnss_paper.pdf` | `7999866e21d91de21812264090d9b59114b184dfca3a5c6f26cc22512c01074e` |
| `gnss_scipy_stages.npz` | `91bbd735836773d69c6faf2f81c000ebd6bab3990795351d435fdd8977143b8d` |
| `gnss_original_stages.npz` | `e5136c93a61b73b1c1f893e76a7d8dd4adb2e8a416e99091d4e89e0bb51ec315` |
| `gnssrefl_archive/gnssrefl/gps.py` | `400fd65fe53a8ea9b7083c8699907a79b36d847efc5fd2ea58fc32d63efb31ed` |
| `gnssrefl_archive/test/data/refl_code/2025/snr/mchl/mchl0100.25.snr66.gz` | `0f72aa1b87948c050666d5a0d77d8b2dd5c879f9e5e6b1342ff2f799eab6ce30` |
| `gnssrefl_archive/test/data/refl_code/2025/snr/mchl/mchl0110.25.snr66.gz` | `767b74c7d70fe5c8cdd1a53c0996858ad9731050b0fc10511c2e40a72feeecd0` |
| `gnssrefl_archive/test/data/refl_code/2025/snr/mchl/mchl0120.25.snr66.gz` | `477f34a05184151fcd54adbf0de5c7838d233fe3aab89877fba52f3abc5ab565` |
| `gnssrefl_archive/test/data/refl_code/input/mchl/mchl.json` | `061176b7b35636be645c76414ff4e01c4bc27236e36f8df156423e246b83fb3e` |
| `gnssrefl_archive/test/data/refl_code/input/mchl_refr.txt` | `64de9f3a95a480b0373de8c937d5b8f02cd5c4ab5a42aa955475fd394ae62314` |

## Final worker execution

Shared CPU main/Judge exit0;12 tests pass in1.51s, all324 fresh heights agree exactly. Native nbconvert complete9.73s,5activecodecells,4actualPNGfigures; entire selected acquisition and3freshsensitivity reconstructions plus324forwardresidualfits actively computed. All four figures inspected individually; one alias switch on sample thinning is retained and discussed. Canonical executed outputs preserved with portable python3 kernel metadata. No temporary executed notebook remains in task output; private gnss_executed.ipynb and gnss_figures hold capture evidence. Earlier pending local gates resolved; independent parent acceptance remains pending.


## Independent parent acceptance

Canonical kernel metadata is portable python3; all executed cells, counts and scientific outputs are preserved from the private shared-runtime execution.

CLI, standalone Judge, pytest, active nbconvert and dual-engine math rendering were independently rerun: all exit0. Parent notebook: 5 executed code cells, 4 computed PNG outputs, 2 formulas. Private evidence: research_batch11_20261027/acceptance/mitchell_gnss_reflector_height/execution.json and semantic_review.json. All panels actually inspected by parent; source/physics/GT roles and limitations semantically checked. Canonical notebook is this parent-executed artifact. No new source or notebook execution is inferred from cached upstream outputs.
