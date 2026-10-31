# Provenance, corrected oracle and completed validation

This is a real measured-observation to **spatial two-component current-field inverse**, not calibration fitting. Actual current truth is unavailable. The hidden standard-method reference is the independently run, explicitly minimally corrected modern Fourier Biot–Savart implementation. Original untouched output is retained separately and is not relabeled as true current or unchanged corrected reproduction.

## Sources and rights

Private batch root B4 is `D:/yan1/agent/AutoSkill/imaging-101-maintainer/research_batch4_20261021`; P denotes `B4/bio_med/squid`, S denotes `B4/material_physics/current`. All PDFs, source, original logs/capture scripts and figure files are private outside this task. P/manifest.json contains exact SHA256s and archive paths; B4/bio_med/report_catalog.json provides group navigation.

| Source | Verified record |
|---|---|
| Method paper | Bradley J.Roth,Nestor G.Sepulveda,John P.Wikswo Jr., *Using a magnetometer to image a two-dimensional current distribution*, Journal of Applied Physics65(1),361–372(1989), https://doi.org/10.1063/1.342549. Received16June1988,accepted13September1988; this is the12-page journal article, not an abstract or later review. |
| Full paper acquisition | The old source link https://www.vanderbilt.edu/lsp/documents/jap-roth-using-89.pdf currently returns404. Parent found the current author-institution publication list https://www.vanderbilt.edu/lsp/publications_90-85/ and its migrated PDF https://www.vanderbilt.edu/lsp/wp-content/uploads/sites/154/2026/08/jap-roth-using-89.pdf. This downloaded successfully as P/roth1989.pdf(1082554bytes); exact URL/hash in P/paper_source.json. It is a scanned PDF: text extraction alone is empty. All12embedded page images were extracted into P/paper_pages and visually read, including equations and scientific result groups. Failed ResearchGate author-upload and publisher attempts are honestly preserved in P/paper_downloads.json, not treated as full text. |
| Modern source | https://github.com/loganbvh/current-reconstruction exact **remotely verified** commit `443b3a1c8efa3f9a37c1be5ce01e2e2da84d0aa0`, dated20June2024. The remote API response is B4/material_physics/current_commit.json, copied/indexed privately. Exact mathematical source is S/current_reconstruction/current_reconstruction.py, SHA256 `8996a725b36eb386600fff0900114dbf7f524601a10c06734d6fe4463aebb8b7`. This is modern reference code, not software provided by the1989 authors. |
| Existing measured observations | S/sample_data/mag.npz, deposited flux151×201 with xs,ys,standoff and units. Source README states scanning SQUID lock-in imaging at approximately600Hz,1microampere AC. No device material/anatomy is inferred or invented. All measured pixels are retained, no new benchmark measurements/noise generated. |
| Independent calibration | S/sample_data/psf.npz, measured209×209out-of-plane magnetic-dipole scan in mPhi0. Unknown dipole moment precludes absolute current units. The publicly supplied301×401coarse calibration is the actual original pad_to_match/resample output captured independently of target reconstruction. Native scan and all calibration axes remain public. |
| License | Complete MIT license Copyright(c)2024Logan Bishop-Van Horn retained in LICENSE; adaptation/source/data scope in NOTICE. Source repository includes these sample measurements and no separate restrictive data notice was found. Paper copyright is not presented as MIT; its PDF/images remain private research evidence. |

## Oracle-first sequence and frozen contract

1. The prior original command `python B4/material_physics/capture_current.py` actually ran untouched source on the exact deposited observation and dipole calibration; its captured B4/material_physics/current_oracle.npz is retained. It reported33.92s. No source-format or cached-image substitute was accepted.
2. Before local task source existed, `B4/root/oracle-env/Scripts/python.exe P/oracle_capture.py` independently executed the unchanged selected source again, including original fine-grid padding/cubic-spline preparation. Runtime22.104827600000135s. Its cropped components are **bit identical** to the earlier original oracle. Source spectrum, coordinate, filter and operator locals were captured by Python trace hooks without changing arithmetic. The full complex field obtained from those frozen source spectra is P/unmodified_original.npz. P/oracle_capture.log and oracle_record.json identify the run.
3. Independent plane-wave and centered-delta experiments P/independent_preimplementation.json were recorded **before the corrected oracle and before local source**. Their exact odd-grid bin/origin discrepancies motivated only the three disclosed corrections below. P/minimal_correction.patch records every change; P/corrected_original.py is private modified original code.
4. The minimally corrected original ran all three cutoff values0.5,1,1.5with the exact bound flux and exact original coarse calibration. Actual per-run times and stages are P/oracle_record.json and corrected_cutoff_*.npz. A second corrected nominal run was **bit identical**. Reusing the independently captured original coarse calibration avoids large repeated interpolation allocations; it does not change its pixel values or substitute a local calibration model. Original native calibration preparation was already actually executed in step2.
5. Before local extraction, P/oracle_record.json froze full output float64(1,2,301,401), component orderJx,Jy, mean-zero gauge, original source orientation/scale, nominal window, separate component numerical tolerances and physical ratios. Parent approved the full-field contract; observed-region cropping is only derived visualization. A full field keeps the periodic forward model closed rather than silently discarding outside currents. No actual current truth is available or manufactured.
6. Tests and fixtures were written first by P/build.py. P/preimplementation_red.log records the expected missing-local-module collection failure. Only then were local operators/inverse written. The first complete suite passed24tests; an independent visualization metric test and canonical-boundary-consumption contract test were added during review, and the final suite passes26. Judge actually consumes the seven top-level metric_boundary entries; a test changes a top-level boundary and proves the reported score decision changes, then removes it and checks evaluator error. Values retain the preimplementation criteria unchanged.
7. P/capture_direct_trace.py additionally traced the **actual jx,jy complex local variables before the original crop.real lines**, for both untouched and corrected source with the exact already captured calibration. P/direct_untouched_full_trace.npz and direct_corrected_full_trace.npz are bit identical to the prelocal frozen spectrum-derived full fields. This confirms that uncropped reference values have the actual source execution meaning claimed; no local reconstruction function was used to manufacture fixtures.

## Independent investigation, signs and minimum changes

| Issue | Independent evidence | Explicit selected resolution |
|---|---|---|
| Odd grid frequency axis | For n=9,mode2unit-pitch plane wave, the original linspace bin differs from analytic4π/9by−π/9. Exact fftfreq assigns the correct bin. Source forces odd image lengths, so this affects the bound301×401case. | Replace only both linspace frequency arrays with2πfftshift(fftfreq). No data/method/cutoff change. |
| PSF array origin | A centered odd delta must have unit Fourier transfer. fftshift before FFT instead introduces a one-pixel phase ramp; ifftshift gives unit response. An even centered kernel has real transfer under the corrected origin. | Replace the inner PSF fftshift with ifftshift; preserve actual measured calibration values and center index. |
| DC nullspace | Correct bins contain exact K=0; uniform planar current cannot be inferred from out-of-plane field. Dividing0/0and hiding the result is not a gauge. | Initialize factor0and evaluate only K>0, fixing both current means0. No floor, fitted offset or artificial regularization. |
| Complex field hidden by crop.real | Untouched full Jx imaginary/real norm0.36005503696306407; Jy0.22321780000506614. Corrected maximum ratio8.543366540737229e−16. | Capture both untouched complex fields; local implementation rejects substantial imaginary residue before taking real. |
| Original signed response | Paper Eq6has positive-exponent FT; NumPy uses negative exponent. Direct right-hand Biot–Savart derivation gives +iμ0/2 e^(−Kh)P(k)(kxJy−kyJx)/K for +Bz, while modern source corresponds to its negative. Unknown dipole polarity/moment and lock-in phase do not establish absolute physical orientation. | Preserve source signed response explicitly as the benchmark definition, **without** an unsupported extra sign flip or fitted sign alignment. README states FFT,axes,effective equation and arbitrary signed scale. Analytic sinusoidal-current test verifies the declared negative-response convention and separately identifies standard+Bz. This is a convention-limited relative inverse, not proof of true absolute polarity. |
| Coarse PSF physical pitch | Original preparation yields dx1.2952720000000681um,dy1.1473366666666323um versus flux padded dx1.2923999999999864um,dy1.1440000000000055um. Relative differences0.222222%and0.291667%. Center measured-coordinate offsets are approximately(−0.0234,−0.053)um. | Preserve exact original sampled values as the supplied **effective discrete index kernel**, publish both native/coarse axes and quantify approximation. Do not claim perfect physical co-registration, fit the target, normalize kernel sum or reinterpolate to a new method. |

The correction is **not** advertised as unchanged reproduction. The unmodified source output remains available separately, and its real projection has filtered forward residual0.14181706413143685 and correct-grid divergence ratio0.01239514062190711. After correction, independent original full-spectrum filtered forward residual is1.9150318639202218e−16 and divergence2.0035609137400756e−17. These checks supplement, rather than replace, source parity.

The dipole scan is an empirical measured response in the source model. Unknown moment, dipole/sensor geometry and transfer approximation prevent an absolute coil-only kernel or absolute radiometric claim. Preserving a well-defined effective operator does not prove that all physical sensor/sample effects are exactly described by it.

## Equation/function/test traceability

| Mathematical role | Paper/reference source | Local implementation | Validation |
|---|---|---|---|
| Quasistatic divergence-free planar field | Paper Eq1,2,10,11; source reconstruct_current | physics_model.forward_spectrum, invert_spectrum | Independent analytic sinusoid, explicit signed +Bz comparison, zero/divergence tests |
| Finite periodic discrete frequency grid | Paper numerical implementation pp363; source k-space arrays | frequency_grid, preprocessing.pad_flux | Analytic mode and exact corrected original Kx/Ky/K and padded axes |
| Independent response convolution | Paper Eq14–17(ideal coils); modern source measured PSF FFT | psf_transfer | Center delta, even PSF, parity of actual complex P; measured empirical kernel distinction |
| Hanning spectral regularization | Paper Eq18; source hanning_2D | hanning | Exact original H on three cutoffs; radial support/DC tests |
| Closed-form transverse-current inverse | Source reconstruct_current factor,jx_k,jy_k | invert_spectrum, solvers.reconstruct | Original factor,spectra,full complex images and all3cutoff outputs |
| Imaginary/Hermitian correctness | Fourier transform of real current; original crop.real limitation | reconstruct imaginary gate | Independent even-kernel/mode checks and bound imag ratio |
| Physical closed forward and explicit gauge | Biot–Savart transverse operator and K0nullspace | diagnostics; separately Judge.physical_metrics | Analytic forward/inverse and all fresh full-field physical constraints |
| Current-reference metrics | Declared numerical method reference, not paper Eq19truth MSD | visualization.metrics; independent Judge.cosine/relative_l2 | Known analytic metric cases and component-specific quality tests |

Source extraction uses only readable local NumPy FFT/arithmetic and SciPy's physical constant. There are no domain-library, cross-task, opaque inversion, optimizer or neural-network calls. No new synthetic benchmark generator exists; analytic unit-test modes are small physics checks, not substituted observations. All signs, factors, FFT normalization, original window defaults, zero-ramp boundary and source numerical units are explicit in approach.md. Design signatures and actual data/caller roles were semantically reviewed against source.

## Reference, fixtures and measurement consumers

| Hidden asset | Meaning and consumer |
|---|---|
| evaluation/baseline_reference.npz | Minimally corrected original nominal full real current; numerical **standard-method reference**, not actual truth; Judge and parity tests. No ground_truth.npz is shipped. |
| test/fixtures/corrected_stages.npz | Prelocal original corrected mag_k,psf_k,frequencies,window,factor,component spectra,full complex fields,padded flux/axes,dx/dy; every retained key consumed by stage parity tests. |
| notebooks/reference_outputs/corrected_0.5,1,1.5.npz | Independently run corrected original full fields for all3cutoffs; cutoff tests and notebook posthoc parity. |
| notebooks/reference_outputs/unmodified_original.npz | Untouched source full complex current and old Kx/Ky; notebook discrepancy figures and DC-bin mismatch diagnostics. It is not a fresh method result or truth. |
| notebooks/reference_outputs/cleaned_measurements.npz | Fresh active notebook measurements and runtime/table; written then read as a separately identified record, never as inverse input. |

P/gallery/numeric_sources.npz binds each fresh current/spectrum/window/field and ablation to5PNG/SVG/PDFgroups; metrics.json stores full-precision scalars. Native/raw observations and calibration are read from the public NPZ, not hidden fixtures. Paper historical simulated loops/noise/pickup-coil sweeps are not passed off as new measurements: this task selects the modern measured-data method and actively computes cutoff,assumed-height,sensor and boundary sensitivities. There was no supplied upstream notebook whose cached plots replace active computation.

## Metric definitions and threshold rationale

Each signed full component separately uses cosine without mean subtraction, and relative_L2=norm(estimate−reference)/norm(reference). This is explicitly **not dynamic-range NRMSE**, not paper normalized mean-square deviation, and not actual current accuracy. Both cosines≥0.99999999,both relativeL2≤1e−8. The reference itself produces zero error; a nonzero numerical tolerance was selected before extraction from independent bit-identical original repeats and physical precision, rather than using mechanical1.1×zero.

Require filtered_forward_residual≤1e−8and divergence_ratio≤1e−8,DC_ratio≤1e−10. Filtered target is H*FFT(padded flux)with DC0; the physical prediction uses the complete submitted **real**field with the nominal unfiltered forward. Raw high-frequency/noise/DCattenuation is not relabeled as forward error. Divergence is norm(kxJx+kyJy)/norm(K*sqrt(abs(Jx)²+abs(Jy)²)); DC ratio is norm(component means)/RMS(all components). No scale,sign,phase,crop,permutation,alignment or interpolation fit is used. All required metrics must pass. Preselected stage rtol2e−11/atol2e−10and full relativeL2tolerance1e−8were never relaxed.

The standalone Judge resolves hidden assets from its own location, uses allow_pickle=False, validates exact real float64shape and finiteness, evaluates actual truth preferentially if ever supplied, and imports no src or submitted code. Reports have status,passed,metrics,thresholds,errors and valid finite JSON. Exit0pass,1poor valid prediction,2malformed/evaluator/write error. Tests include zero/scaled/offset/swapped current,missing file/key,wrong shape/dtype,complex,NaN/Inf,object,overflow,foreign cwd,report creation/write failure and evaluator failure/truth precedence.

## Scientific interpretation and limits

This reconstructs relative current under a2Dquasistatic effective operator, independent calibration, source sign convention and explicit boundary assumption. It is not actual current truth, A/m calibration, absolute direction, exact sensor registration, a3Dreconstruction, or validation of unmeasured off-image currents. Boundary sensitivity is substantial: switching the assumed ramp to zero padding changes the full field by relativeL2≈0.5058; the notebook shows this rather than concealing it. Height/PSF/cutoff ablations are computed model-sensitivity studies on the same measured data, not repeated acquisitions or identified optimal parameters. Unknown calibration moment/noise variance and coarse-kernel geometry remain limitations. Forward agreement alone neither establishes physical uniqueness nor removes regularization bias.

Final exact execution records, full figure inspection and gate ledger are appended after the completed run below. Private manifest hashes are authoritative over rounded prose.

## Final executed validation and gate ledger (2026-10-21)

Environment: shared `D:/yan1/agent/AutoSkill/imaging-101-maintainer/research-runtime/Scripts/python.exe`, Python3.11.11,NumPy2.4.6,SciPy1.17.1,Matplotlib3.11.1,pytest9.1.1,nbconvert7.17.1,Windows CPU. OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1,PYTHONDONTWRITEBYTECODE=1. No environment upgrade or new dependency installation. P/cleaned_environment.json and original_environment.json capture executable,platform,versions and complete compiler/BLAS/LAPACK configurations. Original oracle uses only B4/root/oracle-env/Scripts/python.exe. NumPy uses OpenBLAS0.3.31.188.0ILP64,SciPyOpenBLAS0.3.30LP64. Optional PyYAML configuration-formatting warnings did not require installation.

Final memory snapshot P/memory_check.json: free physical17630748KiB,free commit39449592KiB,both above6GiB. Before original capture the system reported17,755,996KiBphysical/39,235,836KiBcommit free; parent confirmed the other large experiment had completed. Fine-grid original preparation ran privately once; released runtime uses exact supplied independent calibration and requires no huge spline buffer.

| Executed command | Exit | Wall seconds | Exact command/log |
|---|---:|---:|---|
| CLI | 0 | 0.410618 | P/final_cli.json/.log |
| standalone Judge | 0 | 0.410816 | P/final_judge.json/.log |
| full pytest | 0 | 4.823167 | P/final_pytest.json/.log |
| active notebook | 0 | 32.638897 | P/final_notebook.json/.log |

Commands are actual `python -m src.main`; `python evaluation/judge.py --prediction output/reconstruction.npz --report output/judge_report.json`; `python -m pytest test/ -v -p no:cacheprovider --basetemp P/pytest-final`; shared `python -m nbconvert --to notebook --execute notebooks/squid_planar_current_imaging.ipynb --ExecutePreprocessor.kernel_name=imaging-101-research --ExecutePreprocessor.timeout=300`,with private temporary output. P/verify.py records full exact command arguments,cwd,runtime and return codes. Timeout300s covers full fresh sensitivity calculations and vector/bitmap exports with substantial margin. Final notebook has explicit cell IDs and inline backend; ordinary Windows Proactor/ZeroMQ local-kernel warnings are retained in log but no notebook error outputs.

**26/26 tests passed. Canonical notebook has3executed code cells,zero errors,and5nonempty image/png scientific groups.** Executed private notebook was copied to canonical only after these checks; output/ contains no temporary notebook. All5actual final PNGgroups were visually inspected, including final per-component imaginary diagnostics; corresponding SVG/PDF files and full-precision source NPZ/JSON remain private.

| Fresh nominal quantity | Actual value |
|---|---:|
| ncc_jx | 0.99999999999999989 |
| relative_l2_jx | 0 |
| ncc_jy | 0.99999999999999989 |
| relative_l2_jy | 0 |
| filtered_forward_residual | 4.1918793570890797e-16 |
| divergence_ratio | 6.8817977567313165e-15 |
| dc_ratio | 4.0348773285682049e-18 |
| imaginary_ratio | 8.5433665407372288e-16 |
| unmodified_imaginary_ratio_jx | 0.36005503696306407 |
| unmodified_imaginary_ratio_jy | 0.22321780000506614 |
| unmodified_real_filtered_forward_residual | 0.14181706413143685 |
| unmodified_real_divergence_ratio | 0.01239514062190711 |
| constant_boundary_relative_difference | 0.50577922210821935 |
| total_fresh_experiment_seconds | 1.002574199999799 |

The local full two-component nominal array is **bit identical** to the independently frozen corrected original array. All3cutoff stage/reference checks pass unchanged preimplementation tolerances. Both numerical components and physical diagnostics meet fixed criteria. No acceptance threshold was tuned after inspection. The sizable boundary sensitivity is a reported scientific limitation, not a hidden alignment or threshold relaxation.

| Gate | Status | Evidence |
|---|---|---|
| Paper full text,exact remote source revision,rights,data hashes | PASS | Sources table; paper_source.json; remote API response; manifest |
| Original exact-bound execution and independent repeat | PASS | Parent original + bit-identical private22.105srun; original spectra and direct full complex trace |
| Minimal correction independently justified and separated | PASS | analytic mode/delta; minimal_correction.patch; untouched and corrected numeric archives |
| Oracle-first fixed contract/thresholds and parity/physics | PASS | oracle_record.json before src;26tests; all3cutoffs |
| Constants,signs,units,PSFpitch/origin and boundary model | PASS | explicit public effective transfer; analytic signed mode; exact original stages |
| Complete CLI and independent standalone strict Judge | PASS | final command logs and output/judge_report.json |
| Malformed,poor,overflow,foreign-cwd,evaluator/report failures | PASS | contract and full-pipeline tests |
| Public allowlist sufficiency/no truth or URL leaks | PASS | all9public arrays documented; raw/calibration equality; public scan |
| Local mathematics/minimal unchanged shared environment | PASS | AST imports/call-role audit; no original/domain/cross-task runtime import |
| Complete active NB,scientific comparisons and visuals | PASS | all fresh cutoffs/heights/kernel/boundary ablations;5inline groups |
| Metrics/reference/history distinctions | PASS | separate signed components; actual truth absent; untouched complex output separately labeled |
| README,plans,signatures and semantic consistency | PASS | final_audit.json signatures/calls; explicit source/math review |
| Final inventory,licenses,manifest/private provenance | PASS | LICENSE/NOTICE; manifest; no temporary task artifacts |

All mandatory gates are passed locally; status is **ready for independent parent audit**, not parent acceptance. The preceding six frozen task directories were not modified.

## Exact bound data hashes

| Asset | SHA256 |
|---|---|
| D:\yan1\agent\AutoSkill\imaging-101-maintainer\research_batch4_20261021\material_physics\current\sample_data\mag.npz | c165904f2a5fc0d00ad140dbaeaa69dc561dba35921ae3ad9d0adbf3154c0ecb |
| D:\yan1\agent\AutoSkill\imaging-101-maintainer\research_batch4_20261021\material_physics\current\sample_data\psf.npz | a7b76fac118399d89970657d42eb1d5ca04244dc75ad1bf16f5c15fa2dd21b82 |
| D:\yan1\agent\AutoSkill\imaging-101-release\tasks\squid_planar_current_imaging\data\raw_data.npz | 9e574546cc8880430adfeb93fe01e7f71227d998e91dee46e28e0effc1b38c75 |
| D:\yan1\agent\AutoSkill\imaging-101-release\tasks\squid_planar_current_imaging\evaluation\baseline_reference.npz | cea3da7b0688eeed8ca4150d94988dbe17dfc3b03e4dadabcab6b8ea3c19e7c2 |

Final provenance-label review: figure2 identifies the untouched output as an unmodified oracle reproduced in this batch, not author historical cache. Its matching notebook caption is corrected consistently. Full active notebook and all scientific exports were rerun after this text-only change; numerical method, thresholds and field arrays remain unchanged.


## Parent acceptance review

Parent independently ran26 tests, foreign-cwd Judge0 and source/inventory audit errors[]. Read local Fourier magnetic operator and deconvolution, original odd-frequency/PSF-shift/DC correction evidence, physical sign and relative-unit limitations, and all5 actual scientific groups. The complete padded two-component current field is judged; crop is display only. Unmodified original and minimally corrected oracle are separately labeled and repeated on the exact real flux observations; both corrected components match bitwise. Independent filtered-forward, divergence and DC tests pass. Final active notebook reran32.639s after provenance-title repair; figure2 explicitly identifies this-batch reproduced unmodified oracle as not truth. Independent PSF calibration and unknown absolute polarity/amplitude are explicit; no absolute A/m or actual current truth invented.
