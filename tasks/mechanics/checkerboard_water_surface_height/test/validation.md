# Provenance and validation

Status: ready for parent audit, not self-accepted. Mandatory gates below passed. Reused real optical observations; no independent experimental height truth is available.

## Paper, source, data and license

Physical method: Wildeman, Real-time quantitative Schlieren imaging by fast Fourier demodulation of a checkered backdrop, Experiments in Fluids59,97(2018), DOI https://doi.org/10.1007/s00348-018-2553-9. Primary author paper https://phikwadraat.nl/papers/Wildeman_Schlieren_ExpFluids_59_2018.pdf verifies carrier demodulation and measured water-wave use. Optical model: Moisy,Rabaud,Salsac, A synthetic Schlieren method for the measurement of the topography of a liquid interface, Experiments in Fluids46,1021-1036(2009), DOI https://doi.org/10.1007/s00348-008-0608-z, primary author version https://www.fast.u-psud.fr/~moisy/papers/mrs_eif09.pdf. Phase unwrapping: Herraez et al., Applied Optics41,7437-7444(2002), DOI https://doi.org/10.1364/AO.41.007437.

Selected separately authored modern reference: https://github.com/Trapped-modes-2025-LTG/Trapped-Modes-LTG, remotely verified commit a8430e63046c0b97b463efd49bc51eb5c356f720, original API response trapped_commit.json. MIT license copyright2025 Juan Giordano preserved at task LICENSE. Its research archive includes reference_2.png and202406_1457001661.bmp, the exact real images paired in examples/fcd_example.py, together with the four-layer optics and2.2mm square size. No separate restrictive data terms were present; the distributed research archive's MIT notice is preserved. These are not presented as observations or code by the2018/2009 method authors. The term synthetic Schlieren names the optical modality, not a claim that the bound data are simulated.

Original phase-unwrapping dependency: scikit-image0.26.0, downloaded tagged source src/skimage/restoration/unwrap_2d_ljmu.c and _unwrap_2d.pyx, with exact source hashes in the private archive index. Original C by Munther Gdeisat/Miguel Arevallilo Herraez, modified by Gregor Thalhammer; scikit-image team copyright and BSD3 terms retained in task NOTICE. Local source implements the explicit reliability/edge/grouping mathematics using NumPy and weighted union-find; no compiled unwrapper or domain solver is bundled/imported. The reference C source's boundary random-cost convention is reproduced with explicit portable integer arithmetic, not a dependency on the host C runtime.

All primary images, source files, remote responses, capture scripts, original logs and failed screening trials remain in D:/yan1/agent/AutoSkill/imaging-101-maintainer/research_batch4_20261021/material_physics. surface_archive_index.json records source/data/script hashes and sizes. No upstream checkout, standalone PNG/PDF or capture script is inside the task package. This inverse recovers liquid elevation from optical refraction, distinct from an existing material-surface DIC displacement task.

## Exact observation binding and original oracle

The original1024-square observed frame has an opaque upper-right probe and edge artifacts. An initial whole-frame screening reconstruction is archived separately as surface_oracle.npz/log: it shows large unwrapping/height artifacts near that occlusion and is not the bound benchmark. Fix the transparent-fluid region at zero-based original rows256:897 and columns80:721 before local implementation. Both reference and observed image use this identical641-square crop. No interpolation, intensity normalization, generated object, hidden seed or target-derived calibration enters the public observations. The crop and altered periodic boundary are public acquisition metadata.

The source also contains an incorrect even-grid integration branch, which zeros frequency indexN/2+1 rather than the self-conjugate Nyquist mode. This was independently demonstrated, not silently copied: capture_surface_integrator.py evaluates an analytic seven-cycle cosine and its exact gradient. Original16-square reconstruction has relative L2 error0.5000000000000003; original17-square reconstruction error6.7263e-15. Frozen forensic inputs/outputs are consumed by a dedicated test. The selected odd641-square branch evaluates the correct original Fourier integration and no numerical correction of that bound source function is required. Cleaned integration explicitly rejects even shapes; it does not claim general correctness of the source's even-grid branch.

Private oracle environment: oracle-env/Scripts/python.exe, Python3.11.11 Windows CPU, NumPy2.4.6,SciPy1.17.1,scikit-image0.26.0,Matplotlib3.11.2. Original capture imports original pyfcd classes unchanged. The original loader expression io.imread(...,as_gray=True).astype(float32) is executed directly to avoid importing unrelated pandas/OpenCV analysis helpers; no loader arithmetic changes. Original functions compute_carriers, compute_phases, compute_displacement_field, layered height, and Fourier integration execute on the exact public ROI. No upstream source/control flow/formula is patched for this run.

Commands: oracle-env/Scripts/python.exe capture_surface_roi.py, then capture_surface_roi_rerun.py, with OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1 and PYTHONDONTWRITEBYTECODE=1. First corrected stage capture1.5673269s, exit0, surface_roi_oracle.log/json/npz; independent rerun1.6044449s, exit0, surface_roi_oracle_rerun.log/json/npz. All stage and final arrays are bitwise identical. Both source phases contain full cycles and require unwrapping; bypassing it is not accepted as the benchmark method.

An initial diagnostic wrapped-phase capture accidentally used NumPy FFT while the original full reconstruction uses SciPy FFT. This produced up to9.3e-5rad stage difference due float32/complex64 promotion. The full original result was unaffected. Capture was corrected to use the actual SciPy FFT path, then rerun and frozen before cleaned extraction. Local phase parity is now exact modulo the unavoidable constant2*pi offset, with no pixelwise fitting.

## Fixtures, references and traceability

| Stage or mathematical step | Original function | Local function | Independent evidence |
|---|---|---|---|
| Grayscale input/crop/axis binding | analyze.load_image expression and fixed capture ROI | preprocessing.load_data | Raw hash, shape/dtype/public allowlist tests |
| Angular frequencies, reference peaks and pattern scale | fourier.find_peaks/pixel_to_wavenumber; fcd.compute_calibration_factor | frequencies/find_peaks/carriers | Exact frozen peaks, masks, scale and complex reference carriers |
| Two reference-relative wrapped carrier phases | fcd.compute_phases | physics_model.phases | Exact frozen wrapped phases using source complex64 FFT path |
| Wrapped second differences, reliability-sorted graph unwrapping | scikit-image unwrap_2d_ljmu.c | unwrapping.unwrap_reliability | Captured real phase maps exact modulo constant; independent smooth multi-cycle phase case |
| Two-carrier2x2 displacement inverse | fcd.compute_displacement_field | physics_model.displacement | Frozen full pipeline and dimensional/sign review |
| Layered paraxial optical distance | fcd.height_from_layers/effective_height | physics_model.effective_height | Explicit layer constants and exact scalar test |
| Periodic least-squares gradient integration | fourier.integrate_in_fourier odd branch | integrate_gradient/height_gradient | Exact bound pipeline; analytic Fourier mode; normal-equation residual; even-grid forensic case |
| Real field metrics | Benchmark cosine/dynamic-range conventions | Independent Judge and visualization.metrics | Known scaling, zero/degenerate cases and shared numerical comparisons |
| Forward carrier phase consistency | Acquisition phase=q dot displacement | Independent Judge observation_residual | Fresh two-carrier RMS, null comparison, visualization agreement |

All test/fixtures/upstream_stages.npz arrays originate in the original ROI run: output_height,output_phases,output_wrapped,output_displacement,output_calibration,output_effective_height,output_peaks,output_frequencies,output_masks,output_reference_carriers. Stage and full parity tests consume them. test/fixtures/upstream_integrator_forensics.npz contains original16/17-grid analytic diagnostic inputs and original/analytic outputs, consumed by the named forensic test. Notebook reference_outputs/upstream_reconstruction.npz identifies the original source height separately. evaluation/baseline_reference.npz contains key reconstruction, real float64(1,641,641), the same original standard-method height; it is not actual truth. No ground_truth.npz is invented.

Parity criteria were fixed before local acceptance: exact peaks/masks/calibration/carriers/wrapped phases; unwrapped phase equality after only one constant phase offset per carrier, atol1e-12; full height rtol1e-8,atol1e-12m. Observed maximum height difference1.6263e-19m. Local weighted union-find replaces the source linked-list grouping representation while preserving reliability and integer phase mathematics. The selected Win32 C rand sequence seeded0 is explicitly evaluated with state=(214013*state+2531011)mod2^32, output=(state>>16)&32767. Boundary-cost sequence is an algorithm convention and never a hidden object-generating seed.

## Physical and numerical interpretation

Source carrier coordinates are(row276,column369) and(row271,column277), stored frequencies(ky,kx). Displacements/slopes are stored in(component-x,component-y) order. Calibration derives only from the independent flat-surface reference, with the source's component-average frequency convention; it gives0.00031747269890795626m/pixel. Four supplied layers give H=.0329956084466042m. The original code's special index2 reference to water is preserved for this explicitly fixed four-layer acquisition and not advertised as a general arbitrary-layer model.

The target is the periodic relative elevation component. Fourier integration discards mean gradient, so absolute surface tilt and constant elevation are excluded rather than hidden alignment degrees of freedom. No mean subtraction, gain, translation or crop is performed in field scoring. The observed slopes need not be integrable; independent normal-equation tests show the returned height is the least-squares projection onto periodic gradient fields. The observation Judge allows only two spatially constant carrier-phase offsets to account for the discarded mean slope, as disclosed publicly.

Original wrapped carrier RMS0.14872557562675262rad; zero height0.9728911868162814rad. Fresh selected residual agrees independently. This substantial reduction supports meaningful recovery of observed wave structure, but is not independent height accuracy. Source finite-domain residuals are larger at boundaries; figures retain those pixels and expose them. No noise-variance calibration exists, so fit residuals are not labeled chi-square or uncertainty intervals.

Original rerun equality motivates nonzero reference tolerances NCC>=.999999 and dynamic-range NRMSE<=1e-5. The independent phase residual boundary .15021283138302015rad is1.01times original, fixed before cleaned Judge acceptance. These strict boundaries were not relaxed. Huge finite submissions receive finite field metrics; for heights beyond1e100m the wrapped-phase diagnostic conservatively returns pi instead of overflowing. Such outputs fail the field criterion independently. Missing/malformed/evaluator/report failures have empty metrics and statuserror.

## Executed commands and results

Cleaned runtime is the existing shared research-runtime/Scripts/python.exe, with packages/OS/backend recorded in surface_environment.json. No cleaned environment package install or upgrade was performed. Source, tests, notebook and Judge import only standard library,NumPy,SciPy general FFT/label/sparse-free primitives,Matplotlib. There is no skimage/domain unwrapping runtime import or cross-task import. All commands run from the task directory with single-thread environment except nbconvert's full task path from repository root.

| Command | Exit | Evidence |
|---|---|---|
| python -m src.main |0| surface_cli.log,4.777580s, writes declared NPZ |
| python evaluation/judge.py --prediction output/reconstruction.npz --report output/judge_report.json |0| surface_judge.log and report; fresh NCC~1,NRMSE6.16e-17,phase RMS.14872557562675262rad |
| python -m pytest test/ -v -p no:cacheprovider --basetemp PRIVATE/pytest-surface-final |0| surface_tests_final.log,22passed15.93s |
| python -m nbconvert --to notebook --execute notebooks/checkerboard_water_surface_height.ipynb --output surface_executed_final.ipynb --output-dir PRIVATE --ExecutePreprocessor.timeout=600 |0| surface_notebook_final.log; explicit shared imaging101-research kernel, complete fresh path and both sensitivity variants |

The tests cover original stage/full parity, analytic unwrapping, Fourier inversion, least-squares gradient normal equations, unsupported even grids, original even/odd forensic cases, layered units/calibration/public constants, visual/Judge metric agreement, zero/degenerate visual cases, missing submission/key,shape,dtype,NaN,Inf,zero and1e308 quality failures,foreign cwd,missing evaluator asset and report-write failure. Correct0/1/2 exits and finite strict JSON are verified. Full notebooks execute all selected pixels; no archived result is substituted for a fresh reconstruction.

Notebook comparison table distinguishes reproduced original, fresh selected, fresh no-unwrapping and fresh0.75-radius aperture results. The no-unwrap comparison has NCC.16664,NRMSE.23413,phase RMS.45434rad; narrower aperture NCC.99760,NRMSE.009864,phase RMS.21319rad, all measured here rather than reported historical values. These alternatives are explanatory comparisons, not omitted or zero-filled methods. Source example's height and both phase maps are retained, with additional physical displacements, residuals, spatial sections and metrics.

## Review, figures and inventory

Four private figure groups in figures/checkerboard_water_surface_height: acquisition/reference/carriers; original/fresh/difference plus actual variants; wrapped/unwrapped phases and two physical displacements; forward phase residuals and height/slope sections. Each has PNG/SVG/PDF and exact numeric_sources.npz, comparison_metrics.json, SHA256 manifest. All four actual PNGs were visually inspected for wave structure, units, scales, clipping and labels. The near-zero difference panel is explicitly a numerical reproduction comparison. Sources/metrics are freshly exported by the final active notebook.

README/approach/design/notebook mathematical strings were checked for control-character corruption/lost LaTeX backslashes and reviewed semantically: refraction sign,carrier vector axis order,2x2determinant,metre calibration,H-layer sum,phase cycles,periodic zero mode,objective normalization and output gauge match code. Runtime callable signatures/defaults/side effects match design.md. No matplotlib.use occurs in importable code. Public materials contain no hidden height,source path,GitHub URL or task-specific seed. Final source includes a dedicated readable unwrapping module; no opaque vendored framework. No generator/PDF/image archive/temporary executed notebook/parallel report is packaged. Required MIT/BSD attribution is preserved.

## Mandatory gates

| Gate | Status | Evidence |
|---|---|---|
| Papers, exact remote source revision, data/license/hash mapping |passed| Above and private archive index |
| Exact-bound original/local complete paths and rerun |passed| Original capture/rerun and fresh CLI/NB |
| Independent fixtures plus applicable physics |passed|22tests and independent Fourier/phase cases |
| Constants, units, coordinates, numerical conventions |passed| Exact stage parity and semantic review |
| Standard CLI and strict standalone Judge |passed| Command logs and generated output |
| Malformed, poor, extreme, evaluator/report cases |passed| Contract tests |
| Complete public model/data/output contract, no leaks |passed| README/raw/requirements review |
| Minimal shared dependencies, local executable inverse math |passed| Explicit FFT,demodulation,graph unwrap,2x2solve,integration |
| Active notebook and preserved comparisons |passed| Canonical successful executed notebook |
| Metric/reference/figure provenance and thresholds |passed| Above and numeric manifest |
| Plans/source/notebook semantic agreement |passed| Recorded formula/precision/caller review |
| Template, cleanup and license notices |passed| Final inventory review |

SHA256 data/raw_data.npz: 7a01b410f17c7c6d643556e22ce732a3bacf66e4927cc35de499545d12b67922

SHA256 evaluation/baseline_reference.npz: bedde17a11613cf28df6ebd340801b932051fa5db9fc2080101b42c2a9ff51e0

SHA256 test/fixtures/upstream_stages.npz: e7410f7a37ae2565fc622066c0b9ad27b2fa8c8acd3e63cfabee6a928460b63a

SHA256 test/fixtures/upstream_integrator_forensics.npz: e0352fad603013c4d91344afb3ce9f9e5f2ebb5a92be18f5481e66943d90f612

Parent document audit correction: three Unicode superscript sequences in approach.md had been converted by shell encoding into question marks. They now use explicit ASCII H^2+V^2+D1^2+D2^2, squared L2 norms, and kx^2+ky^2. Full README/plan/notebook source/output text scan finds no remaining question-mark corruption or control characters. No numeric code, reference or executed notebook output changed. Final test count is22, independently run in surface_tests_final.log(15.93s), superseding earlier19/21 intermediate counts.


## Parent acceptance review

Parent independently ran22tests (22 passed in17.35s), foreign-cwd Judge0 and source/inventory audit errors[]. Read explicit Fourier carrier calibration, reliability-sorted phase union updates, multilayer refraction and periodic gradient inverse; inspected all4 actual scientific figure groups. Real641-square transparent-water ROI and independent flat checkerboard calibration; original complete selected-ROI algorithm twice executed, local height difference1.63e-19m. Source even-grid Nyquist suppression error independently demonstrated and unsupported even grids rejected, preserving correct odd-grid bound method. Forward carrier residual.1487rad versus null.9729rad and active no-unwrapping/aperture comparisons retained. Formula superscript encoding repaired in plans without changing numerical code or executed notebook outputs; no experimental height truth invented.
