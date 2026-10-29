# Sources, physical conventions and completed validation

## Bound task and selected method

The task reconstructs a signed initial exploding-reflector wavefield from the complete real GSSI dune profile:512time samples ×1040traces, not a synthetic toy or a spatial subset. The output is a458×1040 planar Stolt image in receiver-count amplitude units; it is not absolute dielectric permittivity. Nominal velocity0.14m/ns and delay5ns are the declared physical model. No independent actual subsurface truth exists. Hidden baseline_reference is the original published standard-method output, never relabelled truth.

The selection is **ImpDAR's own Stolt implementation applied to GPRPy's real acquisition**. It is not the separately licensed CREWES/irlib migration and does not claim unchanged reproduction of either source paper's complete figures. Complete velocity and spectral-boundary comparisons and the source dune demonstration's surveyed-terrain context are retained actively. No uncomputed competing method is represented by zeros or an archived fresh-looking result.

## Source versions, papers, data and licenses

| Source | Exact revision / publication | Verification and role |
| --- | --- | --- |
| https://github.com/dlilien/ImpDAR | **27efd1e5517bfc3d510a1305d82e26a0cfdc820d** | `git ls-remote ... HEAD` verified remotely2026-10-20; full private checkout has the same commit. Original `src/impdar/lib/migrationlib/mig_python.py::migrationStolt` is the oracle. GPLv3. |
| https://github.com/NSGeophysics/GPRPy | **3b1f75eba820764b2147568fc0cc40f3a47919d5** | Remote HEAD retrieved2026-10-20; full private checkout. MIT acquisition reader/background/display functions and full experimental example data. |
| ImpDAR paper | Lilien, Hills, Driscol, Jacobel, Christianson(2020), *ImpDAR: an open-source impulse radar processor*, Annals of Glaciology61(81),114–123; **10.1017/aog.2020.44**, published2020-06-29 | Full published text read, especially§2.2.3–2.2.7 and the explicit Stolt discussion; PDF at https://par.nsf.gov/servlets/purl/10173862 archived privately and relevant page rendered/inspected. Published article CC BY. |
| GPRPy paper | Plattner(2020), *GPRPy: Open-source ground-penetrating radar processing and visualization software*, The Leading Edge39(5),332–337; **10.1190/tle39050332.1** | Full6-page paper downloaded from https://www.alainplattner.net/downloads/Plattner2020.pdf , text extracted and page333 rendered/inspected. Archived copyrighted PDF is private, not distributed with task. |
| Complete real radar file | https://raw.githubusercontent.com/NSGeophysics/GPRPy/3b1f75eba820764b2147568fc0cc40f3a47919d5/gprpy/exampledata/GSSI/FILE____032.DZT |1040traces,512samples, acquisition-header time window48ns, density50traces/m; original signed integer values exactly preserved as float64. |
| Independent surveyed terrain | https://raw.githubusercontent.com/NSGeophysics/GPRPy/3b1f75eba820764b2147568fc0cc40f3a47919d5/gprpy/exampledata/GSSI/FILE____032p.txt |36(x,elevation)stations, legitimate independent acquisition geometry, not target-derived calibration. |

The GPRPy repository's MIT terms and author attribution are reproduced in NOTICE; ImpDAR's GPLv3 is in LICENSE. Source paper figures/PDFs, complete checkouts and capture scripts remain private. The original dune demonstration credits a Princeton University freshman-seminar field trip taught by Frederik Simons, Christopher Harig and James Smith. The present task uses the identified example files and does not invent a more precise acquisition location or acquisition date not established by their headers.

| Asset | SHA-256 |
| --- | --- |
| FILE____032.DZT | e7e1e9b087addebf27a55b2b62bff5180a560b4225a9e84b77f9de0abd48ff8a |
| FILE____032p.txt | a4e41cefd4d94e693dccd0fc21eb2edfcc2c88f5cb99ad1188631f50336a19d9 |
| Bundled data/raw_data.npz | ae4935b7fef14284dcd679be82016755864eba16855ba258f62ffcdc4cceb5c4 |
| Published ImpDAR PDF |366111faeb6b7b55aae16299003b663cf0f68e542c30654b848338a336a21390 |
| Published GPRPy PDF |46f428a279e33be151dec8edae06b5a6a9b709b6326e680dabd47bba124cfbe0 |

Both remote revisions were actually network-verified, not inferred local tracking freshness. Paper PDFs are the published versions retrieved on2026-10-20; no unverified claim about later paper corrections. The author-hosted GPRPy PDF initially failed Windows TLS certificate validation; a narrowly scoped public-PDF request retrieved the actual PDF, whose header/hash/text and rendered page were then checked. This affected only private source acquisition, not runtime or source-code verification.

## Private archive and fixture consumers

Base: `D:/yan1/agent/AutoSkill/imaging-101-maintainer/research_batch_20261020/earth_mechanics/`.

| Private artifact | Purpose / consumer |
| --- | --- |
| impdar-upstream/, gpr-upstream/ | Complete exact-revision source/data/license archives. |
| impdar-select/ | Earlier exact-revision selective source/docs downloads; private research only. |
| impdar-paper.pdf/.txt, gprpy-paper.pdf/.txt, *paper-page*.png | Actual papers, extracted text and rendered review evidence; not task input. |
| mig_fk.py | Inspected rejected CREWES-licensed source; **not executed, copied into task, or used by runtime**. |
| oracle-env/ | Isolated source-verification environment. Private PyMuPDF used for paper review has no task dependency. |
| capture_gpr.py | Original GPRPy reader/preprocessing/display and original ImpDAR kernel execution, with explicit declared time calibration and data-container adapter. |
| gpr_acceptance.json | Scope, output, fixed tolerances and independent physics requirements frozen after original run and before local implementation. |
| gpr_oracle.npz, gpr_oracle*.log, gpr_oracle*stderr.log | Independent numeric capture and run evidence; full-array stage/image/terrain outputs. |
| gpr_oracle_repeat_equality.json | Exact repeated-oracle equality, including identical known diagnostic/display NaN masks. |
| gpr_stage_tests.log, gpr_tests*.log, gpr_notebook*.log | Stage tests, final tests, full active notebook execution. |
| gpr_fresh_evidence.json | Full-precision timing, parity, numerical-boundary measurements, environment and per-file hashes. |
| gpr_executed_review.ipynb, gpr_review_0.png..gpr_review_3.png | Executed maintainer notebook and inspected full scientific figures. |
| finish_gpr.py, validate_gpr.py | Maintainer notebook/evidence authoring scripts; not imported by task. |
| oracle_numerical_backend.log, cleaned_numerical_backend.log | NumPy numerical-backend configuration evidence. |

Packaged `test/fixtures/upstream.npz` is consumed by test/comparing_test for exact public-array parity, interpolated/background stages, original FK/KK/scaling, all3velocity images and terrain display. Its33MB size retains full independent full-profile evidence rather than a small favourable subset. `notebooks/reference_outputs/upstream.npz` holds the3explicitly identified upstream image comparisons for the notebook; `cleaned_measurements.npz` holds its actively computed source-convention/velocity/bandlimited fields and runtime. `evaluation/baseline_reference.npz` supplies only the declared standard-method reference. Each numerical copy therefore has a distinct documented consumer. No loose NPY/MAT/DZT/PDF, unused scripts or executed-notebook duplicates remain in the task.

## Original execution and acceptance

Command: private `oracle-env/Scripts/python.exe capture_gpr.py`. The script directly imports **unmodified original source files**: GPRPy readdzt, dewow, remMeanTrace, correctTopo; ImpDAR migrationStolt. A SimpleNamespace supplies exactly the documented fields data/snum/tnum/dt/trace_int/dist; it does not replace or reimplement the original inverse. A Python return-event trace captures FK/KK/scaling locals from the executing original function without changing its source or arrays. The trace is disabled for subsequent comparison runs. No cleaned src import exists in capture.

Original full primary kernel1.7732549000065774s with capture instrumentation, untraced comparison kernels about1.69s/2.72s (exact run log). First unaugmented primary capture1.7559853999409825s. Run with independent terrain display exit **0**; full repeat command exit **0**, and **every captured array exactly equal** to frozen initial output, documented in gpr_oracle_repeat_equality.json. Known NaNs occur only at undefined diagnostic scaling[0,0] before its image coefficient is explicitly zeroed, and outside the terrain display's observed support. Public data and required prediction are finite throughout.

Oracle Python3.11.11 Windows build26200, NumPy2.4.6/SciPy1.17.1, single-thread OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=1. General-purpose tqdm is used by original GPRPy module imports. No GPU. Source warnings are Python SyntaxWarning `is` with integer literals in unrelated GPRPy helper branches; selected functions execute successfully. No numerical warning changes scored output. No stochastic bound-data generation, training or seed.

Acceptance fixed in gpr_acceptance before local source existed: whole-array preprocessing/image rtol1e-10/atol1e-8counts; complex spectral absolute floor1e-5 with the same relative tolerance (spectral amplitudes reach7.52e7); scaling rtol1e-13/atol1e-14. Full physical identities and point focusing are separate gates. Numerical Judge boundaries NCC≥.99999999 and range-NRMSE≤1e-7 remain unchanged. Baseline self-comparison1/0 is not substituted for fresh measurements; positive tolerances reflect repeat/float64 arithmetic, not physical accuracy.

## Equation / function / test mapping

| Required mathematical stage | Original source | Local implementation | Validation case |
| --- | --- | --- | --- |
| Acquisition signed amplitude/time/space | GPRPy gprIO_DZT.readdzt; header conventions | frozen public NPZ; preprocessing.load_data/load_topography | original raw/time/x/survey arrays exact equality; constants test |
| Continuous5ns calibration | declared acquisition application, independent scipy interp1d in capture | preprocessing.prepare explicit two-point linear interpolation | full458×1040 shifted-array fixture; known uniform time and output-size checks |
| Constant trace offset / mean trace | gprpyTools.dewow/remMeanTrace global windows | preprocessing.prepare two explicit means | whole stage fixtures and independent zero-mean identities |
| Aperture weights and FFT | ImpDAR migrationStolt taper/rfft2 | preprocessing.taper; solvers.migrate | original FK fixture and harmonic/linearity tests |
| Half-speed scalar dispersion / obliquity | paper§2.2.7; source migrationStolt; migration docs Stolt equation | physics_model.dispersion | independently checked dispersion relation, zero-horizontal-frequency identity, two-way diffraction focus |
| Linear sampled spectral map | original RectBivariateSpline(kx=1,ky=1) at fixed acquired kx | interpolation_weights/remap/remap_adjoint | original KK fixture; complex adjoint identity; explicit endpoint examples |
| Nyquist/DC and inverseFFT | source range(snum//2), KK[0,0]=0, irfft2 | solvers.migrate | entire primary and .12/.16velocity images; odd-length rejection and zero-response cases |
| Terrain display after migration | GPRPy correctTopo PCHIP/integer shifts | visualization.terrain_display | full596×1040 display exact equality, including geometric NaN mask |
| Source bandwidth approximation | original spline endpoint extrapolation | selected default; explicit bandlimited=True diagnostic | independent numerical energy/image-impact quantification below; active comparison |
| Metrics and CLI | benchmark declared signed-field contract | independent evaluation/judge.py | known metric examples and complete malformed/quality/evaluator/foreign-cwd/report tests |

The small point-diffraction helper is an independently derived analytic validation model, not an upstream parity fixture or a generator of benchmark observations. It uses t=2sqrt((x-xs)²+z²)/v with a Ricker pulse and cylindrical spreading; migration focuses its known point within one horizontal and3depth samples. Thus agreement between duplicated source formulae is not the only physics evidence.

## Investigated conventions and numerical/physical limits

* **Migration license:** GPRPy's wrapper normally downloads the separately licensed CREWES/irlib kernel with non-commercial restrictions. It was inspected and rejected. The present original oracle and local extraction use ImpDAR's own GPL implementation, with all required notices. No wrapper around a reconstruction-domain package is shipped.
* **Sample geometry:** Original GPRPy wrapper uses an endpoint-inclusive line0..1040/50, stretching its step slightly. The selected application instead uses exact acquisition-header spacing1/50 and all1040 traces. The time axis preserves original inclusive0..48ns/512 convention, not a recomputed alternative48/512. README and tests match the actual supplied constants.
* **Time zero:** GPRPy demonstration crops the nearest acquired sample and resets its first coordinate. Here declared continuous linear interpolation starts exactly at5ns, retaining458 valid uniform samples. This is explicit acquisition preprocessing for the ImpDAR application, not a hidden change to the core inverse or an unchanged-figure claim.
* **DC and Nyquist:** Source excludes the final real temporal Nyquist output row and zeroes KK[0,0]. Local code preserves both, rejects generic odd temporal lengths to avoid source inverse-length truncation, and retains original signed amplitude normalization. No added v/2 factor: discrete frequency-cell conversion cancels that factor at output dz=v*dt/2.
* **Unsupported temporal band:** Source spline extends the highest measured value when omega*>omega_max. Independent full-data quantification gives unsupported-grid fraction **.020656354515050167**, unsupported KK-energy fraction **4.983924646640737e-7**, last acquired-frequency energy fraction **9.784685199529186e-8**. Explicit zero-outside-band diagnostic changes the image by relativeL2 **.0007059691669358328**, range-NRMSE **2.9671960731516316e-5**, max **23.18485085132905counts**. The small <0.1% relativeL2 approximation is accepted as a declared numerical boundary convention, not described as strict bandlimited physics. The alternate field is actively computed/plotted; no standard reference or threshold was changed to hide it.
* **Topography:**36 independent stations support a real elevation-aware display. Post-migration trace shifts do not correct travel-time propagation for the sloping acquisition; this remains a physical approximation. The required image remains planar and finite, while only geometric display padding is NaN. PCHIP/shift output matches original source exactly.
* **Scientific comparison scope:** The ImpDAR paper's Fig5 compares multiple migration algorithms on a different glacier profile. This task deliberately selects Stolt on the bound dune data, rather than asserting or fabricating those other experimental results. The notebook explains that scope and retains active3velocity/bandwidth/terrain comparisons, common scales, numeric tables, paired bars, observed/prepared/migrated images and explicit residuals. GPRPy paper/prototype terrain display context is preserved; no uncomputed method is zero-filled.

## Cleaned result and execution evidence

Shared interpreter `D:/yan1/agent/AutoSkill/imaging-101-maintainer/research-runtime/Scripts/python.exe`, Python3.11.11, NumPy2.4.6, SciPy1.17.1, Matplotlib3.11.1. No package upgrade in shared environment. Only these numerical packages plus stdlib are imported by src/Judge/tests/notebook. Private original code and paper-reading tools are not task dependencies. No GPU; largest scientific arrays are a few MB each, total numerical working memory far below1.5GiB. File sizes remain below standard repository/Git limits; full evidence is intentionally not downcast.

Fresh full raw-to-image runtime **.06331610004417598s**, final notebook full inverse **.067056s**. Max absolute upstream differences: time shift/dewow/background each **3.637978807091713e-12counts**, FK **1.4901161193847656e-8**, KK **1.624624095827143e-7**, image **1.978150976356119e-11counts**, .12velocity **2.4556356947869062e-11**, .16velocity **2.091837814077735e-11**. Terrain display on the upstream field matches elementwise exactly. No tolerance was relaxed after local implementation.

Fresh Judge NCC **1.0**, dynamic-range NRMSE **7.158318914645772e-17**. RMS image amplitude1657.7546001096914counts versus upstream1657.7546001096912. Active .12/.16velocity NCC against standard .14is .9462572243534222/.9393279209362301, NRMSE .013831267006752059/.01455440979194264. These are comparison metrics, not true-reflector accuracy. There is no calibrated noise variance, so a noise-weighted chi-square≈1is not claimed.

Commands from task directory using shared interpreter:

* `python -m src.main`: exit **0**, reads full public acquisition and writes declared finite float64 output.
* `python evaluation/judge.py --prediction output/reconstruction.npz --report output/judge_report.json`: exit **0**, status pass with fresh metrics above.
* `python -m pytest test/ -v`: final exit **0**, **21passed in2.65s**. Initial red collection failed because src did not yet exist. Full suite includes independent physics, full original parity, source interpolation boundaries, quantitative band-edge approximation, signed metric cases, missing file/key, shape, float32/complex, NaN/Inf, finite1e308 amplitude overflow, zero/reversed poor output, evaluator-assets failures, foreign cwd, report-write failure and backend/CLI checks. The overflow contract verifies both evaluate and actual CLI exit2 with a valid error JSON, empty metrics and descriptive numerical-overflow error; no NaN/Infinity reaches report serialization. TEMP/TMP uses the private writable archive temp due to existing system-temp access constraints.
* `python -m nbconvert --to notebook --execute notebooks/gpr_dune_stolt_migration.ipynb --output gpr_dune_stolt_migration_executed.ipynb --output-dir output --ExecutePreprocessor.timeout=180 --ExecutePreprocessor.kernel_name=research`: final exit **0**, private JUPYTER_PATH explicitly selects the shared interpreter.180s accommodates startup/full multi-variant reconstruction/plot rendering, with actual reconstruction timing above. Windows zmq selector fallback/debug warnings do not affect results. Final executed outputs embedded in the maintained notebook; temporary output notebook archived privately and removed.

All4figures were visually inspected: full raw/prepared/signed migration, source residual≈1e-11counts, bandlimited boundary difference up to23counts, all3velocity panels with explicitly labelled common nominal depth, surveyed-terrain elevation image with geometric exterior masked, paired RMS and boundary-energy bars. Axes/units/colour scales are readable; terrain strata and source/fresh comparisons are present. No figure presents archived results as fresh or reference as truth. Library imports never call matplotlib.use().

Semantic/signature audit checked FFT order/normalization, time interpolation count/units, constant/background signs, source taper, scalar half-speed dispersion, endpoint extension, Nyquist/DC, dimensionless obliquity, all comparison velocity flows, terrain PCHIP/rounding and strict Judge contract. All README constants and both plans agree with actual code; public NPZ keys/shapes/units are fully documented. Source/Judge/test/notebook symbol/call-role audit contains no domain-library solver shortcut, hidden-answer load, checkpoint, cross-task import or unused generator.

## Completion gates

| Gate | State | Evidence |
| --- | --- | --- |
| Papers, exact verified revisions, raw/paper hashes and equation mapping | passed | Source and traceability sections |
| Original and cleaned exact complete-observation execution | passed | Original3velocity/terrain run and exact repeat; full fresh CLI/notebook |
| Independent upstream parity and applicable physics | passed | Full frozen stage/pipeline arrays plus analytic dispersion/harmonic/point/adjoint tests |
| Constants, coordinates, units, precision preserved/explicit | passed | Original inclusive time convention, exact header spatial pitch, explicit5ns continuous calibration; no downcast |
| Standard output and standalone Judge | passed | Required signed finite array, fresh pass, all samples evaluated |
| Invalid/poor/evaluator/report contracts | passed |21-test suite including finite-amplitude metric-overflow JSON |
| Self-contained allowlist and no leakage | passed | All5measured/calibration keys and full output contract in README; no truth/privileged data/metadata sidecar |
| No public GitHub URLs | passed | README/requirements scan; original URLs kept only in maintainer provenance |
| Minimal shared dependencies/local core mathematics | passed | Explicit NumPy spectral remap/weights/inverse, no domain library or cross-task imports |
| Active notebook/scientific comparisons/visual review | passed |4inspected figures, all variants actively computed and source scope explained |
| Result/metric/boundary provenance and rigorous thresholds | passed | Independent capture/repeat vs fresh results separated; numerical boundary quantified; nonzero unchanged tolerances |
| Plans/README signature and semantic agreement | passed | Actual call-flow and mathematical audit above |
| Final inventory/licenses/archive cleanup | passed | Template plus LICENSE/NOTICE; temporary executed notebook/source PDFs/checkouts kept outside task |

No mandatory gate remains failed or unverified. Planar/constant-velocity assumptions, finite-aperture and spectral-boundary approximations, instrument gain and unavailable physical truth remain openly stated limits.


## Independent parent acceptance (2026-10-20)

Parent independently reran the final21 tests (2.69s, exit0), including finite-extreme-value error JSON; ran the Judge from a foreign working directory (exit0); reviewed source, both plans, acquisition/output contracts, independent physics and original stage fixtures; and visually inspected all4 actual scientific notebook figures. The active full notebook has completed. Structural/import/public-input audit found no errors. Parent explicitly reviewed the measured endpoint-extension approximation and the display-only terrain scope. Accepted as method reproducibility on real observations, without claiming material-property truth.
