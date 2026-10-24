# Provenance, qualification and validation

This is a **qualified numerical reconstruction of the published Eq.19 joint objective**, with documented source corrections and a fixed numerical rank convention. It is not unchanged reproduction of the original source solver or its ambiguous Eq.21 proposal. All four physical fields remain unknown. The unchanged author algorithm was also actually run and is preserved as a separately identified comparison. No mandatory gate remains unresolved for this qualified scope. Coefficient-direction equality is explicitly **not certified**; the declared physical-field/update parity gate passes.

## Source, paper, data and private archive

Primary source: https://github.com/jnh277/Joint_strain_d0_tomography at remotely verified commit `12affa403c901845b537d75ca27e3a3fa5c3eb4e`, GPL-3.0. The remote repository metadata and 1,958,011-byte archive are retained; this is not a claim based only on a local tracking revision. GPL text and attribution are included in LICENSE/NOTICE.

Paper: Hendriks, Jidling, Schon, Wills, Wensrich and Kisi, *Neutron Transmission Strain Tomography for Non-Constant Stress-Free Lattice Spacing*, arXiv:1905.06854v2, 18 July 2019; published DOI https://doi.org/10.1016/j.nimb.2019.07.005, Nuclear Instruments and Methods B456,64-73. Archived primary PDF SHA256 `455e0fe3249ae38fbe5f8fc3bd087da0f720936028f931cbfb2265627338955c`.

Private archive root is `../imaging-101-maintainer/research_batch6_20261022/groups/physics_materials` relative to release root, abbreviated P below. Complete sources, PDF, download ZIP, compatibility helpers, capture scripts and original logs reside there, not in this task. Main archive index:

| Artifact in P | Role |
|---|---|
| neutron_joint_strain_source_record.json; neutron_joint_strain.zip; neutron_joint_strain/ | Remote revision/license metadata and untouched original source |
| neutron_joint_paper.pdf; neutron_joint_paper.txt | Actual paper version and searchable text |
| neutron_compat/; neutron_compat_tests.m/.log | Private generic geometry/block callback and UI compatibility; eight independent cases |
| run_neutron_cantilever.py; neutron_cantilever_run.m; neutron_cantilever_original.json/.log/.mat | Full original native example execution and observations/results |
| neutron_capture.m; neutron_original_stages.mat; neutron_frozen_stages.npz | Independent original-function stage capture, frozen before extraction |
| neutron_joint_scaled12/; neutron_scaled12_run.m; neutron_scaled12.json/.log; neutron_scaled12_original.mat; neutron_scaled12_trace.mat | Selected qualified source, full execution and independent numeric stages |
| neutron_scaled12_repeat.json/.log/.mat | Independent second full qualified process, bitwise equal fields/observations/coefficients |
| neutron_selected_qualification.patch/.json | Exact original-to-selected solver diff and hashes; use this final index rather than earlier experimental diffs |
| neutron_parity_criteria.json; neutron_rank_sensitivity_plan.json | Frozen original criteria and bounded source-only rank study before local adoption |
| neutron_coordinate_qualification.json; neutron_last_cutoff_backend_audit.json; neutron_local_cutoff_parity.json; neutron_physical_update_parity.json | Numerical investigations, failures, coefficient discrepancy and physical-field parity |
| neutron_independent_physics.json; neutron_derivative_unit_audit.json; neutron_octave_backend.log | Independent quadrature, geometry, derivative, unit and backend evidence |
| neutron_tests_final27.log; neutron_nbconvert_verified.json/.stdout.log/.stderr.log; neutron_execute_notebook_verified.py | Final 27-test run and actual shell-free nbconvert exit0 evidence |
| neutron_experiments.npz/.json; neutron_figures/ | Actual numeric comparisons and six PNG/SVG/PDF figure groups with numeric provenance |

Data class is **author_defined_simulated**: this session ran the author's original native generator; observations were not an existing repository measurement file. There is no claimed real neutron experiment. The redistribution is an output of the GPL-licensed author simulator and is distributed with the source attribution; no third-party measured-data rights are asserted. Public NPZ SHA256 is `b7d256ece33de13c47eaf69c52661024c108c6fda936f70e5d3cc01b5b40af28`. It retains all1798 native noisy line observations unchanged, float64, and legitimate geometry/calibration only. The source PDF and author source are not solver inputs.

Hidden generator conditions: 40x40 grid over x=[0,.02]m,y=[-.005,.005]m; Saint-Venant cantilever t=.006m,h=.01m,l=.02m,E=200GPa,nu=.3,load2000N,I=t*h^3/12. Strain xx=P(l-x)y/(EI), yy=-nu*xx, xy=-(1+nu)P(h^2/4-y^2)/(2EI). Stress-free spacing is 4.056+.0168 exp[-.5*x^2/(.015/2)^2-.5*(y-.007)^2/.006^2] angstrom. Original `rng(1203)` Gaussian observation noise SD is 4.056e-4 angstrom. Neither load, object formula nor simulator seed is exposed in README/raw inputs. The known nominal 4.056 angstrom is an independent material calibration, not estimated hidden endpoints. All lattice-spacing pixels remain unknown.

`src/generate_data.py` explicitly regenerates the same acquisition geometry, physical fields and noise law with independent quadrature. Its NumPy Gaussian random stream is not claimed bitwise equal to original Octave observations; it never replaces the bundled source-generated sample. Geometry-only basis placement uses Python MT uniforms matching original Octave uniforms, independently of Gaussian target noise. Instrument grid, normals and ray coordinates are public; truth is not.

## Actual original and qualified executions

Original command recorded by `run_neutron_cantilever.py`: WSL Ubuntu-24.04, octave-cli8.4.0, native source examples and private compatibility path, single thread. Original mathematical files were byte unchanged. Generic compatibility supplies polyline intersection with segment indices and struct.data block callback; plotting is suppressed by UI-only stubs. This is a compatibility-qualified original execution, **not a claim of running MathWorks polyxpoly/MATLAB unchanged**. No domain reconstruction is implemented by those helpers. The full example includes native simulation, joint original solver, and constant-d0 linear GP baseline. It completed exit0 in19.1082053s on the exact bundled1798 observations. Eight generic cases independently cover rectangle, slanted segment, endpoint, tangent, corner, collinear segment, NaN-separated indices and block callback behavior. Native ray intersections additionally agree with independent analytic slab geometry to3.82e-17m.

Selected qualified command (literal argument list in neutron_scaled12.json): `wsl -d Ubuntu-24.04 --cd /tmp/imaging101-neutron/neutron_joint_scaled12/examples --exec env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 octave-cli --quiet ../../neutron_scaled12_run.m`. Exit0,18.6497066s; second independent process exit0,21.3516313s, bitwise equal output. The original and qualified runs consume exactly the same noisy observations, geometry and seed-defined basis. No easier acquisition, cropped data or substituted physical model is used.

Octave numerical backend was inspected: reference BLAS and LAPACK3.12.0 at `/usr/lib/x86_64-linux-gnu/blas/libblas.so.3.12.0` and corresponding lapack path, SVD driver gesvd. **It is not OpenBLAS.** Octave image/signal provide environment compatibility only; no upstream package is imported by the shipped runtime. Original commands, timings, statuses, stdout/stderr and actual MAT files are private. Routine UI-shadow warnings are retained in logs, not concealed.

Clean runtime is CPU Windows10 build26200, Python3.11.11, NumPy2.4.6, SciPy1.17.1, Matplotlib3.11.1, single-thread settings. Actual executable: `D:/yan1/agent/AutoSkill/imaging-101-maintainer/research-runtime/Scripts/python.exe`. No upstream domain package or MATLAB/Octave is needed by solver, generator, tests, Judge or notebook. Private `neutron_selected_qualification.json` records actual executable/platform/version and hashes. The notebook prints `sys.executable`; its kernel is the research runtime, even though the nbconvert frontend can reside in Anaconda.

## Source discrepancies and bounded numerical study

The following changes were investigated mathematically and authorized as a separately qualified method, before adopting corresponding local code:

| Source behavior | Selected qualification and independent evidence |
|---|---|
| Mixed Hessian uses plus nu although physical forward/Jacobian uses minus nu | Differentiate the same bilinear Eq.19 forward model: use minus nu. Quadrature and finite differences verify it. |
| Prior negative-gradient sign is positive and Hessian prior precision is doubled | Use negative prior gradient and one precision. Direct objective gradient/Hessian differences verify signs/factors. |
| Eq.21/source combines convex mean mixing and a vector called a Newton direction | Use additive Newton displacement with genuine actual-objective line search. Paper direction/proposal terminology is ambiguous; this is a custom Newton implementation for Eq.19, not unchanged Eq.21 reproduction. |
| Constant GP initial normal-equation RHS suffers cancellation, condition3.0981311979e10 | Direct thin QR solves the same augmented least-squares objective, without forming normal RHS. Independent residual/gradient checks; nominal field error4.24e-10. No target is used. |
| Joint source treats relative1e-4 noise as absolute angstrom SD | Convert using known nominal4.056 angstrom, giving physical4.056e-4 angstrom, as native generator and paper experiment specify. Original source comparison retains its original convention. |
| Raw Hessian has mixed physical coefficient units and condition3.7439e24 | Fix target-independent initial-Jacobian/prior column scaling, congruence g_z=S^Tg,H_z=S^THS, physical step S*delta_z. Scalar scales and rank convention are frozen before local adoption. |

Original raw-Hessian reproduction **failed** physical parity: same frozen H/g produced about6.08% backend direction difference; fully recomputed direction difference about7.5%, lattice field difference.000784 exceeding the predeclared2e-5 boundary. Local g/H/J relative discrepancies were only1.18e-12/4.12e-16/2.92e-15. This failure is preserved, not relabeled success. Raw H truncation retained only47/227 modes and depended on mixed coefficient units.

The first scaled source-only cutoff1e-10 retained144 modes and converged to cost5162 with truth rangeNRMSE xx.1613,xy.7116,yy.1618,d0.1437: failed quality. A final **bounded source-only** study tested1e-12 and1e-14, as frozen in neutron_rank_sensitivity_plan.json. Selected1e-12 retained188 modes, three accepted updates, cost[209767.06031396365,973.1602373912231,971.6844452832459,971.6844452801005]. The1e-14 source variant had truth errors [.00583068449,.03021438628,.00615065761,.01136626610] but more coefficient-direction backend sensitivity (4.809e-4 vs2.107e-4). The1e-12 choice defines the released numerical rank scheme; no judge/parity threshold was relaxed. Earlier unscaled, unstable-initialization and wrong-noise variants remain clearly labeled exploratory artifacts in P.

Initial scaling uses only nominal constant/zero-strain mean, initial Jacobian geometry and prior. The mixed curvature has zero diagonal. Tests explicitly perturb all observations and verify the initial Hessian diagonal is unchanged. Normalized sine bases have units m^-1; density coefficients have angstrom*m, Airy coefficients m^3, potential m^2, and strains/normalized tractions dimensionless. Corresponding S entries carry these coefficient units; scaled variables, gradient and Hessian are dimensionless. A known power-of-two coefficient unit transformation and congruence gives exactly invariant physical direction with rank188. A cutoff applied to the unscaled physical Hessian is not claimed unit invariant.

## Equation, implementation and test traceability

| Paper role | Upstream file/function | Local implementation | Validation consumer |
|---|---|---|---|
| Sections3.1/3.2 spectral GP, normalized basis, spectra | bayevar_airys/placeBases.m:placeBases, d0basisfunc.m:d0basisfunc | physics_model.circular_modes/place_bases/density_basis | test_original_bases_and_analytic_integrals; bitwise wavenumber/spectrum fixtures |
| Airy stress/strain and force equilibrium; Eq.23 mapping | bayevar_airys/dbasisfunc.m:dbasisfunc, epsbasisfunc.m:epsbasisfunc | derivative_basis/strain_basis | test_original_derivatives_product_order_prediction_jacobian; independent physical_fields |
| Eq.15 normalized boundary traction | bayevar_airys/tracbasisfunc.m:tracbasisfunc | traction_basis | original tb fixture; independent traction/unit tests |
| Eq.14 and Appendix B analytical line integrals | bayevar_airys/intd0basisfunc.m:intd0basisfunc, intprodbasisfunc.m:intprodbasisfunc | integral_density/integral_products/acquisition_operators | id0/ip/ip2 fixtures; independent Gauss-Legendre quadrature |
| Bilinear prediction and first derivatives | bayevar_airys/measurement_model.m:measurement_model, measurement_model_J.m:measurement_model_J; var_inference.m | prediction/jacobian | original prediction/Jd/Jc fixtures |
| Eq.19 physical likelihood and Gaussian prior | qualified var_inference, exact diff retained | objective/derivatives | frozen first_gradient/Hessian/Jacobian; independent central differences |
| Constant nominal GP mean | original initialization; qualified direct thin QR | solvers.initialize | frozen what/mu/SLambda and physical update tests; independent nominal-fit audit |
| Qualified Newton/scaling/rank and actual-cost search | neutron_joint_scaled12/bayevar_airys/var_inference.m | newton_direction/solve_weights | first physical update fields, congruence/unit tests, full-source field parity |
| Full fields and evaluation | selected source native example; actual simulator truth | reconstruct; independent Judge and field_metrics | full pipeline, analytic metric cases and strict Judge contract cases |

Source file identities beyond var_inference are preserved in archived full source; no equation-number shortcut substitutes for the explicit formulas in approach.md and code. Original var_inference SHA256 `b5488fe1c924420b7f54d4c3f74d8b0876fe5a6d6e80f4b99a528231cac54055`; selected qualified SHA256 `bf167acc2880bf3698e79408a1f195207232d87dca1453dd16d885a978edf81f`; exact diff SHA256 `0fcbf4b0b23204519d97dd9fc264330775746cbcab6ae901cc5ecd07b4fe492f`.

## Frozen criteria, independent physics and measured parity

The original independent fixtures and criteria were frozen before local extraction. Stage relative/absolute tolerance2e-8, geometry absolute tolerance1e-13m, physical full-field and update rangeNRMSE tolerance2e-5, derivative relative tolerance2e-5. Subsequent qualification refroze source-only results before corresponding local adoption; physical tolerance stayed2e-5. Coefficient-direction parity was never retroactively declared passed.

Independent 64-point Gauss-Legendre integration versus analytic source product integrals has maximum normalized discrepancy2.1268975841e-12 over four nondegenerate native rays, three basis pairs and all three derivative components. Rectangle intersections agree to3.82e-17m. Physical truth through an independent discretization matches the known noisy law; observation noise is not used as a hidden calibration. Central objective gradient relative discrepancy1.9738190e-8 and negative-gradient/Hessian discrepancy1.81571622e-7 at scaled FD step1e-3. Unit-congruence invariance discrepancy0.

Original-versus-local coefficient direction differs2.1069033638746793e-4 relative because of weak, nonidentifiable modes. **This is disclosed, not certified equal.** Actual physical update discrepancies for the first three states are d0[6.8972e-8,1.8651e-8,1.79066e-8] and strain[3.19785e-8,9.36341e-9,9.84992e-9] rangeNRMSE. Full selected-source/local field differences: lattice1.7906608248e-8, strain9.8499212075e-9. All pass the unchanged2e-5 physical criterion. Full objective histories differ about1.4e-4 in the weak coefficient prior; no exact objective-path parity is claimed. At identical frozen source coefficients, objective derivative/formula tests and independent finite differences pass.

## Reference ownership, metrics and notebook measurements

`evaluation/ground_truth.npz` contains actual native simulator strain and lattice_spacing, not solver reconstructions. Hidden notebook references are separately named: original_source.npz contains actual original heuristic fields and its linear_strain constant-d0 GP; qualified_source.npz contains actual selected qualified source fields/history; fresh.npz contains actively recomputed local fields/history. They are consumed by notebook comparisons. `test/fixtures/qualified.npz` contains exact-bound dm/entry/exit/nhat/L; stages.npz contains original basis/integral/derivative/prediction/Jacobian captures; trace.npz contains selected source mean/prior/initial derivatives/scales/direction. Every retained fixture key has a numeric test consumer; full unused capture arrays remain private.

Metrics use cosine NCC without mean subtraction and dynamic-range-normalized RMS error, all1600 pixels of each field; no crop, scale, offset, phase or translation alignment. Per-field thresholds are0.9*selected-sourceNCC and1.1*selected-sourceNRMSE, all eight required. Constant nominal spacing/zero strain and missing components fail. No vacuous constant-field metric is used: all actual truth fields have nonzero dynamic range.

| Computed result | xx NRMSE | xy NRMSE | yy NRMSE | d0 NRMSE |
|---|---:|---:|---:|---:|
| Reproduced original source heuristic | .006691719784 | .008492776080 | .006735442024 | .011171978487 |
| Reproduced constant-d0 GP | .338278442616 | 1.267103530792 | .774495905498 | .358745398504 |
| Reproduced selected qualified source | .011808554936 | .050517231913 | .012109930573 | .014767794958 |
| Fresh local selected method | .01180853934 | .05051717798 | .01210991492 | .01476778481 |

These are actual session computations on the bound acquisition, not historical paper numbers. Full precision, data hash association and all sensitivity measurements are in P/neutron_experiments.npz/.json and the final inventory. Whitened fresh observation residual RMS is.99803036; maximum normalized traction is4.833e-6. Independent physical accuracy and source parity are separate checks. The original heuristic's better error on this particular simulated object does not establish that its erroneous Hessian is the objective derivative.

Canonical notebook has six chapters: problem, data, physics, method, results and conclusion. It actively calls local reconstruct for full native observations and independently recomputes cutoff1e-10 and1e-14 variants. Fixed arrays are used only for explicitly identified source comparisons. Six actual figure groups show geometry/residuals, four-field truth and methods, signed errors, metric comparisons, objective/singular spectrum/geometry, and active rank sensitivity. Signed fields/errors use zero-centered diverging scales; method panels share physical ranges. All six figures were visually inspected and exported PNG/SVG/PDF with real numeric NPZ/JSON.

Final verified command: actual research `python -m nbconvert --to notebook --execute notebooks/neutron_joint_lattice_strain_fields.ipynb --output neutron_executed.ipynb --output-dir output --ExecutePreprocessor.timeout=600 --ExecutePreprocessor.kernel_name=imaging101-research`, with private JUPYTER_PATH and explicit private export directory. `neutron_execute_notebook_verified.py` launches this exact argument list using subprocess.run(shell=False,capture_output=True), records the actual child returncode/stdout/stderr, and only adopts the notebook after exit0 and cell checks. Actual final returncode is **0**, elapsed73.2096843s; evidence is neutron_nbconvert_verified.json and its stdout/stderr logs. Full selected reconstruction is approximately20s and active two-case sensitivity approximately41s, hence600s is justified. All five code cells have execution counts and no error outputs. Earlier PowerShell exit1 from the global frontend was insufficient evidence and is superseded by this actual verified subprocess exit0. Any warnings remain in stderr. The executed artifact replaces the canonical notebook; temporary output notebook was removed. Figure1 explicitly labels90 evenly sampled displayed rays, with all1798 used in inversion. Figure2 retains shared truth color ranges, adds appropriate colorbar saturation triangles, and explicitly states that all metrics use full numerical values.

## Final checks and gates

`python -m pytest test -q --basetemp=<P>/neutron_pytest_handoff` on actual research runtime: **27 passed in44.21s** (neutron_tests_final27.log). This includes source stages/full pipeline, analytic physics/FD/units, independent complete generator contract, strict missing/key/shape/float32/complex/nonfinite submissions, poor valid predictions, foreign-current-directory Judge, unavailable hidden assets and report-write failures. Main writes both declared arrays; standalone Judge passes the fresh output. Judge imports no solver and returns0/1/2 as specified.

Source and plan review checked actual callable signatures, constants, signs, factor0.5, objective/noise units, normalized basis coefficient units, coordinate congruence, rank, line search and stopping. README contains complete nine-array public NPZ table/acquisition metadata and both output quantities. Privacy audit found no GitHub URL, hidden seed/object/load, hidden truth path or target-derived calibration in public materials. Runtime imports only standard library, NumPy, SciPy and Matplotlib, with no cross-task imports, upstream imports, ready-made inverse operator or reference-array shortcut. Figure creation is separate from solver; library imports do not switch backends.

| Mandatory gate | Status and evidence |
|---|---|
| Paper/version, exact remote revision/license/data hashes and traceability | PASS; source record, hashes and mapping above |
| Actual original plus separately qualified exact-observation run | PASS for explicitly qualified scope; full original comparison retained, source-only repeat bitwise equal |
| Frozen numeric parity and independent physics | PASS for stated physical-field/update criteria; coefficient direction difference disclosed, not falsely certified |
| Constants, units, precision and all unknown fields preserved | PASS; explicit code/tests/plans and unit-congruence audit |
| Main, fresh strict Judge and adverse contract cases | PASS;27 tests and actual output report |
| Public sufficiency/privacy and no target-derived calibration | PASS;9 public arrays plus acquisition metadata, independent nominal spacing |
| Minimal dependencies and local forward/inverse mathematics | PASS; CPU runtime without upstream packages; call-role audit |
| Active canonical nbconvert notebook and computed comparisons | PASS; final executed artifact/log, six inspected figures |
| Metric/reference identities and justified boundaries | PASS; all eight metrics, actual truth, no alignment; poor outputs fail |
| README/plans/source/generator/Judge semantic consistency | PASS; explicit qualification, signs, coefficient units and call flow reviewed |
| Final inventory, license and temporary file cleanup | PASS; final private inventory, LICENSE/NOTICE, no PDF/archive/checkout/temp notebook in task |

Limitations: author-defined simulated cantilever, two-dimensional plane-stress/equilibrium model and known support/zero top-bottom traction. Finite spectral basis and selected rank convention determine the numerical benchmark. This does not validate arbitrary three-dimensional materials, unknown traction or real experimental neutron calibration. The retained source issues and unsuccessful numerical variants are evidence of the qualification, not passing unchanged reproduction claims.
