# Provenance and validation

## Binding, independent chronology and scientific scope

The unknown is an 80-bin normalized phonon density of states for the existing experimental vanadium observation. Acquisition is the original neutron momentum–energy intensity/variance map, not simulated counts. The published selected method is the full incoherent single/multiphonon and multiple-scattering self-consistent estimator, including all four phonon orders, thermal populations, density-dependent recoil exponent and every measured Q bin. No lower-dimensional substitute or simple one-kernel deconvolution is used.

Paper: Lin, Islam and Kresh, JOSS3(21),440, published2018-01-30, DOI10.21105/joss.00440; primary two-page final paper read, including its explicit self-consistent one/multiphonon/multiple-scattering description. Its primary cited methodological paper is Kresch et al., “Neutron scattering measurements of phonons in nickel at elevated temperatures”, PRB75,104301(2007), DOI10.1103/PhysRevB.75.104301. The bound observation is vanadium, not the nickel or aluminum experiments. JOSS source author spelling is Kresh. Initial draft reference title incorrectly described aluminum; corrected against the actual primary paper before handoff.

Official author repository https://github.com/neutrons/multiphonon, remotely verified exact main/tree revision **1ff49523b22f00d1a62b7be023579d0df1c87836**. This is a current source implementation associated with the published package, not a claim that its revision equals the2018 archived release. Private selected source checkout: worker1_math_physics_astro/multiphonon_original. API tree/download source index: multiphonon_tree.json and multiphonon_download_manifest.json. Source MIT license copyright2017ORNLNDAV is retained verbatim in LICENSE. Source/data were downloaded by exact revision; all selected upstream files have SHA256 in that manifest. The experimental data are distributed in the source repository's tests/data; no separate acquisition run ID or per-data license declaration was found, so no invented date/facility/run identifier or additional image license is claimed. This staged package is local research, with no external publication action.

Exact data: tests/data/V-iqe.h5,535630bytes, SHA256 **f47c2849b5336a76a025fa215009e028c55f5f84c059b7554a9eb021b9f436f9**. The original test backward/sqe2dos_TestCase.py:test2a explicitly labels it “V exp” and binds T300K,M50.94u,Ei120meV,Ecut55,C_ms0.2,elastic cutoff(-12,6.7), analysis grid−15..79step1. These exact constants are used. The differently configured example YAML with C_ms0.26/Ecut40 is not silently substituted.

Public raw NPZ SHA256 **76ad976e5c49d56a168f299208e8df2fc53ab0620176e42209416e10a829f5e0**. Native float64 intensity, variance, energy and Q retain their original numeric precision; no rounding, synthetic regeneration or replacement by final reconstructions. Native NaNs are preserved, with explicit measured_mask equal to finite intensity. Variance is finite/nonnegative on measured cells. Instrument-independent energy and momentum axes are visible. All calibration, finite-grid conventions and nuisance area normalization required by the target are in README. No target-derived density, Debye–Waller factor or source-fitted covariance is an input.

Chronology: actual default source run, actual tighter source run, and independent physical investigation preceded formal criteria in v_phonon_formal_criteria.json (UTC2026-10-30T10:23:59.876616). Source stage fixtures were captured by unmodified original functions in capture_v_upstream.py, then local probe/extraction began. The original endpoint-mass concern was investigated rather than silently renormalized. Formal scientific boundaries were not widened after local results. Stage rtol1e-9/atol1e-10 and full density dynamic-range NRMSE1e-7 were fixed privately; final density differences are about5e-17 absolute.

## Actual upstream runs and environments

Maintainer-only original oracle eels-oracle: WindowsPython3.11.11,NumPy1.26.4,SciPy1.10.1,h5py; source package on private sys.path. Official histogram0.4.3 noarch conda package is a histogram container/HDF reader only. Its63721byte package and metadata are archived as histogram_conda_metadata.json and histogram-oracle-package; extracted with existing conda_package_handling, never installed in the shared runtime. No Mantid function is used. A generated _version.py build metadata file was supplied because raw source checkout lacks its normal generated version; no original numerical function was changed. Single-thread CPU environment OMP/OPENBLAS/MKL/NUMEXPR=1.

| Actual run | Command/wrapper | Process seconds | Exit | Archive |
| --- | --- | --- | --- | --- |
| Default full source | eels-oracle/Scripts/python.exe run_v_phonon_original.py | 1.2453660001046956 | 0 | v_phonon_original_full.json/.log |
| Tighter full source | same plus --tight | 1.4279484000289813 | 0 | v_phonon_tight_original_full.json/.log |
| Independent source stage capture | eels-oracle/Scripts/python.exe capture_v_upstream.py | 0.818878100020811 | 0 | v_phonon_capture_upstream.json/.log |

Default max20/tolerance1e-4 actually performs6updates (core0.3709843s). Tighter max100/tolerance1e-10 actually performs14; it is a labeled comparison, not substituted default. Source seed: none; method is deterministic. Original HDF round outputs and final-dos.h5 are privately retained in v_phonon_original_work and v_phonon_tight_work. Expected source divide/invalid intermediate warnings at unsupported/missing bins are retained in original logs; final DOS and variance are finite. Source variance at yielded intermediate states differs from final-dos.h5 after residual uncertainty addition; both roles were identified, with final variance captured correctly.

Fresh cleaned runtime uses the unchanged validated shared research runtime Python/NumPy2.4.6/SciPy1.17.1/Matplotlib3.11.1. No upstream domain package is installed or imported. Requirements document only actual task subset. CPU single-thread runs validated.

## Equation, original function, local function and independent test mapping

All source paths below are relative to original src/multiphonon. Paper summary describes the self-consistent estimator; numerical conventions are defined by the exact current source rather than invented paper equation numbers.

| Physical stage | Original file/function | Local implementation | Frozen/independent check |
| --- | --- | --- | --- |
| Finite measured-window interpolation | sqe/__init__.py:interp | preprocessing.preprocess | original processed intensity/variance and missing masks, exact parity |
| Thermal gamma and zero endpoint | forward/phonon.py:gamma0 | physics_model.gamma0 | independent Debye h refinement, dimensional recoil scaling |
| Thermal A1 and central convolution orders2–4 | forward/phonon.py:computeA1E,computeAnESet,AnE_from_n_1 | physics_model.spectral_orders | captured initial/final A arrays; detailed balance; normalized higher orders |
| Recoil Debye–Waller exponent | forward/phonon.py:DWExp,recoilE | physics_model.debye_waller | captured DW, independent Q² law and units |
| Strict neutron triangle | units/neutron.py:e2k and ms.py | physics_model.kinematic_valid | zero-Q finite-loss exclusion, elastic accessibility, all observed cells inside triangle |
| SP/MP/MS coupled forward | forward/__init__.py:dos2sqe,ms.py | physics_model.forward | captured initial/final SP/MP/MS/total; independent Judge forward |
| Q-summed multiplicative density update | backward/singlephonon_sqe2dos.py:sqe2dos | solvers.single_update | captured first update and every actual round |
| Area scaling, correction and stopping | backward/sqe2dos.py:sqe2dos | solvers.reconstruct | all6default rounds, full14tighter final, final variance |
| Conditional whole-Q target | original forward on actual source densities, private v_phonon_quality.py | independent evaluation/judge.py | prior fails, default/tighter pass, all output constraints and malformed contracts |

No fixture is generated by local src functions. test/fixtures/upstream_stages.npz comes from actual original source capture and is consumed by comparing_test. It contains original processed maps, initial/first/final densities, every default round, spectral orders, thermal factors and contributions. notebooks/reference_outputs/default_source.npz is the original stage capture separately labeled for Notebook comparison. tight_source.npz contains the original14-round density/variance plus independent analytic Debye refinement results; consumed by tests and Notebook. evaluation/baseline_reference.npz is a standard-method result, **not actual truth**, used by contract tests and maintainer comparison; Judge deliberately does not force identity to it. No ground_truth.npz is invented.

## Scientific target and numerical evidence

The source is an incoherent approximation, not a noise-limited model of each detector cell. All inelastic measured Q cells yield about13.855% relativeL2 and mean variance-normalized squared residual148.8. Over the entire−15..79window including the unmodeled elastic peak, relativeL2 is94.5602%. Both complete maps and diagnostics remain in Notebook. No hidden fitting mask is used.

The scientifically bound target is instead the source's conditional complete-Q marginal over7..79meV with its declared DOS constraints, support, Debye convention and coupled model. Initial prior marginal relativeL2=0.257257; reproduced default=0.007229869879588262; tighter=0.007244541488038947. The frozen boundary0.00795285686754709 is1.1times independently reproduced default and is not based on cleaned self-comparison. The tighter trajectory need not improve every metric; it is retained honestly. Any other admissible inverse with the same or better public residual is accepted. The original multiplicative zero trapping at support bins is not a hidden identity requirement.

The A1 mass is about0.98275044(default),0.98275039(tighter). We do not assert false exact conservation or silently normalize it. Independent analytic Debye densities on h=1,.5,.25,.125meV yield masses0.9920397819767498,0.9959814972443652,0.9979811110906721,0.9989881413719817. The defect halves, establishing first-order endpoint quadrature bias. Higher orders have unit discrete mass and all orders obey detailed balance to about1e-17. These source limitations are explicitly public.

Local resampled intensity/variance and masks are exact matches. Spectral/forward stage differences are below1e-16; every default density round below5e-17; final variance maximum difference1.61e-20. The independently implemented Judge and source/local forward agree at about1e-17; fresh marginal metric0.007229869879588249. Norm error0, negative/support errors0, low-energy error4.34e-19. All constraints are evaluated, without source solver imports.

## Tests, active Notebook, review and inventory

Final tests and CLI evidence are indexed in v_final_tests_confirmed.json/.log (37passed,9.98seconds) and v_final_cli.json/.log and v_final_judge.json/.log (both exit0). Judge contracts include missing files/keys, extra key, wrong shape, integer/complex dtype, NaN/Inf, zero/scaled/negative/reversed/rolled/prior/zero-energy-point failures, float32 valid result, malformed bool/string/array/null/NaN/Inf/negative boundaries, empty/missing/extra metric configuration, wrong comparison, foreign working directory, unavailable evaluator observation and report-write failure. Config numeric type excludes bool explicitly.

Earlier development failures are retained honestly: default pytest system temp had a permission failure (private explicit basetemp resolves it); an A1 assertion accidentally used the tighter run's rounded mass for the default run (replaced by independent default source fixture and physical bias interval); and one Inf-case subprocess did not produce a report in v_final_tests, without a captured launch diagnostic. The exact same Inf submission immediately returned correct error2/empty metrics in a direct rerun. The test helper now records return code/stdout/stderr if a report is absent, and the complete final37tests passed without retrying individual tests. No scientific boundary was changed.

Full active nbconvert command/run record: v_nbconvert_final_figures.json/.log, exit0,5.6350451seconds, timeout300s justified by measured source/local subsecond inversions plus plotting. An explicit private kernelspec points to the validated shared runtime; portable shipped metadata is python3. All default and tighter reconstructions are computed fresh from raw data. The earlier provisional execution accidentally selected global python3; it was not used as final environment evidence. Another first run asserted an inelastic residual bound against an all-window residual including elastic scattering; the diagnosis led to explicitly retaining both distinct metrics, not altering any target threshold.

The canonical executed Notebook has #1Problem/#2Data/#3Physics/#4Method/#5Results/#6Conclusion with actual rendered headings (space after hash), six density comparison groups, approximate uncertainty label, all raw/forward/residual maps, both momentum and integrated residuals, every default iteration, source and fresh tighter comparisons, detailed balance and quadrature refinement. Five embedded actual PNGs were extracted privately and individually opened. Figure4 was improved with true inelastic/residual insets to avoid hiding the target behind the large elastic feature, and Figure5 ticks corrected; both updated figures were opened again. PNG SHA256s in v_notebook_figures/manifest.json identify canonical outputs.

All README/plans/Notebook formulas were extracted, checked for control characters and preprocessing pitfalls, rendered with independent MathJax3.2.2 and KaTeX0.16.22 (92formulas,0failures), and actual two-page browser-rendered mathematical sheets were opened and semantically inspected. README originally used escaped-brace delimiters, corrected to lbrace/rbrace before final rendering. Logs/SVG/screenshots are math_reviews/vanadium_phonon_density_of_states. Plans/signatures, signs, units, default/tighter distinction, normalization, NaN mask and full data flow were reviewed against actual source. main.run actual signature is task_root=None. No library module selects a Matplotlib backend.

README and raw observations plus dependencies define the complete public mathematical target. No GitHub/raw-content/solver download URL occurs in public material. Raw arrays contain measurements/calibration only, not a density, hidden DW factor, output-quality mask or final fitted scale. Standalone Judge has no src/domain-library imports. Call-role audit confirms explicit interpolation, direct convolution and mathematical formulas using permitted general NumPy/SciPy primitives.

Private original paper SHA256 **49b19aab3ba8937ef0f458a9e61e5dc61cd1f9982d8148518d288f85ffb147c7**, fixture/source/environment/run records and final stage file hashes are in v_task_evidence_manifest.json outside this task. Complete upstream checkouts, HDFs, archives, capture scripts and logs remain maintainer-only. Canonical Notebook keeps executed outputs; its temporary output copy was removed. No synthetic-data generator is present because the observations are existing experimental data. No original299task was modified.

| Mandatory gate | Local status and evidence |
| --- | --- |
| Paper/exact revision/data hashes/mapping | passed; above and private source manifest |
| Original exact-data complete selected run | passed; default/tighter/capture exit0 |
| Independent fixtures and physics | passed; frozen criteria and comparisons |
| Precision/constants/units/conventions | passed; native float64/NaN data, public model and tests |
| Fresh CLI and strict standalone Judge | passed; dedicated final records and report |
| Negative/malformed/evaluator contracts | passed; final tests |
| Public solvability and no leakage | passed; complete mathematical target and visible mask, no source URL |
| Local explicit math/shared runtime | passed; no upstream package dependency |
| Full tests | passed; dedicated final record |
| Active complete Notebook and actual figure review | passed; nbconvert exit0 and five actual figures |
| Provenance/metrics/nontruth labels | passed; source/fresh/tighter/diagnostic roles explicit |
| README/plans/implementation and actual math review | passed;92dualrender0 and semantic review |
| Template/final inventory | passed; worker manifest and generated-cache cleanup |

Parent independent acceptance and harness integration are a separate review; this worker does not modify public registry or release tasks.

## Parent kinematic-formula semantic correction

Parent semantic review found that the README wave-number conversion denominator incorrectly placed10^10 under the mass rather than multiplying hbar/m, changing the public formula by10^20 even though both renderers parsed it. This was a documentation physics error, not a code error. The corrected direct SI formula is sqrt(2*m_n*10^-3*q_e)/(hbar_n*10^10), in inverse angstrom per sqrt(meV), approximately0.6947. A new independent numerical test derives k=p/hbar from SI units and checks both neutron-triangle boundaries around10meV loss. It passed exit0; both renderers now pass96formulas. The corrected actual formula sheet was viewed. The complete source/data/evaluation/Notebook file bytes remain identical to the previously verified manifest, including every active cell and embedded figure. No inverse/reference/metric boundary changed, so a new full inversion or Notebook execution was unnecessary for this text-only correction. Evidence: v_kinematic_document_correction.json and v_readme_kinematic_constant_test.json/.log. This demonstrates why syntax rendering alone is insufficient for physical formula validation.


## Independent parent acceptance, batch13

Parent reviewed the complete300momentum by149energy real scattering map, native measured missing-cell mask, bilinear boundary protocol, temperature/recoil units, detailed balance, four direct spectral convolutions and uniform accessible-Q multiple scattering. All13095missing intensity/variance cells remain actualNaNs with matching publicmask, not invented zero measurements; parent export allows these explicitly identified gaps only. The publictarget is the complete measured momentum marginal over7..79meV, with nonnegative unitmass supportedDebye density, not forced hidden density identity or cellwise noise accuracy. Parent38tests, fullfreshCLI, foreign Judge and96formula dualrender passed. Parent read all canonical cells and viewed all five actual figure groups, retaining6/14update source/fresh paths, elastic-region and inelastic2D model residuals, all contributions, uncertainty limitations and independently convergent A1quadrature bias. Parent discovered an incorrect displayed neutron wavevector conversion; README now agrees with independently derived momentum/sqrtenergy units and numericaltriangle tests. Reconstruction/code/data/evaluator/notebook bytes did not change for that documentation correction. SourceA1endpoint bias is explicitly retained and vanishes under independentDebye gridrefinement; it is not falsely normalized or presented as physicaltruth. No unavailable experimentalDOStruth is invented.

Parent public allowlist export contains README, raw NPZ and requirements only; the shared environment is provided separately for this private staging test. Trusted harness score_submission evaluated the inert submission with isolated imports and exited0. This is an export/contract test, not an OS-isolated Agent trial. Private acceptance/vanadium_phonon_density_of_states/ records commands, reports, actual extracted canonical figures and final hashes.
