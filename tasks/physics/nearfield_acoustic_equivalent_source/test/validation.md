# Benchmark v2: convex complex second-difference objective

The historical20,000-step Condat-Vu source and its upstream parity below remain unchanged. Acceptance now uses the public explicit convex data-plus-vector-second-difference objective, with its actual selected35dB tuning coefficient, plus the original unchanged complex observation residual limit. No hidden map comparison is performed. The objective boundary4206091749400.686 equals the independently double-evaluated feasible standard quality plus1e-8relative numerical allowance. This removes dependence on an exact finite update trajectory without changing the acquisition model, derivative regularizer or physical complex scale. Original arrays remain unchanged.

Independent general L-BFGS refinement of the feasible standard field converges in431iterations to objective4206091653232.534 and pressure error0.030362611456669034, passing. Its initialization is explicitly the existing field: this is an independent numerical-refinement control, not a raw-only blind reconstruction. A separate zero-initialized10,000-iteration generic control reaches objective4206466849440.484, slightly above the budget, and fails; this failed control is preserved rather than relaxed into a pass. Zero and phase-rotated controls are rejected. An analytic scalar complex case independently verifies the second-difference norm objective minimum and factors.25unit/contract tests pass in2.07s; unchanged source/upstream parity evidence below remains separate.

Private evidence:nearfield_metrics_v1.json,nearfield_control.py,nearfield_v2_controls_refined.log,nearfield_v2_warm_control.json/log,nearfield_pytest_v2b.log. Notebook old method/reference error figures are historical reproduction diagnostics, not current acceptance. Its numeric flow does not index changed Judge keys; the current evaluator interface is executed through unit/foreign-CWD CLI tests without needlessly rerunning unchanged20,000-step reconstruction comparisons. No fresh blind agent solve is claimed.

# Provenance and validation: near-field acoustic equivalent sources

## Fixed scope and distinction from existing tasks

Recover a40x50 complex equivalent-monopole strength image from a9x12 single-frequency complex pressure plane. Parent formally reserved this task as nearfield_acoustic_equivalent_source. The frozen existing139 index was checked. The existing microphone_array_acoustic_source_power recovers nonnegative incoherent power from a cross-spectral covariance matrix C=sum(q h h^H); this task recovers coherent complex source amplitudes from p=Gq, preserving inter-source interference. Ultrasound/PAT and sound-speed tomography have different physical unknowns and observation operators. No old task was edited or imported.

Selected method: original published classical second-order derivative baseline, with LN and first-order TV comparisons. This is a selected-method benchmark, not a reproduction of every experiment in the30-page neural-network paper. The source's CS, PNN and VN are not represented as computed or substituted with zero maps. No supplied original notebook exists. The source README says no online source generator is published and dummy training data remain; no trained VN checkpoint is bundled. The fixed classical scope does not make a network-accuracy claim.

## Primary paper, source and legitimate existing measurements

Manuel Pagavino and Franz Zotter, **Data-driven and physics-constrained acoustic holography based on optimizer unrolling**, Acta Acustica9,46(2025), DOI10.1051/aacus/2025030, published24July2025,30pages, version of record. Official article https://doi.org/10.1051/aacus/2025030 and official PDF https://acta-acustica.edpsciences.org/articles/aacus/pdf/2025/01/aacus240143.pdf. Direct urllib/PDF downloads returned403; no PDF download is claimed. Official search-indexed full primary article text was actually read for Sections2,3.1,3.2,5.1,7.1,7.2 and conclusion. Archive nah_official_indexed_primary.json records the official indexed result, notably the amplitude/phase operations and absence of measured source truth. Parent independently read the official full primary paper. Auxiliary institutional/Crossref/OpenAlex fetches do not replace primary mathematical evidence.

Official original code https://github.com/manuelpagavino/nah-variational-network at **56dfb274db3a403e3e704ac911d1ee935f567c61**, remotely verified GitHub commit API and codeload download. API response nah_commit.json, source nah_source.zip, checkout nah_source/nah-variational-network-56dfb274db3a403e3e704ac911d1ee935f567c61. License MIT, copyright2025 Manuel Pagavino; exact notice retained in task/LICENSE. Repository data/src_data.pt contains existing measurements and is distributed under this repository license without a different data notice. Its `lsp` tensor is the only selected observation. Other plate/piston/quadrupole source tensors and loudspeaker-position patches are not public input and are not treated as truth. The Zenodo15852281/15852282 record currently supplies a711B placeholder rather than the measured data; this task binds the actual GitHub-commit data instead.

Paper Sec7.1 Eq39–42: swept-sine impulse response per independently driven loudspeaker, Fourier sample1150Hz, equalize each contribution's maximum magnitude and rotate that contribution's maximum-pressure phase to0, then add four contributions. The released lsp already contains this composition; no individual raw speaker field or absolute calibrated Pa is recoverable. Input retains its numerical amplitude and phase. Units U and U m³/(Pa s) are explicitly stated; not absolute pressure/volume velocity. Known speaker positions are geometry, not source-strength truth, and are not hidden-reference labels. There is **no ground_truth.npz** because no actual source-strength truth was measured.

## Equation/source/local/test traceability

| Paper or algorithm stage | Original file/function | Local function | Independent validation |
|---|---|---|---|
| Sec2 point-source expansion and free-space pressure transfer | dataloader/esm.py EquivalentSourceModel.setup_geometry/setup_propagator | physics_model.coordinates/propagator/forward/adjoint | exact captured G and pressure; independent analytic monopole, Hermitian identity and Helmholtz PDE finite difference |
| Sec7.1 Eq39–42 composed measurement | data/src_data.pt['lsp']; test.py lsp branch assigns esm.p | preprocessing.load_pressure | exact bound author tensor; fixed dtype, all108 samples; no synthetic generation |
| Sec3.2 Eq12 LN right-inverse | methods/benchmarks/ln.py LeastNorm.forward/tikhonov | solvers.solve(method='LN') direct SVD | full original numeric fixture and analytic SVD equivalence |
| Sec3.2 spatial derivative analysis priors | methods/benchmarks/tgv.py diff/diff_adjoint | solvers.difference/difference_adjoint | original complex128/random fixtures; independent complex128 adjoint; affine interior and zero-edge check |
| Squared-error gradient including2 | tgv.py grad | solvers.data_gradient | source gradient at0 plus independent directional finite difference |
| Sec5.1 regularization tuning | test.py init_model; conf/methods/{ln,tv,tgv} | solvers.solve lambda,smax,tau,sigma | exact original float32 stage constants; LN30dB, TV/TGV35dB |
| Condat–Vu complex dual-ball update | tgv.py proj_l2/forward | solvers.project_dual/solve | captured iterations1,2,10,100,101 primal+dual; all20000-step TV/TGV outputs; independent feasibility/idempotence |
| Reprojection and comparison | original direct Gq; paper Sec7 no measured q truth | local forward and independent standalone Judge equation | NumPy complex128 pressure residual, known metric cases, invalid/poor output tests |

## Discrepancies and explicit conventions

The original code calls the second-order penalty TGV, but actually applies a two-vector (Dxx,Dyy) with complex isotropic pixelwise norm. This differs from auxiliary-field TGV and from the scalar summed Laplacian suggested by paper shorthand. Both implementation and plans explicitly describe what runs. This is not silently advertised as the other formulations. Original zero padding/corner handling is retained and independently shown to be an adjoint. All derivative steps are in index units, with no omitted-but-claimed physical-spacing scale.

The paper's general fused CS-TV/TGV discussion uses mu=1, whereas the released conf/methods/tv/tv.yaml and tgv/tgv.yaml explicitly select **mu=0.0**, rho=1 for these standard comparisons. The task follows these released source configurations and states that extra spatial sparsity is absent. It does not claim bitwise reproduction of a mu=1 paper figure. Original second-order indexing triggers a deprecated non-tuple-indexing warning; the local code uses tuple slices with identical values, not an operator change.

Actual grid centers are inclusive endpoints0..825mm and0..600mm. Figure extents include half a pixel beyond the centers and labels use meters. Manual array repositioning and exact acoustic source mechanics are imperfect/unknown, as the paper notes; the ideal regular grid and free-field assumption are retained. q is an equivalent numerical distribution, not uniquely measured mechanical vibration or speaker outline. No target-dependent support, data fitting of geometry, hidden calibration, complex-gain fit or phase alignment is used.

## Upstream-first evidence and environment

The original implementation ran before local extraction on the exact bundled lsp observations. Private nah_original_screen.py invokes original EquivalentSourceModel and LeastNorm; nah_original_comparisons.py invokes original TotalGeneralizedVariation for both orders, exactly20000 steps with original tuned SNR. Only explicit IO/orchestration is private capture code. The complete original source functions execute; original solver math was not replaced. Random tests use seed22; the real-data reconstruction has no stochastic step. The nominal random parameter ranges have identical endpoints, so all acquisition values are fixed. Frozen stage capture precedes local extraction. Stage/full-output relative tolerance2e-5 was selected before local extraction; tests also use absolute2e-6 for stage arrays.

Oracle environment: isolated oracle-env Python3.11.11 Windows10.0.26200, NumPy1.26.4, torch2.9.1+cu128 CPU, one thread. Full torch numerical backend is archived in nah_oracle_environment.txt. Isolated hydra/torchmetrics/lightning-utilities dependencies are only original-import dependencies. Clean runtime shared Python3.11.11/NumPy2.4.6/torch2.9.1+cu128/Matplotlib3.11.1, CPU one thread, no domain reconstruction library. Task requirements contain only these actual runtime packages; pytest/Jupyter remain shared development tools. No GPU is required. All runtime modules/tests/notebook imports were reviewed: explicit local G, derivatives, gradients, projection and iteration; permitted torch SVD/eigh/norm and array primitives only. Judge is independent NumPy, no src/upstream/cross-task imports. No generated observations, so generate_data.py is intentionally absent.

Initial PowerShell stderr redirection displayed original indexing warnings as shell error records despite completed numeric outputs. To avoid misreporting shell status, the full original comparison was rerun through a private subprocess runner recording the actual child return code: **exit0**, 53.034s. Command/exit/environment are in nah_oracle_verified_exit.json, actual combined log nah_oracle_verified_exit.log. Original LN residual .21543346345424652; TV residual .04258209466934204; TGV residual .03036261908710003. lambda LN5626457600, derivative3163989504; smax393492.46875; tau3.2292165903086145e-12; sigma TV19359266816,TGV4841009152.

## References, numeric consumers and evaluation

evaluation/baseline_reference.npz contains original TGV method output complex64(1,40,50), not truth. notebooks/reference_outputs/upstream_methods.npz contains original LN/TV/TGV maps and is consumed by visualization.review. test/fixtures/upstream_methods.npz contains original pressure,G,all method maps and stage random cases, consumed by test_original_physics.py. test/fixtures/upstream_stages.npz contains separately captured101-step sequence and derivative cases consumed by the same parity test. No unused fixture/reference array copies were added.

Metrics use no NCC/NRMSE convention substitution: relative complex L2=||q-qref||/||qref||; magnitude cosine compares abs fields; reprojection=||Gq-p||/||p||. Each is defined in metrics.json/README/notebook. Boundary relative complex L2<=2e-5 and magnitude cosine>=.999999 are nonzero numerical reproducibility tolerances; the original itself is the reference, so zero self-error is not multiplied to set a zero error boundary. Observation threshold .03188075004145503 is1.05 times measured original residual. Independent complex128 reprojection is .03036261152047294; single-precision visualization residual differs by rounding, not a different physical normalization. Actual fresh LN/TV/TGV maps are bitwise equal to original captures (complex relative error0 for each). No source-accuracy claim follows from this agreement.

Judge checks exact complex64 shape/key, finite arrays, nondegenerate reference, finite/range-valid nonboolean thresholds, all-finite numeric metrics, and actual-truth priority if an independent truth file is ever supplied. Existing malformed truth never falls back. It rejects wrong phase, sign, amplitude and zero; malformed inputs yield empty metrics/errors exit2, valid poor outputs exit1, pass exit0. Hidden paths anchor to the Judge; relative prediction/report paths belong to caller. Foreign-CWD/report-parent/write-error behavior is tested.

## Final actual execution and figures

Commands in nah_final_checks.json: python -m src.main **exit0 23.869s**; standalone Judge **exit0 0.143s**; python -m pytest test/ -q -p no:cacheprovider --basetemp <private task-specific path> **37 passed, exit0 45.542s**. Native pytest remains usable; the private basetemp avoids this host's unrelated shared temporary-directory ACL issue, without changing conftest/tempfile behavior. Unit/contract and original/physical full-pipeline tests are all under test/.

Final complete active notebook via python -m jupyter nbconvert --to notebook --execute notebooks/nearfield_acoustic_equivalent_source.ipynb --ExecutePreprocessor.timeout=600 --output nah_executed.ipynb --output-dir <private archive>: **exit0 112.419s**. Kernel imaging101-batch5-material points to the shared tested environment; source metadata and printed versions verify it. Canonical notebook contains all7 executed code cells and exactly5 embedded PNG groups. Actual fresh LN/TV/TGV,20000 derivative iterations,30/40dB sensitivity and independent Judge execute inside it; no cache-preview is used. The per-cell600s limit exceeds measured complete review runtime. Windows Jupyter emits selector-loop/TCP warnings, not cell errors. Temporary executed notebook is retained privately, not task/output.

Five groups under private figures/nearfield_acoustic_equivalent_source: (1) actual processed pressure magnitude/phase/real/imag; (2) original versus fresh TGV source and actual error; (3) actively computed LN/TV/TGV and residuals; (4) measured versus reprojected complex pressure; (5) actual iteration evolution, regularization sensitivity and measured runtimes. Every group exported PNG/SVG/PDF, with exact numeric_sources.npz, full-precision metrics.json and SHA256 manifest. All five groups were visually inspected for units, labels, actual pixel-center extents, shared comparison scales and legible panels. The zero complex discrepancy is a measured exact equality, not a missing result filled with zeros. Sensitivity observations and regularization artifacts remain visible; no invented source truth.

Approach and design were semantically checked against actual signatures, operation order, dtype, normalization, signs, zero boundaries, fixed stopping conditions, CLI and notebook paths. They explain the selected source-specific baseline and unit limitations. Public materials were scanned for hidden assets, GitHub URLs, privileged source arrays and target-derived calibration; none are present. Final template includes no standalone metadata, PDFs, checkouts, private capture scripts, duplicate tests, cross-task imports or domain libraries.

## Gates

| Mandatory gate | Status | Evidence |
|---|---|---|
| Primary paper, exact remote code revision, data hashes and equation/function/test mapping | passed | primary indexed article, commit API, mapping and hash table |
| Original and local exact-bound-data runs | passed | actual exit0 original log, main and active notebook |
| Independent upstream parity and physics | passed |13 parity/physics cases within37 tests |
| Constants, units, precision, geometry and boundary conventions | passed | explicit metadata, original G, independent Helmholtz/adjoint checks |
| Fresh output and standalone Judge | passed | final checks and JSON report |
| Judge poor/malformed/evaluator failures and truth priority | passed | unit contract tests |
| Public allowlist sufficient, no privileged calibration/truth/code link | passed | README/raw/dependency scan |
| Minimal independent local numerical runtime, forbidden-symbol audit | passed | local source and AST/manual import/call review |
| Full tests | passed |37 tests; no skipped required gate |
| Active canonical executed notebook and real figure comparisons | passed |7 executed cells,5 embedded groups, archived15 exports+numeric sources |
| Historical/original/fresh results and justified thresholds | passed | separate original captures, fresh metrics and provenance |
| README/plans/source semantic agreement | passed | explicit mathematical and interface review above |
| Final inventory, licenses and archive | passed | exact MIT notice, private archive index, task manifest |

No mandatory failed/unverified gate is concealed. Absolute source truth, individual speaker signals and learned-network comparison are **not available and not claimed**, not substituted with reconstruction. Parent independent acceptance is pending.

## Private archive SHA256 index

All paths below are relative to the maintainer-only group archive; additional figure hashes are in figures/nearfield_acoustic_equivalent_source/manifest.json and nah_archive_index.json. This table does not expose any of these materials to solvers.

| Private source/capture/log | SHA256 |
|---|---|
| nah_source.zip | ddf782e47405e0c2a98e49f4ac39633818f49735c65a78336490f3dc82d72ff9 |
| nah_commit.json | fe0a7104eaa1971e6cf5265302d179d344017ad52dcc9b783c8b672b96ef347d |
| nah_source/nah-variational-network-56dfb274db3a403e3e704ac911d1ee935f567c61/data/src_data.pt | 95204e774dac37620408b403b6b605ca66fe8bbfa03088b28aefb70241805bd3 |
| nah_source/nah-variational-network-56dfb274db3a403e3e704ac911d1ee935f567c61/LICENSE | cc80b941e99bbaa94cd5bb4cdd8beeebc6fe360ae5f9704744be2aab5246ad86 |
| nah_original_screen.py | 59f8768b5107b604a5eabbf834cd254cad0805bf421c35c64f60e7d4b1fdb1ef |
| nah_original_screen.npz | b36f72c6491541818b115690cb4bc04b499f46c6622f4dee29884a3699337874 |
| nah_original_comparisons.py | 3fdd058de513082b7eeb3a8b22e428126ce2cb84bb1ba0a64d3789250d6ee3bf |
| nah_original_comparisons.npz | cba2709a7b202719a1f544e0401ddf5eff7d2b35744645699397cd8fe3394018 |
| nah_original_comparisons.json | e6218f2f22d93d83b27e8eccbd0e2db32541cbc8d98180ea3bd3378ecdf35323 |
| nah_freeze_stage.py | af9cbe4abe6bb34ed37153c114ea75affaa15a5521fb948deef6b11b6580453a |
| nah_original_stages.npz | 14270504b2977622960dfe85a72fbe301f889038462ba90f33dbd6248438b94e |
| nah_oracle_verified_exit.log | cd1955167ae9a1210c61d025eb595627eae773ddf6f9fcdcfb804b23f26a3a41 |
| nah_oracle_verified_exit.json | 3d3befce7541ce6f955f9bf27adbf0d89f5c009bc36284ec422cb43c41c6fb3f |
| nah_oracle_environment.txt | 494e61da15c995893ab66cf69742f23051e6040c7bc18fab75ac59a89b8c0ff1 |
| nah_official_indexed_primary.json | 07b599ac51aaf2abe8518e33a2cae633cd832f2591a9b69eedad8d863cd8ad8e |
| nah_notebook.log | 4fcb40d82ee9a514d57fb1ce92babd173ca753f1df9c3da963d89bc5d6a46c6e |
| nah_notebook_run.json | dfeec25af216746ea9978d07d6350ea2b3e016057ca77830e062898052715395 |
| nah_final_checks.json | efc1001473e0f1b74d70df8a799d47f30786b2a30e381f22a08eab028e587305 |
| nah_final_main.log | b147a82408b48d2b1e0aaf806bf7bfc191a6b1b56d7fb782e85985b01071c394 |
| nah_final_judge.log | e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 |
| nah_final_tests.log | 4117d44797b4b6d2e945ce05557bb8659cfab6f62d8cce39288b38fbb4b00ccb |

Task raw_data.npz SHA256: 058b4d1b30980bb6901dc5831a109480a5cace8426fe2fde4d51302ec6ed7c99. Original data/src_data.pt SHA256: 95204e774dac37620408b403b6b605ca66fe8bbfa03088b28aefb70241805bd3. Canonical executed notebook SHA256: f88d4ce9bee168d05ed560c3ab1f061698f8907325bafe1379f0ec76fe7339c8.


## Parent acceptance review

Parent independently ran the complete 37-test suite (44.00 seconds; exit 0), the standalone Judge from a foreign working directory (exit 0), template/public-data/import-call audits, and reviewed the source, paper/code discrepancy record, README and both plans. All five executed notebook figure groups were opened and inspected individually. The mathematical review checked the exp(+i omega t) outgoing Green function, i omega rho factor, absence of a pixel-area multiplier for point sources, complex adjoint, zero-edge first/second differences, squared-loss factor two, complex dual projection, source-specific I100 spectral estimate and the full 20000-step Condat-Vu updates. Frozen original LN/TV/second-order outputs and the first 101 primal/dual updates are independent source captures; full pipelines and independent Helmholtz, adjoint and directional-gradient checks pass. No learned-network result is claimed. The original second-order derivative variant is explicitly distinguished from auxiliary-field TGV and its mu=0 configuration from the paper's general fused discussion. Existing real measurements are amplitude/phase processed author units U, not absolute Pa; the output is a nonunique complex equivalent-source distribution, not measured mechanical surface velocity. No hidden truth exists or is invented. All three baselines, regularization sensitivity, convergence and pressure consistency are actually computed, and the shared kernel's executable and printed NumPy/torch versions are recorded. Accepted as a new physics inverse task, distinct from incoherent acoustic power/covariance recovery.
