# Validation and provenance: Hawaii gravity density volume

All mandatory gates below passed on 2026-10-21; ready for independent parent audit. This task reconstructs an effective spatial density field, not an independently known geological truth. Earlier 119 tasks were inventoried before selection; surface-field continuation is a different unknown/operator from this fixed underground 3D prism density inverse. No old task was changed.

## Primary sources, rights and exact observations

- Framework paper: Cockett et al. (2015), SimPEG: an open source framework for simulation and gradient based parameter estimation in geophysical applications, Computers & Geosciences85,142–154, DOI https://doi.org/10.1016/j.cageo.2015.09.015. Final open-access CC-BY paper downloaded and read from https://gif.eos.ubc.ca/sites/default/files/Cockett_2015.pdf; private Cockett2015.pdf. This is the selected framework, not a claim that the paper ran this Hawaii example.
- Original source https://github.com/simpeg/simpeg, remotely verified release v0.25.2 commit5f2d643c484ebe8b8ccfa029aa7377d133f2f53f; private verified API response simpeg_verified_revision.json and checkout SimPEG. MIT license preserved. Prism evaluation uses Geoana0.8.1 MIT source, both full licenses and copyright notices included in task LICENSE/NOTICE. Shallow editable packaging reports version0.1.dev1+g5f2d643c4; the verified commit/release, not that packaging-derived string, identifies the mathematical source. No original mathematical source was modified.
- Real measurements: Jim Kauahikaua (2017), USGS Gravity Data for Island of Hawai'i.csv, DOI https://doi.org/10.5066/F7V1230Q, ScienceBase item589e087fe4b099f50d3a0673. The actual fetched record is hawaii_usgs_record.json; CSV239893bytes, XML and text metadata privately preserved. USGS catalogue identifies public domain and original XML explicitly states Access_Constraints:none and Use_Constraints:none. Attribution retained. Latitude/longitude are WGS84, elevations orthometric metres above sea level, measured downward gravity mGal.
- Survey context only: Kauahikaua, Hildenbrand and Webring (2000), Deep magmatic structures of Hawaiian volcanoes, imaged by three-dimensional gravity models, Geology28,883–886, DOI https://doi.org/10.1130/0091-7613(2000)28%3C883:DMSOHV%3E2.0.CO;2. The bound fixed inverse is not a reproduction of its published geological model or a new validated geological interpretation.

All3611 source rows were read; exactly1019 unchanged rows satisfy strict19.25<latitude<19.55 and -155.45<longitude<-155.10. No synthetic observations were added, so generate_data.py is absent. Public observations contain only four original columns, each float64 shape(1,1019); no station has been gridded or interpolated. The source sample values and row selection are preserved exactly. All hashes are below and in the private archive index. The fixed mesh geometry and normal/height reduction constants are independent of unknown density, declared in README.

## Oracle-first sequence and environment

The independent original oracle was completed and hawaii_frozen.npz plus hawaii_fixture_contract.json frozen before local implementation. Private oracle_hawaii.py invokes original Simulation3DIntegral(engine='geoana', sensitivity_dtype=float64), LinearSimulation with the projected kernel, L2DataMisfit, WeightedLeastSquares, InexactGaussNewton and BaseInvProblem. All1019 bound observations and2048 fixed cells are used. No source solver is imported by the cleaned runtime, tests, Judge or notebook.

The private gravity-oracle-runtime is Python3.11.11 WindowsAMD64, NumPy2.4.6, SciPy1.17.1, SimPEG pinned commit above, discretize0.12.0, Geoana0.8.1, pymatsolver0.4.0. Exact backend/compiler details are in hawaii_original_environment.txt. Original command and exit are in hawaii_original_command.json; console and source log are separately archived. OMP_NUM_THREADS, OPENBLAS_NUM_THREADS and MKL_NUM_THREADS were1. The source reconstruction took1.3572564s (full subprocess2.298714s), exit0. No random sampling/seed or GPU was used. Original reruns agreed numerically; elapsed time is not treated as an invariant.

The original driver initially used unsupported num_processes and LinearSimulation mesh keywords, then attempted direct GaussNewton against a LinearOperator Hessian. These capture-driver configuration attempts did not establish success and did not change original source mathematics. The successful selected source method is InexactGaussNewton with identity LinearOperator approxHinv; source warnings about an unused is_symmetric solver keyword and CSC conversion are preserved in logs. They did not prevent successful evaluation or change the selected update.

Cleaned shared runtime: Python3.11.11, NumPy2.4.6, SciPy1.17.1, Matplotlib3.11.1. No upstream geophysical package is installed/imported for cleaned execution. requirements.txt contains only the used shared subset. The same single-thread environment and PYTHONDONTWRITEBYTECODE=1 were used. No shared environment upgrade occurred.

## Mathematics and traceability

The oracle driver fixes the selected reductions, mesh and quadratic inverse within the paper's generic simulation/misfit/regularization/optimization decomposition. These explicit task choices are not falsely attributed as special Hawaii constants in the framework paper.

| Mathematical stage | Original source/function or bound driver | Local executable function | Independent validation |
| --- | --- | --- | --- |
| WGS84 local coordinates, normal gravity, free-air and Bouguer height corrections | oracle_hawaii.py, declared physical reduction | preprocessing.prepare_observations | test_original_preprocessing_and_coordinates; equatorial/polar analytic gravity |
| Same affine-nuisance projection on data and every kernel column | oracle_hawaii.py, LinearSimulation projected G | preprocessing.project_trend | test_trend_projector_consistency, orthogonality/idempotence and nuisance invariance |
| Upward gz of rectangular prisms and SI-to-mGal/gcm3 conversion | potential_fields/gravity/simulation.py Simulation3DIntegral; Geoana prism_fz | physics_model.gravity_matrix, _primitive | all1019×2048 original coefficients; independent8³ Gauss-Legendre Newtonian volume integration; mirror symmetry and far-field mass |
| Cell-centered finite volume and ordering | discretize.TensorMesh source oracle | physics_model.mesh_geometry | exact centers, widths, x-fast ordering, z-up extent |
| Cell-volume smallness and interior first gradients | regularization/base.py WeightedLeastSquares and term.W@f_m_deriv | solvers.regularization_matrix | original CSR Gram matrix, constant-field smallness and zero derivatives |
| Unhalved weighted data misfit and beta regularizer | data_misfit.L2DataMisfit; BaseInvProblem evalFunction | solvers.objective_gradient | original misfit/regularizer values and independent scalar finite-difference derivatives |
| Newton-CG, explicit Armijo update and active gradient stop | optimization.InexactGaussNewton | solvers.newton_cg | original accepted full update, final gradient, complete density/prediction parity |
| Strict full-volume metrics and projected observation test | independently declared contract | evaluation/judge.py | analytic metric and malformed/poor/error/report-write/foreign-directory cases |

Units and signs: measured gravity is downward; d=-P*(g-normal+.3086h-2*pi*G*2670*h*1e5), where only the SI slab term is multiplied by1e5. Density is g/cm^3; K contains G*1e8. Positive subsurface density gives negative upward gz. Station height is orthometric z; source box is[-20,0]km. There is no terrain correction, hidden trend derived from density, positivity prior or depth weighting. The same unweighted affine projection acts on d and K.

Objective f=|| (A*rho-d)/5 ||²+.001||R*rho||². The5mGal is a chosen inverse model tolerance, not measured instrument noise or acquisition calibration. R has smallness50, x/y neighbor differences50 and z differences60; all nonzero original terms are retained. Original generic boundary/zero-second-derivative rows vanish; local omission changes no R-transpose-R. Original source and local start at zero and accept one full Newton step, then stop under absolute gradient .01. CG rtol1e-12,atol0,max2000; explicit Armijo coefficient1e-4,shrink.5,max10, maximum10Newton updates. Full explanation and actual signatures are in both plans. Moving-x and maxStep source stoppers are inactive on this selected one-step quadratic problem and are not misrepresented as exercised local logic.

## Frozen assets and metrics

test/fixtures/upstream.npz is the independent original capture consumed by comparing tests: normal gravity, corrected/projected observations, coordinates, full original acquisition/projected matrices, regularization CSR data, density, prediction, objective and accepted-iterate diagnostics. notebooks/reference_outputs/upstream.npz is a compact identified original result for scientific comparisons. notebooks/reference_outputs/fresh.npz is computed by the active cleaned notebook. evaluation/baseline_reference.npz stores the original standard density and target-independent original prism kernel needed by the standalone Judge. There is no ground_truth.npz because density truth is unavailable. Each duplicated subset has this specific consumer; no raw MAT/CSV/PDF/source checkout is bundled.

Predeclared tests: kernel rtol1e-9,atol2e-10; regularizer Gram atol1e-10; density rtol1e-6,atol1e-7; normal gravity rtol1e-14,atol1e-8; independent Newtonian quadrature rtol1e-8; final gradient norm<1e-5. These were selected from numerical scale before extraction, not relaxed to pass. Actual maximum original/local kernel difference1.55398e-12mGal/(g/cm^3), density difference1.7522e-12g/cm^3, final local gradient about6.96e-9.

Original result: density range[-.2722583445914349,.5859463179186413]g/cm^3, projected observation RMS1.7411629807405444mGal, unhalved weighted misfit123.56999389943287 and regularizer42276.79929679833. These are independently rerun source measurements, not paper-reported values. Fresh Judge results:

{
  "status": "pass",
  "passed": true,
  "metrics": {
    "density_ncc": 0.9999999999999999,
    "density_nrmse": 4.393173273782605e-13,
    "observation_rms_mgal": 1.7411629807404865
  },
  "thresholds": {
    "density_ncc": {
      "comparison": ">=",
      "boundary": 0.999999999
    },
    "density_nrmse": {
      "comparison": "<=",
      "boundary": 1e-07
    },
    "observation_rms_mgal": {
      "comparison": "<=",
      "boundary": 2.0
    }
  },
  "errors": []
}

NCC is cosine similarity without centering; NRMSE is RMS error divided by the full reference dynamic range. No crop, rescaling or alignment. Self-reference requires nonzero tolerance: NCC>=.999999999 and NRMSE<=1e-7, well above observed1e-12-scale numerical variability yet highly restrictive spatial agreement. Also projected observation RMS<=2mGal. Zero density is a valid poor prediction (NCC0) and fails. Independent Judge implements normal/height reductions and QR nuisance projection without solver imports; its original geometric kernel is legitimate hidden evaluator calibration, not a privileged solver input.

## Executed commands and review

Every command below actually ran, exit0. Full command arrays, cwd, durations and corresponding final_hawaii_NAME.log files are private final_hawaii_commands.json. The notebook timeout180s was justified for a roughly24s active path including three sensitivity reconstructions and figure exports.

| Run | Seconds | Exit |
| --- | --- | --- |
| cli | 1.561583600 | 0 |
| judge | 0.201577400 | 0 |
| judge_foreign | 0.200453700 | 0 |
| pytest | 3.822019700 | 0 |
| nbconvert | 23.752607100 | 0 |

The test command was python -m pytest test/ -v -p no:cacheprovider --basetemp <private>/pytest_hawaii_final:30 passed. The first exploratory invocation using Windows default shared temporary directory had16 setup PermissionErrors before those tests could execute. A private isolated --basetemp corrected that host temporary-directory issue; no numerical code or tolerance was changed. Final complete tests, task and foreign-cwd Judge, CLI and nbconvert all succeeded.

The canonical notebook retains actual executed cell outputs after successful nbconvert; the temporary output executed notebook was removed. It actively reconstructs the full raw dataset using local modules, runs all beta sensitivities and scores the fresh result. Four figure groups were exported and all actual PNGs visually inspected: (1) real observations, original reduction stages, forward fit/residual and histogram; (2) original/fresh/difference at three native density depths; (3) fresh beta=.0001,.001,.01 sensitivity at two depths, shared scales; (4) actual one-step convergence, independently captured source objective endpoints, final gradients, depth sensitivity, projected spectrum, L-curve and depth mean. Two-state convergence is honestly shown without invented intermediate iterates. Deep slices are weaker because gravity depth sensitivity decays; resolution is not implied by voxel count.

Private hawaii_figures contains PNG/SVG/PDF and exact per-group source NPZ, indexed by manifest.json with SHA256. No display uses uncomputed methods or substitutes a historical result for a fresh fit. Scientific limitations are explicit: regional-plane nullspace, missing detailed terrain/curvature corrections, fixed mesh/boundary prior, approximate Cartesian reduction and density-depth nonuniqueness. Reference agreement is not validation of real geology.

README and both plans were semantically reviewed against actual equations, signs, factors, units, source signatures, defaults and caller flow. All scalar/array acquisition constants match README;5mGal weighting and beta are maintained only as inverse choices. Control-character/bare-LaTex audit inspects original bytes without normalizing away bare CR. Source/call-role audit is private hawaii_callrole_audit.json; no domain shortcuts, cross-task imports, public source links, target-derived calibration or truth leakage were found. General SciPy CG is only a linear-system primitive, with local task Hessian/update/stop.

## Gate checklist

| Gate | Status | Evidence |
| --- | --- | --- |
| Paper, original exact revision, licensed data and hashes | passed | Primary sources and archive index below |
| Exact-data original run before local; independent frozen fixtures | passed | original command/log/environment, frozen contract |
| Local acquisition/inverse mathematics; physics and stage parity | passed | Explicit sources and30test run |
| Constants, units, order and precision | passed | Exact observations/mesh tests and independent quadrature |
| CLI, strict Judge and foreign cwd | passed | Actual commands and reports above |
| Poor/malformed/evaluator/report-write failures | passed | Contract test cases; codes0/1/2 |
| Public README/data/environment complete and free of hidden/source leaks | passed | Public allowlist and audit |
| Minimal shared runtime without domain packages | passed | Shared execution and call-role audit |
| All tests and full active notebook | passed |30passed, actual nbconvert exit0 |
| Fresh/source labels, metrics and thresholds consistent | passed | Separate references, own formulas and actual measured results |
| Plans/docs/source and actual scientific figures reviewed | passed | Signature/semantic audit and4actual groups |
| Final inventory, license notices and private provenance archive | passed | Task/private manifests; no temporary executed notebook |

## Private archive and exact hashes

Maintainer root: D:/yan1/agent/AutoSkill/imaging-101-maintainer/research_batch4_20261021/astro_earth_mechanics. Full source checkout is private SimPEG at the pinned commit; original dependency environment is gravity-oracle-runtime. Source, data, papers, logs, capture scripts and figure products are outside the task. hawaii_archive_index.json indexes original evidence; hawaii_task_manifest.json indexes every shipped file. Their hashes do not recursively include themselves.

| Archive path | Bytes | SHA256 |
| --- | --- | --- |
| Cockett2015.pdf | 2291287 | d39f382d397dbfea6bdc616b078cade2706b656544a3d08bff433f5ee0547e12 |
| SimPEG/LICENSE | 1109 | 41b2e03abdefd4ad9fcddbb6bccc8e42114d8fe318419bde7c86f24a88f80f3d |
| SimPEG/simpeg/potential_fields/gravity/simulation.py | 38014 | 3c1ceb43d224c72f095c68d12f6bd0abbc3c8729dd556258613641a3f15da90b |
| SimPEG/simpeg/regularization/base.py | 89599 | 13ad444835de2309dd692c6abef38e8c16b76281e3523a7b98acd85444c5e4c3 |
| SimPEG/simpeg/optimization.py | 52145 | 47330bdc15d1657dcb51212afc907e55899bcb75a0a3c5c326d4bedf886cdd15 |
| SimPEG/simpeg/inverse_problem.py | 11455 | 829ea708e89ff90b42f1788f792fccda400948d1931169cb17c96c62cdf4df3b |
| hawaii_gravity.csv | 239893 | 193d1213bee15b1c00517aafb811775a660140dfc6906d57c0e2e1f5b9edef5d |
| hawaii_gravity.xml | 10626 | 5ae016ea25fb719a3f219523d4ab02e414d127f026cd7dc76bc913097c39c6ed |
| hawaii_gravity.txt | 10308 | 5ec2b9dc5f06753ab1c8f79fc3ed00ccbc749dc45799eb5445bb5474a491cbcb |
| hawaii_usgs_record.json | 14738 | b8e86720531ea5bdfb5fad51611e09f7fcfaa95be109931b9bc96e7665a7957b |
| simpeg_verified_revision.json | 8866 | ba545ad86ad37135d06141d49be1d992a0babb84a382a35ac5c736b8d3786065 |
| oracle_hawaii.py | 4331 | 1ce9669d06e9cc6560ab1daf37e66af0bb20929d89d50b63748973d92a581f91 |
| hawaii_frozen.npz | 31770057 | a75fac280021256f9522cace09437128819885ff5de637fcbd0a8b069949b3c1 |
| hawaii_fixture_contract.json | 584 | ef507e7028a93bde33cbe86c048d97b3f37cdd004c3a367741dc44c85d9e9833 |
| hawaii_original.log | 870 | 9df60bc716d571b0b104ccc2c9b84f80c33ae8faf1e4b54cc396c253c292c98d |
| hawaii_original_console.log | 967 | f13532f70f471f7a079e2e2cb0fe1a6b2e055221b63d2302070466c7c8791898 |
| hawaii_original_command.json | 360 | dc49748d60a6a113df5ff85604ce8eda378c88f0934c65a759d7d783cde8fd5e |
| hawaii_original_environment.txt | 2708 | 28549eb8e93fabbb26a3abccc6e25acbafe9b6155c19086ab9a41e42fa21d403 |
| hawaii_original_summary.json | 321 | 11d396b330302899701ffa6b163d1f34f7e4b0d9406188ac4ec7a8acd4d8e90b |
| build_hawaii.py | 8648 | 8580033d4a5640bd764d147570e31a7f36c94b6d1c0383eeb7c0785379f65ab4 |
| docs_hawaii.py | 20441 | 7918f6ac4ca7b0a6defed61ec60c70b212f253b3b777926340dc4ae5d8ab8e77 |
| notebook_hawaii.py | 14764 | 2eb93796b4de23790122ef239b6f7f0c39acc0c27172498cfccf82e60e1bb604 |
| verify_hawaii.py | 1859 | 4e9670ef462de09cf3fcfa52196c9143c39dfd339dcb452c1d7c6bb5f30460b0 |
| final_hawaii_commands.json | 2814 | 19b492446040b6000f48f79ffc8fe9c03ca8ee55861f1cc5304f43913521bd29 |
| final_hawaii_cli.log | 56 | b2e7ae3d9df4d34f4ed7ab54ed40465751db53d496533fbe6c4b2e28cf1f454f |
| final_hawaii_judge.log | 6 | e0ba8b72f361b6f4f71dc78a0b7a570475bdd44e176caad604ba134208475403 |
| final_hawaii_judge_foreign.log | 6 | e0ba8b72f361b6f4f71dc78a0b7a570475bdd44e176caad604ba134208475403 |
| final_hawaii_pytest.log | 3173 | 733c4dcbc6a5339443fa013a99a016cc474ca95d75370bb61a3b305cac159a89 |
| final_hawaii_nbconvert.log | 866 | 8ee5c285d1d822ffe101e0f36bc14aced5982f5360a9366436a1fcc6489f4678 |
| hawaii_figures/manifest.json | 2556 | 013f91c9af2be31a2add1d2475978e56f01bb727e04f99bce943d71f59dab6ca |

## Shipped observation and reference hashes

- data\raw_data.npz: 0f030e5ddc20a3f8229ea87684ea19a9eca2075f74a82a46d4234d7732d735fa
- test\fixtures\upstream.npz: a75fac280021256f9522cace09437128819885ff5de637fcbd0a8b069949b3c1
- evaluation\baseline_reference.npz: bbfeb58a8f7aa1249fbcec8dc66d26cb3c38555019c6db8ec1f7031974182bfe
- notebooks\reference_outputs\fresh.npz: 3044bf450a6b12c027b01ce73a6353296a1fd5afbf7f91318fe23281dd99a760
- notebooks\reference_outputs\upstream.npz: d53764856f2a42125e486fa1d536638f155f4054bf9712a35548c68bd468bd6a


## Parent acceptance review

Parent independently ran30tests (30 passed in3.51s), foreign-cwd Judge0 and source/inventory audit errors[]. Read explicit eight-corner Newtonian prism integration, consistent regional-plane projection on both data and operator, volume-weighted regularizer and actual one-step Newton-CG objective; inspected all4 executed scientific groups. Genuine1019 USGS stations reconstruct a16x16x8 effective density-contrast volume. Independent quadrature verifies gravity sign/scale and finite differences verify local derivatives; unchanged source fixtures differ only1.8e-12g/cm3. Actual data residual1.741mGal and beta/depth/observability limitations are shown. README Bouguer conversion factor clarified and inversion-weight parameter removed from acquisition metadata. Standard volume is not geological truth.

## Final boundary-schema compatibility correction — passed

Authorized final schema-only correction on2026-10-21: evaluation/metrics.json now stores the sole canonical numeric boundary for each metric under its top-level <metric>_boundary key, with directions in metric_comparisons. Judge read_thresholds actually reads these named fields and derives its report thresholds; no unused compatibility alias or duplicate numeric boundary is retained. Boundary numbers, comparison directions, acquisition/inverse algorithms, observations, truth/reference arrays and output contracts are unchanged.

New test_boundary_schema.py proves active consumption by changing a canonical NCC boundary and observing a valid failure, and proves missing boundary, missing comparison and unexpected metrics return evaluator error. These complement the existing per-component, physical/parity and malformed-submission tests. The pre-change Hawaii schema test was run and failed as expected; the migrated schema then passed. This was a harness-field compatibility correction, not a new scientific method or relaxed acceptance.

Protected-file hashes prove all src numerical files, README, both plans, raw observations and hidden NPZ references are byte-identical to the pre-correction snapshot. All submission and notebook fresh numerical arrays except intentionally varying runtime_seconds are exactly array-equal to the previous accepted run; fresh measured metrics and Judge report threshold values are exactly unchanged. Previous README formula/Cartesian-coordinate corrections remain intact. Actual final reruns:

| Command family | Seconds | Exit |
| --- | --- | --- |
| cli | 1.976775000 | 0 |
| judge | 0.229466400 | 0 |
| judge_foreign | 0.226895400 | 0 |
| pytest | 4.537954200 | 0 |
| nbconvert | 25.585753800 | 0 |

Complete pytest result: 35 passed. Full active nbconvert completed with no error outputs; canonical notebook retains actual executed outputs and the temporary executed copy is absent. Exact commands/cwd/logs remain private final_hawaii_commands.json and final_hawaii_NAME.log. Private boundary_schema_verification.json records protected hashes and numerical equality; the pre-change snapshots are boundary_schema_before/hawaii_gravity_density_volume. All figure exports/numeric sources were regenerated by the active notebook and manifests refreshed. Original oracle assets were not rerun or changed for this schema migration. Parent re-audit of this final schema remains separate from the earlier scientific acceptance record above.

Latest schema/fresh artifact hashes (the earlier tables above identify their earlier accepted runs):

- evaluation\metrics.json: 642e865a3bc58016d4825dbd8be4438ebd4076365cc481ee53458cbc6a4fbdcb
- evaluation\judge.py: 448a0b1e0463c4a4c450d1182aabae48cb3eed62656eb40ba2395a419f47d3e2
- output\reconstruction.npz: 0c56a579ee43fd04c6cef82c63dbae7d670d1c0d45b714fe9df95feeed71ac5b
- notebooks\reference_outputs\fresh.npz: ca0da3291eadc48abe06d7abfc33246de0482d80271db9b5f63a0d4950088c5d
- notebooks\hawaii_gravity_density_volume.ipynb: 1d09f70cafe84906085c8d09f8d6ccd74f0910e019c2d92d4ef290450e6df5bd
