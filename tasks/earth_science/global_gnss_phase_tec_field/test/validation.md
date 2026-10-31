# Authoritative provenance and validation

All technical cleaning gates below are passed. Parent acceptance is a separate
review decision. This is a qualified reproduction of the selected source's
conditional single-shell inverse, not unchanged whole-paper reproduction and
not an independent physical-truth map.

## Fixed physical binding and semantic novelty

The unknown is vertical electron content over a full global field at25 hourly
nodes, from all504358 paired real carrier-phase-derived observations in the
author-provided2017-01-02 package. All313 available station directories were
selected before seeing any quality result. This is not a nowcasting problem.
All output samples(1,25,71,72) are evaluated, without a postfit mask, crop or
alignment. The observation inputs are author-processed TEC differences and
known ray calibration, not raw RINEX and not synthetic measurements.

Full existing-task semantic screening found no ionospheric TEC inverse. The
existing geomagnetic Gauss-field task uses magnetic intensity/direction and a
different physical unknown; geodetic GNSS fault slip uses ground velocities.
Neither duplicates dual-frequency dispersive radio propagation to electron
content. Rejected ROHSA/cBathy/ERT routes remain private and are not counted.

## Paper, source, data and rights

Paper: Padokhin et al., *Phase-Difference Approach for GNSS Global Ionospheric
Total Electron Content Mapping*, Radiophysics and Quantum Electronics65,
 481-495(2023), DOI10.1007/s11141-023-10230-6. The complete15page published English
 author postprint was read, including equations3-16, preprocessing, simulated
NeQuick comparisons, real case and conclusions. Primary full text: https://www.researchgate.net/profile/Artem-Padokhin/publication/370183741_Phase-Difference_Approach_for_GNSS_Global_Ionospheric_Total_Electron_Content_Mapping/links/64437a00d749e4340e2cb413/Phase-Difference-Approach-for-GNSS-Global-Ionospheric-Total-Electron-Content-Mapping.pdf
The original Russian article dates to2022. Direct binary retrieval returned403;
the actual full primary webPDF parsed responses are archived, not a fabricated
PDF file or hash:

- `paper_primary_parsed_0.json` SHA256 `0c79a330b7b321f0e9cabd604c3750c42f0d8671a210e901ed0611da54a17089`
- `paper_primary_parsed_180.json` SHA256 `8a044265e214817bfa7dffa66a8f2ce7ca5b2349adb38029a20a97eb4034a4df`
- `paper_primary_parsed_390.json` SHA256 `6eea826bf1dbad524d6d6c5694e9e0b83ca35b3c5efc2fddc2ac5387da6e9c2a`

Domain source: https://github.com/PadArt/mosgim2 , exact revision
`fc42e31f3559c2bb57e0cef636c378c3d9bb25cc`, verified remotely from the primary
GitHub HTML currentOid on2026-10-22 and retrieved using a commit-pinned archive.
The git remote command reset; freshness is not attributed to that failed call.
`exact_source.zip`2795617bytes SHA256
`7657cbb1f45989ee1166ae6c8b80d6b25509a90fd586ddc35057c159d0cba49e`.
MIT copyright2023 Artem Padokhin, Artem Vesnin is retained in NOTICE.

Author data URL: http://atm007-3.phys.msu.ru/gggimm/2017-002.zip . Actual download
682247360bytes SHA256
`1db56d28cbaa10bd6636c3f652be90f94670be2f8b494714646ab36dbe506df4`.
The package contains13826 ZIP entries and1794366388uncompressed bytes, with313
station directories. HTTP is the actual available author transport; no author
checksum or authenticated transport is claimed. Header records identify actual
RINEX observation/navigation inputs and30s nominal sampling. The source README
provides these data but no separate redistribution license was found. Local
user-authorized research use is recorded; no CC/data redistribution right is
invented. This unresolved redistribution limitation is preserved in NOTICE and
the private report catalog.

Lemke general numerical code is lemkelcp0.1, MIT, Andy Lamperski. Its two Python3
 `range` to `list(range)` position-container compatibility changes are archived in
`lemkelcp_compatibility.json`; they do not alter pivots. Original before/after
source and actual license text are retained privately. Clean runtime contains
explicit local tableau updates and no lemkelcp import.

Private archive root: `D:\yan1\agent\AutoSkill\imaging-101-maintainer\research_batch7_20261022\groups\astro_earth_mech\earth_reserve_screen`. `archive_index.json` records actual paths/hashes,
original source manifest, logs, commands, numeric captures and failed attempts.
Complete source/data/PDF-response assets stay outside the task.

## Discrepancies and explicit qualification

The unmodified author linear branch reverses earlier/later time weights for
both current and reference rays. Its full native run is retained. An independent
degree-zero ramp case has maximum row error .5641895835477563, while its
piecewise-constant branch agrees exactly. The qualified source exchanges only
the four endpoint weights, restoring the article's continuous interpolation;
the same independent case then differs by2.7755575615628914e-17. Constants,
linear ramps, all24node transitions and weight sums are verified independently.
See source/qualified time-interpolation cases, exact token diff, both native
logs and frozen stage captures. The domain file SHA changes from
`b96d9354227d66ef1a69ffbf4e8ad7881d1b28304d7ad79c0ca315e7a882c8f1`
to`7b715bcbe8bf001b47966e94acae978cd04c04430b69819562e63becf68d3e71`.
The Windows text write also changes367LF line endings to CRLF; this nonsemantic
byte difference is explicitly recorded in qualification_line_endings.json.
No other domain expression changes.

The bound configuration is the article's single450km/MODIP/degree15/24hour
linear recipe with sigma0=.02TECu and sigmaRel=.005TECu, rather than the repo's
two-layer default. The provided313station multi-GNSS day differs from the
article's286station GPS illustration. The executable preprocessing decimates
without implementing its paper-described smoothing TODO. Raw input exceptions
(e.g.no ray above elevation cutoff) are retained in original logs; all bundled
current/reference values are finite and above10degrees. No reconstruction-based
observation screening occurs. The source's minimum-reference sample is removed
from its own difference rows; phase ambiguities cancel.

The source LCP selects only initially negative nodes. Its selected2620node KKT
 relative stationarity is3.965503675187664e-14, minimum primal approximately -7.99e-15, minimum
dual0, maximum complementarity3.717289298055388e-10. Across all127800grid samples,
 one previously nonnegative node becomes -0.011068219960970849TECu. This small signed
value is preserved and disclosed, not silently clipped or masked. Strict global
positivity is not claimed. The assumed weighting scale also underestimates real
aggregate residuals; the diagonal covariance ignores shared-reference effects.

## Original-first execution and environments

All numerical jobs use CPU1thread and preserve at least6GiB physical and commit
headroom. The original environment is a private Python3.11.11 venv with NumPy
1.23.5, SciPy1.10.1, h5py3.8.0 and lemkelcp0.1; original NumPy/BLAS details are
captured in oracle_environment_stdout.log. Legacy versions preserve removed
np.int/sph_harm and byte-string converter APIs. No shared installation/upgrade
occurred. The cleaned shared environment is Python3.11.11, NumPy2.4.6,
SciPy1.17.1, Matplotlib3.11.1; exact numerical backend/build metadata is in
clean_environment_stdout.log. Notebook actual environment is saved in
figures/plot_metrics.json and printed in its first active cell:
`{'executable': 'D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research-runtime\\Scripts\\python.exe', 'numpy': '2.4.6', 'scipy': '1.17.1', 'matplotlib': '3.11.1', 'backend': 'module://matplotlib_inline.backend_inline', 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}`.

Every listed wrapper calls subprocess.run(...,shell=False), saves separate
stdout/stderr and captures the actual native child returncode, not just an
artifact. Runtime seconds are actual measurements:

| Private process record | Native exit | Runtime seconds |
|---|---:|---:|
| `unmodified_original_process.json` | 0 | 1243.323821800 |
| `qualified_original_process.json` | 0 | 888.901329800 |
| `qualified_repeat_process.json` | 0 | 893.279628800 |
| `local_main_process.json` | 0 | 631.300991600 |
| `judge_cli_process.json` | 0 | 45.576994200 |
| `notebook_process.json` | 0 | 889.798960900 |
| `pytest_full_process.json` | 0 | 650.709778300 |
| `final_guard_metrics_process.json` | 0 | 1.913021800 |

Unmodified original command: private oracle Python -X utf8 -u
bind_unmodified_original.py, cwd exact commit source. It actively runs the
unchanged loader/arc processing/coordinate conversion and complete inversion.
Preprocessing358.3073265s produces original_preprocessed_observations.npz;
source solve881.3341895s, followed by captured matrices. Qualified and repeat
commands run bind_qualified_original.py and bind_qualified_repeat.py using
exactly that same complete observation file, four corrected source factors,
all6400coefficients and all LCP stages. No random seed or target initializer is
needed: these are deterministic linear/LCP solves. Both independent qualified
outputs and dispersion are bitwise equal.

The first local use of the modern spherical-harmonic primitive passed source
stages but was interrupted after177.0409851s for performance, actual return
4294967295. Those logs are preserved as slow_spharm_y_interrupted_*. No success
is claimed. The shipped explicit Legendre recurrence passes frozen source
stages and the complete source comparison. Its successful CLI command is
`python -m src.main`; no upstream package is installed/imported by clean code.

The active notebook uses research PY -m nbconvert --to notebook --execute with
private pinned gnss_research kernel, timeout3600s, JUPYTER_PATH recorded in
notebook_process.json. The complete selected inverse in the notebook actually
takes638.488862100s; extra inverse/forward/plotting
computations account for its total runtime. Windows Proactor/ZMQ warnings were
recorded, and the native exit is0. Actual backend is
`module://matplotlib_inline.backend_inline`; no library module selects a backend.

After native execution, a pure Markdown amendment defines the imaginary unit
and adds the method citation. All code cells, outputs and execution counts have
identical SHA256 before/after, as recorded in notebook_markdown_amendment.json.
The executed pre-amendment notebook remains privately archived. Canonical
notebook SHA256 is`c65b06080c649b01d3d758194755f21461b41f07253ad46e4e73744bbe210764`. This is a documentation amendment, not a cached
preview or a claim that a changed computational path executed.

## Equation/function/fixture traceability

| Required mathematics/stage | Original source location | Local implementation | Independent evidence |
|---|---|---|---|
| Paper Eq3 real spherical harmonics | mosgim/common.py calc_coefs/vcoefs | physics_model.real_harmonics | qualified_stages ac/ar; analytic degree0/1; independent sphere quadrature |
| Shell mapping factor | coords/geom.py MF | physics_model.slant_factor | exact original SF; zenith and low-elevation physics cases |
| Reference phase cancellation, Eq9 | data/tec_prepare.py process_intervals; mosgim.py construct_normal_system | supplied measurements; physics_model.design_matrix | original preprocessing hashes; frozen source A; identical-ray cancellation and ramp |
| Diagonal differenced error weights | mosgim.py construct_normal_system | physics_model.design_matrix | captured diagP/P and known elevations |
| Weighted normal equations | construct_normal_system; stack_constrain_solve_ns_np | solvers.assemble_normal_system | frozen b/normal probe; all-ray independent objective closure; full-field pipeline parity |
| Eq14 temporal random-walk precision | stack_constrain_solve_ns_np | physics_model.temporal_precision | exact first-difference energy, symmetry/PSD/open endpoints |
| Full direct inverse and source diagnostic | stack_constrain_solve_ns_np | solvers.solve_system | independent complete originals and fresh full tests |
| Eqs15-16 finite-grid constraints | constructG; LCPCorrection | grid_basis; negative_node_correction | original127800node grid, selected KKT and full signed maps |
| Complementary pivots | lemkelcp0.1 lemketableau | solvers.LemkeTableau/lemke | frozen initialization/two pivot tableaux and independent active-set3x3solution |
| Source preprocessing | loader.py/tec_prepare.py/coords.py | preprocessing.load_data preserves already calibrated bound observations | original native358s run; full array byte hashes; no new synthetic data |
| Evaluation and display metrics | independently specified benchmark contract | independent Judge; visualization.map_metrics | shared analytical NCC=NRMSE=1/sqrt(2) known case; reference priority and malformed cases |

Fixtures are captured from source execution, never generated by the cleaned
operator. qualified_stages.npz contains24fixed-index diagnostic rows from the
full acquisition and their A/P/SF/b/normal values; it is not a new binding crop.
The initial object-dtype capture was rejected and retained privately; shipped
fixtures contain only numeric arrays loadable with allow_pickle=False.
lemke_stages.npz contains an independent3variable source tableau case.

## Numerical quality and frozen boundaries

Public raw_data.npz SHA256:`6edc457057382c4f0b70439bbec118e747087ca45a51ddb59d7c5cccb30107d5`. All nine observation arrays are copied
bitwise from original_preprocessed_observations.npz, adding only a leading batch
axis. The unused second-shell coordinates are omitted; no ray is omitted.
Independent upstream map SHA256 in baseline_reference.npz:`9da93b06b601d13925336a4cc578db116d882f4ca5418327384f71c84d89097a`.
The source full map reference is not actual truth; ground_truth.npz is absent.

Two qualified complete independent runs differ by0. A separate measured
float32 serialization/float64 restoration of original coefficients gives
maximum per-hour own-range NRMSE1.9763361629357877e-8 and max map discrepancy
1.232023263852966e-6TECu. Before any local solve, freeze_binding.py fixed:
each-hour NCC>=.999999999999, each-hour range NRMSE<=2.9645042444036815e-8
(1.5times measured serialization error), and full-ray weighted RMS<=
1.5272988162761219TECu(1.02times independent original1.4973517806628645).
No threshold was relaxed after a cleaned result.

Fresh local worst-hour NRMSE=7.2785293759988451e-15, min NCC=
0.99999999999999989, maximum absolute discrepancy=
3.1690206014900468e-12TECu. Independently coded full-ray SH evaluation gives
J=574679716.2425296 versus source normal quadratic574679716.2425327. Normal
stationarity and selected-node KKT checks independently verify the inverse.

All-ray weighted residual RMS(TECu): qualified1.4973517806628645,
unconstrained1.4960235399770478, original wrong-time coefficients evaluated
through correct physics1.8132918114752854, best uniform5.263030157972085,
zero5.922949979638507, time-reversed4.207457266878364, sign-wrong11.532058947701088.
The maximum residual206.29928360019898TECu is retained. Stronger/weaker fresh
temporal scales .0025/.01 yield weighted residuals1.5480398926484227/
1.4786077611427093, with corresponding full-field temporal variations visible.
These are explicitly different conditional estimators, not target changes.

## Tests, Judge and active scientific notebook

`python -m pytest test/ -v -p no:cacheprovider --basetemp <private>` ran all40
then-existing tests, including an active fresh complete-day inverse and
independent every-ray Judge, native0/650.709778300s.
Two pure-contract additions (finite1e308overflow and a shared independent
metric known case) were then included in all41 current nonpipeline tests,
native0/1.913021800s. There
are42 distinct current tests, all covered; the full costly pipeline is not
misrepresented as rerun after those nonnumerical additions. Logs/process
records identify both scopes. Parent may execute the final42test collection.

Standalone Judge actual CLI exit0/45.576994200s.
It independently scores all25maps and504358rays, without solver imports. Its
contract tests cover missing file/key, exact shape/dtype, complex/integer/object,
NaN/Inf, finite overflow, malformed thresholds, evaluator failures, actual-truth
priority, invalid-truth no fallback, degenerate references, poor fields,
foreign cwd and report-write failure. Errors return2 with empty measured
metrics; valid below-threshold predictions return1; finite JSON is required.
Zero/uniform/shuffled/sign/time-reversed outputs fail. The raw calibration
for the forward check is public and independent of the target.

The canonical notebook actively reconstructs the full day, then freshly
computes constrained/unconstrained comparison, all-ray residuals and two
temporal-prior inverses. The paper's unprovided NeQuick synthetic-truth case
is not fabricated. CODE product retrieval attempts failed (recorded timeouts/
404), so no unavailable external product is plotted as if computed. Instead,
task-relevant positivity, temporal coupling, source qualification and physical
observation comparisons are computed on this actual complete binding.
Every original measurement and fresh result is labeled separately.

All six actual PNG groups were visually opened: data coverage/zero-bin legend;
six same-scale complete-field slices; original/fresh signed symmetric residual;
unconstrained/LCP with explicit saturation and full-grid negative counts;
all-ray consistency plus a clearly labeled histogram display zoom; and two
fresh prior variations with common symmetric limits. No zoom changes numeric
evaluation. Paired SVG/PDF preserve vector axes/text and raster image cells.
Private numeric source is figures/plot_sources.npz, metric source is
figures/plot_metrics.json, and plot_provenance.json binds all hashes, notebook,
figure consumers and actual execution. Canonical outputs remain embedded.

## Interface, mathematical and packaging review

README/approach/design were checked against actual code signatures, units,
indices, degree-major order, interpolation signs, objective normalization,
temporal endpoints, source ratio tie rule, LCP scope, output axes and Judge
alignment. The public README includes all12 visible keys and acquisition
metadata, no hidden references, no target-derived calibration or GitHub URL.
The generator is correctly omitted: no new synthetic observations are used
by the benchmark or notebook. Test-only mathematical known cases are not
represented as measurements. No root wrappers, PDFs, upstream trees or
parallel test/report directories exist in the task. Every bundled fixture and
reference NPZ has the explicit consumer described above and in design.md.

upstream_outputs.npz is consumed only by notebook comparison and contains the
independent qualified field, source coefficients and source prediction;
cleaned_outputs.npz records this notebook's fresh field/coefficients/prediction.
Neither is read by src. Baseline_reference is consumed by the independent
Judge and pipeline parity. Requirements contain only the actually validated
clean subset. All temporary executed notebooks, papers and capture scripts
remain private; task caches are reversibly archived, not deleted. Final file
hash inventory and source/call-role audit are in completion.json and
final_evidence.json under the private archive.

## Completion gates

| Mandatory gate | Status | Evidence |
|---|---|---|
| Complete paper, remote revision, data hashes, mapping and licenses | passed | full primary paper responses; exact source/data archives; NOTICE; traceability table |
| Actual original-first exact complete observations | passed | unmodified/qualified/repeat actual native0 process records |
| Independent frozen stage/physics/parity criteria | passed | frozen before local JSON; source rows/pivots; physics and full pipeline tests |
| Local explicit physical and inverse mathematics | passed | recurrence, differenced rays, prior, normal inverse and explicit Lemke; no domain imports |
| Preserved constants/units/coordinates/precision | passed | unchanged arrays; captured SF/A; known cases; source full maps |
| Standard CLI and independent Judge | passed | local_main_process/judge_cli_process native0 and finite report |
| Strict NPZ/evaluator/poor-output behavior | passed | all current contracts and real poor controls |
| Public allowlist sufficiency/no leakage | passed | all12keys, physical model, conditional scales, geometry, full output contract and URL scan |
| Shared minimal CPU runtime without oracle packages | passed | research Python execution; numerical backend captures; no shared upgrades |
| Unit/parity/physics/full pipeline testing | passed | full40native0 plus final41nonpipeline0 covers current42 distinct tests |
| Active full notebook and real comparisons | passed | nbconvert native0; actual kernel/env; six inspected PNG/SVG/PDF groups and numeric data |
| Historical/original/fresh metric provenance | passed | separate upstream/cleaned NPZ and original/fresh/null comparison labels |
| README/plans/signatures/math consistency | passed | reviewed source/interfaces/units and disclosed qualifications |
| Template inventory/notices/cleanup | passed | final source audit, inventory, NOTICE, private reversible cache archive |


### Public archive link audit, 2026-10-29
The privately archived 2017-002.zip central directory was inspected:13826 entries,13511 .dat files and315 directory entries, no .py/.m/.ipynb/C/C++/shell source files. This is a pure acquisition archive, not standard code. Its direct public URL was nevertheless removed as unnecessary; the paper and descriptive data attribution remain. Evidence: benchmark_readiness_20261029/global_gnss_public_zip_role.json. Existing archive URL/hash provenance above is preserved.
