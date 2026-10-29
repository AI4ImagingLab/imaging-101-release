# Spectral single-pixel validation and provenance

## Bound task, papers and exact code

The task reconstructs the complete real tomato_slice_2_zoomx1 acquisition:8192 actual complementary detector spectra,2048 stored wavelengths, native64×64 spatial cells. No truth cube exists and no synthetic observation is generated. The selected method is the published native direct inverse, not the paper's trained128×128 DC-Net. This selection and the changed experimental scope are explicit in README and notebook; all20retrospective local inversions and4scientific figure groups are actively computed.

Paper: G. Beneti Martins, L. Mahieu-Williame, T. Baudier, N. Ducros, OpenSpyrit: an ecosystem for open single-pixel hyperspectral imaging, Optics Express31(10),15599–15614(2023), DOI10.1364/OE.483937, received2023-02-07/revised2023-03-08/accepted2023-03-09/published2023-04-25. The exact published full-text supplied by Optica was inspected through https://www.researchgate.net/publication/370289633_OpenSpyrit_an_ecosystem_for_open_single-pixel_hyperspectral_imaging ; selected-method equations1–4, setup/measurement sections2–3, Table1 and comparison scope were checked directly. Web access snapshot is paper_web_access.json. Publisher direct requests produced empty/inaccessible responses; HAL v2 file encountered Anubis; no nonexistent local paper PDF download is claimed. The publication is located at https://doi.org/10.1364/OE.483937 . Paper wording in the source README uses an older title; the actual published title above is used.

| Repository / role | Exact verified revision | Verification |
| --- | --- | --- |
| https://github.com/openspyrit/spas / original reconstruction | d38a8a896ce0b272d2e51797e334d93516fd18fc | Remotely queried newest commit before2023-05-01 via API, exact codeload snapshot; not claimed to be a tagged original-paper release |
| https://github.com/openspyrit/spyrit / original Walsh matrix | f211c2fa6fd63f4ffafd9cf1407a907219f836be | Same dated remote query and exact SHA snapshot |
| https://github.com/openspyrit/spihim / data description | eea727c8ae02116e0527ea5ade77ac899634f5d2 | Actually verified remote HEAD on2026-10-20, exact snapshot |

Latest SPAS HEAD69c6bafba8a2e2cb35f82c1aa43e791ee3dd6fff was also inspected but not used: its reconstruction import refers to a renamed metadata module. Rather than silently modify that source, the valid earlier source revision above was selected. All archive/query URL/response hashes are in *_2023_provenance.json, *_commit.json and archive_index.json. No local tracking revision is presented as remote freshness.

## Real data and legitimate calibration

The SPIHIM public author collection explicitly declares CC BY4.0 in its independently archived collection record pilot_collection.txt. Acquisition record: https://pilot-warehouse.creatis.insa-lyon.fr/#collection/6140ba6929e3fc10d47dbe3e/folder/61b07ee23f7ce79f5e565c2c . Raw actual readings: https://pilot-warehouse.creatis.insa-lyon.fr/api/v1/file/61b07ee43f7ce79f5e565c45/download . Source metadata: https://pilot-warehouse.creatis.insa-lyon.fr/api/v1/file/61b07ee43f7ce79f5e565c42/download . Object/date/instrument setup are explicitly stored by the author; this is not digitally sampled image simulation. Raw float64 values and full spatial/spectral shape are preserved; JSON pattern identifiers and wavelength literals are parsed into numeric public arrays without recomputation.

| Asset | SHA256 |
| --- | --- |
| Original raw spectral NPZ | 53181772360092e31977c6c293d6372b7195d2ef1b7e92873427fc2719fd81e9 |
| Original metadata JSON | 9f9b327e6621aba9c2558de6a8c50f79e04bcf0a308b3d36dc95d2fc22ee1e4d |
| Public batch-first raw_data.npz | 35725e1b4ef72c519685d856c69cb6cf4e3188b00f518655c6480543a72c1598 |
| Frozen original stage fixture | 8998231a0e39c2addc587a8ef31b1468bdd9bfe7aa064af508ce449c168eb5e5 |
| Frozen original sampling fixture | 4d6abd5c341bf68aa9f1c025068b03ff8fd71dd37f80902402ea4ee3b473a7a5 |
| Hidden standard-method full cube | c3dc9ac0e604bdbd80bf01f8ea10ecb259fb7f8208b6f20e15d222f8fa52a8c6 |

The acquisition order originally comes from an independent generic image database, but only the actually displayed pattern identifiers are needed and exposed. No target-derived optical calibration, hidden seed, simulator object, trained weight or target image is input. Nominal182.4micrometre pitch is the paper's supplied x1 value, not13.7/0.9 recomputed and rounded differently. The actual original spectral calibration is514.7382–750.2672nm; its2048 point values are authoritative. The source-recorded1.400009742966066ms mean timing and11.4674765625s total timing are preserved, rather than replaced by8192×1.4ms. No metadata JSON is in the final package. The original metadata contains historical private Windows paths and stays only in the external archive.

Independent noise gain/variance and absolute radiometric transmission are unavailable. Stable complementary offsets cancel; within-pair drift and measurement noise can remain. Negative intensities are preserved, including overall negative-pixel fraction 0.22338199615478516; low-signal wavelengths and exterior field explain why display-band structure and native signed spectra must be distinguished. This benchmark reproduces a measured linear inverse rather than proving material concentrations or noise-free specimen truth.

## Original run, sequence and environments

Original full-data execution was completed **before local implementation**, using unmodified original reconstruction_hadamard and walsh2_matrix loaded directly from the exact private source files by importlib. The initial run used the prepared maintainer numerical interpreter in an independent process, not cleaned functions; it produced all numeric oracle arrays, then repeated the inverse independently. Initial original inverse runtime0.8975320998579264s, total18.468314500059932s, exit0, repeat max difference0. Records before_local_upstream_run.json/.log/.npz preserve this pre-extraction evidence. The deliberately red test collection then failed because local src did not exist; red_tests.log is not a successful verification claim.

Final verification additionally ran the same untouched original modules and exact observations in a **separate private oracle-env interpreter**. It reuses a read-only path to the already validated general NumPy/SciPy/PyTorch/SymPy primitives, with no domain package installed into the cleaned runtime. Original domain modules are only loaded from private exact-version source files. Final isolated original inverse0.9082373999990523s, total13.762842899886891s, exit0; every stage/full-array value is bitwise identical to the independently frozen pre-local original arrays. Original repeat max difference0.0. Isolated command/environment/equality record: isolated_oracle_evidence.json, isolated_oracle.log, upstream_run.json. This final separate-interpreter repeat is not backdated as the initial run.

Initial15retrospective sampling cases (3orders×5counts) were independently executed and frozen before local extraction. Visual review identified the missing-DC brightness failure in random subsets;5additional original-source cases that retain the actually measured DC pair were computed independently before adding the local random_dc experimental branch. The original initial15cases remained bitwise equal. upstream_sampling_initial3.npz, upstream_sampling_run.json, upstream_sampling_dc_run.json and logs preserve the actual sequence. No threshold was loosened, no simulated noisy observation was introduced and no missing method was filled with zeros.

All runs are CPU single-threaded (OMP/OPENBLAS/MKL1), no GPU. Windows/Python/backend exact versions are in cleaned_environment.json and original run JSON. Cleaned runtime uses only NumPy2.4.6 and Matplotlib3.11.1; Jupyter/pytest/IPython are shared development tools. SPAS/SPYRIT are not installed/importable on the cleaned runtime; their original private file modules never enter solver/Judge/tests/notebook imports. No shared package was upgraded. Original Torch/SymPy imports belong only to the original generic transform module, not to cleaned mathematics.

## Equation / source / local / independent test mapping

| Stage | Published / original implementation | Local implementation | Independent validation |
| --- | --- | --- | --- |
| Raw binary physical masks and counts | Paper Eq1–2; metadata actual pattern IDs | README model; preprocessing.load_observations | Complete identifier and wavelength exact checks; no target calibration |
| Complement subtraction and row placement | Paper Eq3; SPAS reconstruction_hadamard M_breve/M_Had | differential_coefficients |41whole-spectrum channel stage fixture exact |
| Sequency Walsh signs/order | SPYRIT walsh_matrix, sequency_perm_matrix | sequency_order/walsh_matrix and explicit butterflies | Original64×64 W exact; independent orthogonality/DC/impulse |
| Full least-squares inverse normalization | Paper Eq4; SPAS reconstruction_hadamard; SPYRIT walsh2_matrix | physics_model.inverse; solvers.reconstruct | Full64×64×2048 original result bitwise equality; independent adjoint |
| Forward sensing | Paper Eq1–3; H formed by original Walsh basis | physics_model.forward | Complete measured-difference exact reconstruction; known impulse sign/order |
| Retrospective actual-pair selection | Paper discussion of a posteriori acceleration; exact original inverse called on subsets | reconstruct pair_count/order, compute_experiments | All20cases in selected independent original band fixtures; actual binned input exact |
| Metrics | Explicit independent mathematical definitions | Judge reference_metrics/observation_residual; visualization.metric_values | Known nontrivial numeric cases, all poor/error contracts, full independent dense sign-change-sorted operator |

Required mathematical review checked all factors, signs, units, C-order flattening, two-dimensional normalization, seq permutation, actual wavelength/timing literals, complement interleaving versus the paper's block notation, positive/negative pair polarity, and exact-array fidelity. The source's symmetric dense Q versus the local explicit transpose convention is verified by full original parity and the independent forward/adjoint/impulse tests; not assumed from an opaque equation number. Both plans were checked against actual signatures/defaults/units/caller flow. Core runtime has no generic reconstruction shortcut, domain import, hidden answer load, checkpoint or cross-task call.

## Fixtures, references and metric provenance

test/fixtures/upstream.npz contains independently captured41-channel full-stage differential/coefficient/image arrays, original64×64 W and exact calibration arrays; test_original_stages_exact consumes every stored key. test/fixtures/upstream_sampling.npz contains the actual8-band measurements and2selected spectral bands of all20original sampling images/counts/edges; test_retrospective_sampling_original_parity consumes them. Different functions do not generate their own expected outputs.

notebooks/reference_outputs/upstream_views.npz contains original8band sums, band edges and3identified native pixel spectra/coordinates for review. The notebook uses the original band sums in its original/fresh comparison; the pixel spectra/coordinate/edge arrays identify the original measurement view and are verified against matching fresh analysis in review. evaluation/baseline_reference.npz stores the exact original **method result**, not ground truth, for the evaluator/full pipeline test. The final fresh output is recomputed from raw data; its source is recorded in logs and the executed notebook, not substituted from this baseline.

Cosine NCC is uncentred; dynamic-range NRMSE uses RMS error/reference range. The independent Judge forward residual is L2(predicted Walsh differences minus measured differences)/L2(measured differences) over all4096×2048 measurements. No crop, scale fitting, phase/magnitude conversion or implicit alignment. Missing/invalid hidden assets are evaluator errors. Existing actual truth takes precedence over the baseline, proven by independent contract tests.

Before extraction, thresholds were frozen in frozen_acceptance.json: NCC>=0.999999999999, NRMSE<=1e-11, relative measurement residual<=1e-11. Original deterministic repeat is bitwise identical; exact integer arithmetic below2^53 plus binary scaling explains the equality and a nonzero binary64 implementation tolerance is deliberately provided for evaluation. Strict exact-stage/full parity and physical checks remain separate gates. Zero-norm prediction has NCC0and fails; constant/zero reference is an evaluator error. The Judge rejects finite-overflow results with valid error JSON/empty metrics, rather than emitting NaN/Infinity. No boundary was changed after judging the cleaned result.

Fresh measured report: {"ncc": 1.0, "nrmse": 0.0, "relative_measurement_residual": 0.0}. Fresh direct in-memory inverse0.5504606000613421s; CLI includes compression and I/O:4.392295100027695s. Four active code cells and4actual PNG figures are embedded in the canonical notebook. Native single-channel negative values and raw-pair temporal variation are shown rather than clipped from results. Sampling curves quantify method-reference information, not true physical accuracy. At100%all4orders recover exactly; the DC-controlled random group isolates the missing-mean limitation.

## Executed commands and figure review

| Command from task directory | Status | Evidence |
| --- | --- | --- |
| python -m src.main | passed, exit0 | cleaned_cli.log, declared real float64 output |
| python evaluation/judge.py --prediction output/reconstruction.npz --report output/judge_report.json | passed, exit0 | cleaned_judge.log and strict fresh JSON |
| python -m pytest test/ -v | passed,20tests | full_tests_final.log; final source/experiment version |
| python -m nbconvert --to notebook --execute notebooks/hyperspectral_single_pixel_hadamard.ipynb --output hyperspectral_single_pixel_hadamard_executed.ipynb --output-dir output --ExecutePreprocessor.timeout=240 --ExecutePreprocessor.kernel_name=research | passed, exit0, 9.100184499984607s | Private explicit matching research kernel, active_notebook.log; complete executed output embedded in canonical notebook |

240s timeout allows complete native I/O/reconstruction,20sampling cases and all figure rendering; actual elapsed time is above, not an assumed seconds-only requirement. Initial Agg-only notebook execution produced no embedded figures; this was caught during notebook inventory review. The notebook explicitly activates the inline Jupyter backend, then the entire notebook was rerun. No library or src.main import changes a backend. All4embedded figures are now present and were inspected along with SVG/PDF/PNG exports. Temporary executed notebook is archived privately, not left in output. Recorded platform zmq/debug warnings did not stop execution.

Actual scientific figure review:8spectral spatial bands share a physical count scale; source/fresh comparisons use identical limits and explicitly label zero residual; the4order×4count plate retains the missing-DC failure plus measured-DC control; the diagnostic figure includes nontrivial method-reference errors, raw complement sums, native2048-channel spectra and signed-noise fraction. Labels/units/scales/legends are legible without clipping. No panel claims a reconstruction is true specimen data or fills a method with zero. Paper-ready vectorPDF/SVG plus180dpi PNGs and their numerical sampling source are in private figures/ and sampling_results.npz. Their source/claim contract and render ledger remain outside the final task. Final manuscript/gallery assets preserve quantitative limitations, not cosmetic retouching.

## License, private archive and final gates

Task LICENSE preserves full GPL3-or-later. SPAS source is LGPL3; the selected original SPYRIT Walsh file explicitly declares GPLv3, which is preserved despite project-level LGPL labeling. NOTICE identifies authors and extraction roles. Real data CC BY4.0 attribution/transformation notice is retained. No original paper PDF/checkouts/capture script/log or extra exported image is placed in the task. Source zips, metadata, original numerical evidence, environment/capture/review logs and helper scripts are outside the task under D:/yan1/agent/AutoSkill/imaging-101-maintainer/research_batch2_20261020/single_pixel; archive_index.json records hashes, sizes and source filenames. Final task hashes are additionally captured by the parent batch manifest after acceptance.

| Mandatory gate | Status | Evidence |
| --- | --- | --- |
| Paper/version, exact remote source/data/license and equation map | passed | Source/data/version/traceability sections; hashes/index |
| Original and cleaned exact bundled observation runs | passed | Independent pre-local source run, separate isolated repeat, active CLI/notebook |
| Independent numeric parity and physical checks | passed | Exact stage/full/sampling arrays plus DC/impulse/adjoint/orthogonality/forward tests |
| Constants, unit/precision/coordinate conventions | passed | Public raw equality, source metadata literals, plan/semantic review |
| Standard entry and fresh standalone Judge | passed | CLI/Judge exit0; all2048channels scored |
| Poor/missing/key/shape/dtype/NaN/Inf/overflow/evaluator/report errors | passed | Independent contract cases, finite strict JSON, truth priority and foreign cwd |
| Public inverse sufficiency/no hidden information or source-URL leaks | passed | Full acquisition/output contract, numeric public allowlist, no target-derived data |
| Minimal shared dependencies/local explicit mathematics/no shortcut | passed | NumPy+Matplotlib subset; source/import/call-role audit |
| Complete test suite | passed |20passed, original independent expectations |
| Active notebook, comparisons and actual visual review | passed |4active code cells,4figures,240s-selected timeout, full rerun and export QA |
| Honest historical/original/fresh/reference metrics and nonvacuous bounds | passed | Distinct ledgers and baseline identity; unchanged predeclared boundaries |
| README and plans match actual math/signatures/paths | passed | Explicit source-semantic review and constants tests |
| Template inventory, cleanup, archive and legal notices | passed | Only required package materials plus LICENSE/NOTICE; temporary notebooks/figures/cache archived outside |

No mandatory gate remains failed or unverified. Method-reference reproduction does not establish an independent true tomato intensity or absolute radiometric accuracy.
