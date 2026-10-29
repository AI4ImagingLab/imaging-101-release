## Public URL role audit (2026-10-29)

Removed public link `https://doi.org/10.5281/zenodo.15285017` because its role is data CSV archive, but landing page supplies exact standard-code example URL; retain bibliographic text. Original public context: Sourbron S. Single-kidney magnetic resonance renography, version 0.0.3 (2025). [Measured dataset](https://doi.org/10.5281/zenodo.15285017), CC-BY-4.0. The selected implementation is the authors' modern two-compartment steady-state analysis; historical paper and modern implementation results are not assumed identical.

The citation and source/licensing provenance remain maintainer evidence; only the public code-access pointer was removed. Group evidence: `benchmark_readiness_20261029/group_1_url_role_review.json`.

# Provenance and validation: measured renal filtration / transit

## Binding and acquisition evidence

**Classification: author_real_processed.** Complete KRUK released cohort:100 participants,114 scans,228 kidneys,44–125 acquired frames each; no selection on isotope target or fitted parameters. Public arrays preserve float64 decimal CSV values and row order, padded with zeros only outside lengths. Measured aortic and whole-kidney ROI signals are author-processed instrument observations, not synthetic examples or raw images. Basak et al., *Magnetic Resonance Imaging*59 (2019)53–60, DOI https://doi.org/10.1016/j.mri.2019.03.005, author-accepted manuscript https://eprints.whiterose.ac.uk/145186/3/Manuscript_MRI_March_2019.pdf documents dynamic renal MRI at1T/3T and independent 51Cr-EDTA blood clearance with 99mTc-DMSA split-function measurements. It describes clinical studies, not generated signal. The released source ROI acquisition/calibration fields and author example identify this exact cohort. No instrument image or subject anatomy is fabricated here.

Data v0.0.3: https://zenodo.org/records/15285017 ; exact file https://zenodo.org/api/records/15285017/files/KRUK.dmr.zip/content . ZIP146921bytes, MD5 `954dd2e3cf3d3e83b97cba313a004228`; contains data.csv,pars.csv,rois.csv. License CC-BY4.0 confirmed by registrant metadata https://api.datacite.org/dois/10.5281/zenodo.15285017 archived renal_datacite.json. Paper historical127 studies differs from released114; all released observations retained, no claim to reproduce missing historical cases.

Selected author-maintained implementation https://github.com/dcmri/dcmri at remotely verified commit `f03234c68b65c3bd85eaa6bec0d97488d48a5aa3`, Apache2.0. Author example https://dcmri.org/examples/kidney/plot_gfr_validation.html explicitly fits these real data and reports poor clinical accuracy. Original historical paper repository https://github.com/plaresmedima/Basak_et_al_2018 at `af52b36978b403db4e71c3002e93f4664ecaf522` is archived for traceability but its referenced volume/isotope CSVs are absent. **We ran the modern author implementation, not the incomplete historical scripts**. Source revision trees and full ZIPs are private, outside this task. No remote freshness beyond these checked revisions is claimed.

## Fixed physical scope, discrepancies and identifiability

Unknowns Fp,vp,f,Tt plus their fitted measured signal; no independent truth for these parameters. Hidden `baseline_reference.npz` is the original standard-method result, never labelled truth. Derived GFR uses60 Fp f V in mL/min. Independent isotope values are retained only in the review reference and never used to calibrate public inputs. Source dictionary isotope label says mL/min but author's executable example treats values as mL/s; retain exact values and explicitly multiply60 following that example. This inconsistency remains a limitation, not independently resolved unit truth.

Source uses field/agent relaxivity4350 at1T and3500 at3T in s^-1 M^-1, blood T1=1.378/1.6286644951140066s, literature kidney T1=1.017/1.542s;144 kidneys have supplied measured precontrast T1. These modern defaults are not asserted identical to historical rounded paper constants. Per-case TR5.4/5ms is preserved from CSV, not harmonized to rounded prose. Public haematocrit.45 is a modelling assumption. Filtration ratio f=Ft/Fp differs from extraction fraction f/(1+f). Zero initial tracer and measured time sampling are explicit. Source handles impossible arterial normalized signal >=1 by frame-index interpolation; preserved without target information.

All228 retained, including11 transit estimates >10000s (maximum103691.483s) and21/52/96 upper-bound Fp/vp/f fits. Active profile refits expose near-flat long-transit directions. These method outputs are not claims of uniquely identified physiology. Primary measured-T1 clinical GFR bias10.693mL/min, SD25.439, RMSE27.595, P30=32.5%; literature-T1/linear/delayed comparisons also shown. A clinically poor standard method is not called clinically accurate; the task validates its parameter inverse and signal consistency, with no clinical error acceptance relaxed to hide failure.

## Original execution and independent freeze

Private oracle env `bio_med/oracle-env`, Python3.11.11 Windows10 build26200, NumPy2.4.6/SciPy1.17.1, single-thread CPU OpenBLAS/OMP/MKL=1. `capture_renal.py` inserts the unmodified source checkout; only CSV loading replaces pydmr IO. Original dc.Kidney, train/predict, concentration and relaxation operators execute on exactly the bundled decimal observations. No seed or stochastic algorithm. All228 original and684 comparison fits returned without optimizer warnings. Main originals13–16s depending run; first independent repeat and final logged repeat produce identical outputs. Private JSON field `log_nrms_percent` is misnamed: dc.cost is100*L2 residual/L2 observation, not log residual; every released display/metric uses its actual formula.

Before extraction, `renal_pre_extraction_contract.json` froze original arrays, stage tolerances1e-10, Fp/vp/f scaled RMSE<=1e-4 at scales.05/.3/.3, transit RMS log ratio<=1e-3, maximum signal reference L2 ratio<=1e-6 and observation ratio<=.2. Observed original maximum.162584. No post-hoc relaxation. Independent original rerun max difference0. Stage analytic probes frozen separately. A transcription error vp initialization.10 initially caused different weakly constrained solutions; original fixture required.15, correction restored exact agreement. No change of optimizer or objective.

## Equation/stage traceability

| Mathematical stage | Original file/function | Local function | Independent validation |
|---|---|---|---|
| Invert spoiled GRE / baseline | src/dcmri/sig.py conc_ss, signal_ss; ui_kidney.py | physics_model.arterial_concentration, steady_signal | all228 original ca, analytic signal roundtrip |
| Plasma/tubular transport | src/dcmri/kidney.py conc_kidney | physics_model.compartment_concentration | original probes and all228 compartments; independently integrated ODE and total mass balance |
| Piecewise-linear exponential integration | src/dcmri/utils.py expconv | physics_model.exponential_convolution | nonuniform upstream probe; analytic constant-input solution |
| Relaxivity / signal formation | src/dcmri/ui_kidney.py conc, relax, predict | physics_model.predict | all228 original R1 and magnitude |
| Bounded inverse / initial conditions | src/dcmri/ui.py train; ui_kidney.py free defaults | solvers.fit_case, reconstruct | frozen initialization; primary228+3x228 full-fit parameter/signal parity |
| Linear/delay/literature comparisons | same Kidney sequence/free/calibration branches | solvers.fit_case variant; physics_model.predict delay | complete original/fresh comparison arrays |
| Clinical derived GFR | author validation example, Kidney.export_params | Notebook60*Fp*f*volume | source GFR arrays retained; physical unit convention explicit |
| Metrics / strict standalone Judge | independent local evaluator | evaluation/judge.py | known numerical metric, each output corruption, malformed contracts and foreign cwd |

## Validation execution

- **original passed**, exit 0, 16.947 s: `D:\yan1\agent\AutoSkill\imaging-101-maintainer\research_batch10_20261027\bio_med\oracle-env\Scripts\python.exe D:\yan1\agent\AutoSkill\imaging-101-maintainer\research_batch10_20261027\bio_med\capture_renal.py`. Log `bio_med/renal_original.log`.
- **original_variants passed**, exit 0, 30.246 s: `D:\yan1\agent\AutoSkill\imaging-101-maintainer\research_batch10_20261027\bio_med\oracle-env\Scripts\python.exe D:\yan1\agent\AutoSkill\imaging-101-maintainer\research_batch10_20261027\bio_med\capture_renal_variants.py`. Log `bio_med/renal_original_variants.log`.
- **cli passed**, exit 0, 2.937 s: `D:\yan1\agent\AutoSkill\imaging-101-maintainer\research-runtime\Scripts\python.exe -m src.main`. Log `bio_med/renal_cli.log`.
- **judge passed**, exit 0, 0.139 s: `D:\yan1\agent\AutoSkill\imaging-101-maintainer\research-runtime\Scripts\python.exe evaluation/judge.py --prediction output/reconstruction.npz --report output/judge_report.json`. Log `bio_med/renal_judge.log`.
- **tests passed**, exit 0, 24.410 s: `D:\yan1\agent\AutoSkill\imaging-101-maintainer\research-runtime\Scripts\python.exe -m pytest test/ -q --basetemp=D:\yan1\agent\AutoSkill\imaging-101-maintainer\research_batch10_20261027\bio_med\pytest_renal_logged`. Log `bio_med/renal_tests.log`.

Native notebook command: `python -m jupyter nbconvert --to notebook --execute notebooks/renal_filtration_fraction_tubular_transit.ipynb --ExecutePreprocessor.timeout=600 --output renal_executed --output-dir ../..`, exit0. Final active primary run3.763s, four complete fits22.558s plus profile optimization and total/split clinical comparisons. Notebook uses only local cleaned modules. All8 PNG groups actually inspected: acquisition, parameter parity, measured curves/four predictions, clinical scatter+Bland–Altman, four-model clinical residuals, failure/bound distributions, transit profiles, total GFR/split-function comparisons. Labels/units/legends readable with no cropping. Canonical notebook retains executed cells; temporary execution copy is archived privately. Generic Python3 kernelspec uses private JUPYTER_PATH pointing at validated shared runtime, no machine path in task metadata.

Tests18passed: original full pipeline/stages, nonuniform convolution, independent ODE/mass balance, signal inverse roundtrip, unit/constants, public padding/no isotope, clinical known-case, Judge negative/error/foreign-cwd/report-failure. Judge fresh result passes: four parameter and reference-signal errors0, max observation relative error.16258405305627424, padding0. JSON finite, all outputs evaluated. No NCC/dynamic-range NRMSE is claimed; exact definitions in metrics.json and README. All truth-unavailable comparisons are method-reference metrics.

## Public boundary, code and document audit

README+raw_data.npz+requirements alone expose sufficient observation geometry/calibration and complete output contract. All public keys batch-first and listed with units. Isotope targets, original fits, covariance and source links remain hidden. Public files contain no GitHub links. Core forward and inverse mathematics is explicit local code; only NumPy operations and SciPy general curve_fit/least_squares/solve_ivp are used. Source, tests, Judge and notebook have no dcmri/pydmr/domain solver imports or cross-task imports. Judge is independent of solver modules. Matplotlib backend is not changed by libraries. Requirements NumPy/SciPy/Matplotlib pinned to actual shared runtime. AST/call-role review and plan/source semantic review checked transport factors, molar units, initial vp=.15, curve_fit bounds/default tolerances, delay interpolation, output shape, padding and CLI/notebook paths. No unresolved implementation discrepancy.

## Reference consumers

`test/fixtures/upstream_stages.npz`: test_original independent forward/intermediate/probe comparisons. `evaluation/baseline_reference.npz`: Judge and full-pipeline test, original parameter/signal outputs. `notebooks/reference_outputs/upstream.npz`: original primary/three alternatives and independent isotope source values, consumed by Notebook and comparison tests. `notebooks/reference_outputs/cleaned.npz`: fresh full active notebook result, labelled optional review cache, never used as active reconstruction. `output/reconstruction.npz`: fresh CLI submission; output/judge_report.json fresh independent evaluation. No hidden parameter truth is invented.

## Mandatory gate ledger

All mandatory gates passed: bound primary paper/revision/data hashes and traceability; original exact-data run; prior independent fixtures/physics; constants/units; CLI/Judge; negative/error contracts; public sufficiency and no leaks; shared minimal dependencies; explicit local mathematics/call roles;18tests; full active Notebook and8figures; original/fresh/clinical provenance separation; README/plans mathematical/interface review; template/license/inventory cleanup. Scientific limitations above remain explicit and are not labelled failed software gates. Parent independent acceptance pending.

## Maintainer-only archive index

All names below relative to `research_batch10_20261027/bio_med/`; full checkouts `dcmri_source/`, `renal_paper_source/`; raw members `renal_data/`; private figure captures `renal_figure_review/`. Data CC-BY4.0 attribution in NOTICE; extracted original code Apache2.0 LICENSE retained. Paper/archives/logs are not shipped task assets.

| Archive asset | SHA256 |
|---|---|
| `renal_author_example.html` | `7a6879c00d6f38495431ff6c3c237d12dcffd5bb0cf74d12d915db2d64a7e113` |
| `renal_author_example.py` | `fd77761fe4aee10708626f550d66d4ab96bf78cf9179bf3081da07947a2b7b25` |
| `renal_paper.pdf` | `db20452534754b57d2511a67d7ca1a04647eb3c54d42d062a9a9349bcf8500e5` |
| `renal_paper.txt` | `5b6de6b469ceef1ac79b9df98e86751b5e79a5f35cb0bb698313be436b277b38` |
| `renal_paper_source.zip` | `4b0ebaf4da4e76691bf1101dadf24cfa2329bfa46f3a14f95a03be846b94b275` |
| `renal_paper_repo_tree.json` | `144a734b2e035774d84744a2ebb460b343cbdb0f7081715afa1284479d3df667` |
| `dcmri_source.zip` | `b04de056f253a96d86f924a624e088d8f5547365bed62c50df4b86f74570253c` |
| `dcmri_tree.json` | `1ffcbcb4ace0abba00f3d59ed25035dd5f63e295123924576cfe036025a314fb` |
| `renal_KRUK.dmr.zip` | `6e1941aff5deb3b923e9d9949fc5fa389fbe1bf6b14e6cd58d4f98afdb7d386b` |
| `renal_datacite.json` | `21204768a940ddd5020b1afdbe7f2d2c312880235060e17a88b344e2926f9ea0` |
| `renal_original.npz` | `9c9b8e0c6e7232b24d57a0665404884317428d25be52795b0d821fbd770cf6d1` |
| `renal_original_first.npz` | `9c9b8e0c6e7232b24d57a0665404884317428d25be52795b0d821fbd770cf6d1` |
| `renal_original_records.json` | `c5a7e3503ffa48eea08f37720e7e5de0a86da917adbdb44669f2b18e5e61873f` |
| `renal_original_first_records.json` | `0c1b288208d4c3de4ea91776b11f396714c138383fb25670bbe3921c5cf51216` |
| `renal_original_variants.npz` | `5274fc6ee39da390517ca6810f266c69acad4cf6a1076194d09882208e688ed9` |
| `renal_original_variants_records.json` | `015a8892024f8dc6b7a0d7effd5b376cf4b83b5079dab6abee9f6df67eec89a6` |
| `renal_physics_stages.npz` | `1604bd2c403a9f97f98b102a81ef94ae548fdf5a539eef564e0b37385964c836` |
| `renal_pre_extraction_contract.json` | `03b20ffe5b1797e35814763018b766abc677e9042066388c6b76710c6c73ecad` |
| `capture_renal.py` | `1bf9362c76146cf9f993f175edd45685ab783fe17f12ad68d27e1eaa910140c9` |
| `capture_renal_variants.py` | `95d5e3ada1621c3240a034c510ac9b6d02b60427acd7a2debd32179f82fc43d7` |
| `renal_executed.ipynb` | `305a1accbf763d34ff32f46c9092513809162306b62cc9b078c8ff26affcd67a` |
| `renal_validation_records.json` | `e69415ce05766fc7b4bb2857e06948cc485d4b41ad0adb13ce1068a2507e8a9a` |
| `renal_original.log` | `c139ae6be413859fa87232b2e6a294ae21a899eab297bcc19b42e359252bf172` |
| `renal_original_variants.log` | `80a2d26ef1cdf25ae84d7090bc86395a2af523258fa176f2f121c6d3c3be3364` |
| `renal_cli.log` | `c6ee33b74a6e5d2fd40fa32ef2bde92f4efef22deb3f6f32b29530a632004acf` |
| `renal_judge.log` | `7eb70257593da06f682a3ddda54a9d260d4fc514f645237f5ca74b08f8da61a6` |
| `renal_tests.log` | `c8e7b509fe9e32006dc4fce6673fb559a438670734f83837f1cac98347e09f8a` |

Public NPZ SHA256 `040b4bba5cad2a5af66e733c24bc9be3adaf98e423cfbe6e9d900c505fedc142`.


## Independent parent acceptance

CLI, standalone Judge, pytest, active nbconvert and dual-engine math rendering were independently rerun: all exit0. Parent notebook: 8 executed code cells, 8 computed PNG outputs, 57 formulas. Private evidence: research_batch10_20261027/acceptance/renal_filtration_fraction_tubular_transit/execution.json and semantic_review.json. All panels actually inspected by parent; source/physics/GT roles and limitations semantically checked. Canonical notebook is this parent-executed artifact. No new source or notebook execution is inferred from cached upstream outputs.


## Public-contract revision 2026-10-29

2.0-observation-identifiable-renal-class: V1 demanded identity with a weakly identifiable finite optimizer parameter estimate; v2 independently validates all GRE signal/parameter identities and per-case squared-error budgets. An independently fitted 200000-second transit alternative passes; physically valid zero-flow controls fail. No actual truth exists. Full source algorithm and arrays unchanged.
