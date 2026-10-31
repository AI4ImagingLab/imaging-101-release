# Provenance and validation: joint protein CD–IR structure

## Fixed scientific binding

Biology; infer four secondary-structure fractions for four measured protein spectra by the complete joint SELCON3 method. Targets are CD-IR28 columns0,10,18,27: myoglobin, lysozyme, ubiquitin, concanavalin A. **All four target columns and all four target annotations were removed from public calibration**, leaving24 independent proteins. This is a deliberate four-protein holdout benchmark, not the paper's leave-one-out27-reference experiment. Every original-code comparison was rerun on this exact24-reference binding. Nearest old tasks involve EPR distances/FRET folding landscapes, not joint electronic/vibrational structural composition; root reserved this new acquisition/unknown pair before extraction.

Paper: Hoffmann, Jones and Rodger, *QRB Discovery*6,e10(2025), DOI https://doi.org/10.1017/qrd.2025.4 . Full author article HTML and10-page supplementary PDF archived privately; methods, acquisition-source clarification, supplementary S1/S2 and figure comparison logic inspected. Author original repo https://github.com/AU-SRCD/SSCalcPy , remotely verified exact commit **7284986d0a0aaac1f4feaefeea56de1f5615851e**, archive1,180,569bytes. Original SelconsFunction.py identifies Hoffmann's2021 Python conversion of the Wallace-group2005 Matlab code and2006–2007updates. Source and paper CC-BY-NC-SA4.0, copyright2024AU-SRCD/code and2025authors/paper; original license plus attribution preserved in NOTICE. No commercial-relicensing claim is made.

## Instrument-measured source evidence and limits

Classification **author_real_processed**. Paper methods identify CD-IR28 as the28 common proteins with experimental CD in SP175/SMP180 and IR in RaSP50. CD comes from PCDDB curated experimental spectra in per-residue differential molar extinction. SupplementS1 explicitly identifies the author's source file AU-A128_PCDDB-Nov22.txt and notes a small baseline correction to lysozyme based on the authors' own SRCD measurement; source processing is preserved, not mislabeled untouched raw instrument output. The author combined file AU-A28_CD-IR_Aug24.txt is the bound source. Independent calibration fractions are structural annotations from known crystal structures, not estimates from the target observations.

IR provenance is especially important: the paper's methods and open peer-review clarification establish the **dry protein film ATR remeasurement** of RaSP50, from Goormaghtigh et al.(2006), DOI10.1529/biophysj.105.072017, reused through Pinto Corujo et al.(2022), DOI10.3389/fchem.2021.784625. It is not the earlier2003 liquid-sample RaSP50 acquisition. The processed spectra span1600–1800cm^-1; baseline subtraction, zeroing1718cm^-1 and unit-maximum normalization are documented. File-specific instrument serial, concentration, pathlength and complete raw optical calibration are not available in the package and are not invented. CD solution and IR dry-film measurements share protein identity but not necessarily conformation. The private crystal annotations are an independent structural **proxy**, not same-condition GT. Accordingly evaluation uses baseline_reference, never ground_truth.

The source full matrix contains267×28values:66CD wavelength bins240…175nm then201IR bins1600…1800cm^-1. Every public value is exactly the original np.loadtxt(dtype='f',delimiter='\t') float32 value; no unvalidated precision change. The source reduced matrix127×28 is exactly full[0:66] concatenated with full[66:187:2], verified across all28proteins, so active local selection introduces no interpolation. Public NPZ provides full267bins for target and calibration proteins, enabling IR-only comparisons without privileged observations. Calibration fractions are4×24. Public targets are4×267. No identical target occurs in the calibration library; even CD-only nearest-library RMS distances are1.46435,.39219,.75005,.93758, so no direct target copy explains good refits. The query itself is legitimately augmented during the self-consistency algorithm without inserting its true fractions.

## Original runs and frozen criteria

Original code ran unmodified in private oracle-env Python3.11.11/NumPy2.4.6, CPU OMP/OPENBLAS/MKL=1. It requires only NumPy; no installation into shared cleaned runtime. `capture_sscalc.py` invokes SelconsPy directly with plotting disabled but all numerical stages enabled. Python sys.settrace captures numeric local variables at return without modifying original equations or control flow. Original text refits are also written privately but rounded text is not used for parity. The capture preserves full-precision HJ5, sorted indices, SELCON1/2/3, prefix/rank choices, iteration counts, threshold values, candidate spectra, means and population spreads.

There are28complete original runs: four targets×IRweights0,1,5,10,15,20 plus IR-only. Weight0uses the original CD-IR mode with IR omitted, preserving four structural categories. IR-only uses all201IR values and its original.025refit threshold; joint/CD.25. First28runs took1.224226s, second1.218810s; all476initially captured stage/final arrays matched exactly. The second capture additionally recorded28stDev arrays. A final original audit rerun is logged below. No seed or stochastic inverse exists.

`sscalc_pre_extraction_contract.json` was frozen before local extraction: defaultIRweight15; all4targets withheld; parity rtol=atol1e-9; maximum fraction error1e-6; CD-refit RMSE1e-6; unscaledIR-refit RMSE1e-7; sum deviation<=.06; fractionminimum>=−.025. Independent rerun agreement supports nonzero numerical tolerances; these are not accuracy thresholds against crystal structures. No relaxation was introduced to conceal divergence. Local default and all28comparisons match their independent original arrays in this environment, including every frozen intermediate stage. The selected local SVD/pseudoinverse operations are general numerical primitives; all target-specific selection and iteration is explicit local code.

Default iterations are2,3,5,6. Retained candidate counts13,7,9,10. Concanavalin A triggers the **original SELCON2 fallback** because the helix filter has no surviving candidate; this is preserved and explicitly printed, not silently omitted or described as a successful strict-helix subset. Fraction sums .9866715269537591,1.005514679711287,.9857729101043661,.965751625467266; minimum−.009975061264275217. Signed values and nonunit sums remain unaltered. The published selection uses a zero-based best_basis>0 test, excluding a rank-one winner. This convention is retained. The possible degenerate branch in which valid candidates exist but all winners have rankone would produce an empty mean upstream; local code raises a descriptive error there. No bound default or comparison takes that branch, and this exceptional diagnostic does not alter selected numerical results.

## Mathematical and code traceability

| Paper/reference stage | Upstream source | Local executable stage | Validation |
| --- | --- | --- | --- |
| Spectral pairing/block selection/scaling | LeaveOneOut_AR.py, CD-IR loader | preprocessing.prepare | Source-full/reduced exact equality; array/no-overlap test |
| Rank-five HJ estimate | SelconsFunction.py HJ/HJ5 | spectral_svd,structure_estimate; selcon.hj5 | Frozen HJ5 for all4targets; known two-component mixture |
| Similarity sorting/initial augmentation | SelconsPy Arrange A/F block | solvers.selcon sorting/augmentation | Frozen ix, no target-annotation calibration |
| Variable prefix/rank fraction selection | SelconsPy SELCON1 | select_candidates | Frozen sel1 and fullpipeline; exact sum/tie/rank conventions reviewed |
| Self-consistency iteration | SelconsPy F[:,0] update,.0025/50stop | selcon loop | Original iteration counts and sel1 parity |
| Spectral residual selection | SelconsPy SELCON2 | spectral_refit and selcon threshold loop | Frozen sel2/refit_rule; SVD projector/reconstruction physics |
| Helix rules and fallback | SelconsPy SELCON3 | helix_selection,selcon fallback | Frozen sel3/prefix-rank pairs; boundary branch tests |
| Mean and candidate refit spectra | SelconsPy Mean_refit_prot | spectral_refit,selcon/reconstruct | Full spectra parity; all28mode/weight tests |
| Structure/refit metrics | Explicit task contract | Judge and visualization.rms_error | Independent known numeric cases and strict contract tests |

No gradient/adjoint/proximal update is required by this SVD/self-consistency algorithm; independent projector symmetry/idempotence, analytic mixture inverse, SVD reconstruction and helix tests apply instead. np.linalg.svd/pinv are permitted primitives; core acquisition representation, weighting, subset/rank selection, adaptive method rules, iteration, helix selection, fallback and refit are local readable equations. AST imports and actual call roles inspected across source/Judge/tests/notebook: only standard library, NumPy and Matplotlib in cleaned runtime, pytest for tests. No domain library, opaque framework, cross-task import or upstream runtime source is imported. No synthetic generator exists.

## Results, notebook and provenance

Hidden baseline_reference stores this task's original default fractions(1,4,4) and unscaled mixed-unit refits(1,4,127). It is not truth. Test fixtures contain only consumed original default stages with output_ names. Notebook upstream.npz is consumed for all28fresh comparisons and also contains clearly identified withheld crystal_proxy; these are private. Fresh notebook output is stored separately in cleaned_measurements.npz. No historical paper score is used as measured evidence. The published full28-protein LOOV result is contextual only; this task uses a four-protein holdout and new original runs.

Canonical notebook actively reconstructs full default and all28comparisons from public input. Explicit sections Problem/Data/Physics/Method/Results/Conclusion cover the empirical mixture equations and actual selection method. Eight figure outputs were individually viewed: measured paired spectra; calibration composition/singular values; original/fresh/crystal fractions; four-protein paired measured/refit spectra; modality residuals; CD-only/IR-only/joint/proxy grouped bars; weighting sensitivity with separate CD/IR RMSDs; retained candidate refits, fraction spread and per-protein proxy error bars. No scientifically relevant comparison is replaced by zero or cached output. CD-only has no IRfit, and that missing modality is omitted explicitly, not filled with zeros. The measured myoglobin CD-only refit is numerically exact under the augmented selected subspace; it was actually computed. Candidate spreads are not called posteriors. Figures/labels/legends inspected in private sscalc_figure_review/; no clipping or hidden target use.

Fresh default computation~.086s. Judge reports maximumfraction/CDrefit/IRrefit errors0, sumdeviation.034248374532733994, minimum−.009975061264275217. Independent structural proxy errors remain visible and vary byprotein; method-reference agreement is not asserted as physical accuracy.

## Execution environment and logs

Shared runtime Python3.11.11 Windows10build26200, NumPy2.4.6,Matplotlib3.11.1, CPU single-thread. Original isolated environment uses the same NumPy version but independently executes archived source. No shared environment installation or GPU requirement. Full benchmark arrays are kilobytes; headroom exceeds6GiB physical memory and15GiB disk. Native nbconvert uses a private external kernelspec pointing to the validated runtime; notebook metadata is genericpython3. Warnings from frozen debugger modules, Windows zmq selector fallback and local TCP transport are benign, with zero error cells. Pytest uses private external basetemp to avoid inherited Windows temp permissions. Full commands/logs/hashes/import audit are in sscalc_validation_records.json outside task.

| Command | Exit | Wall seconds | Private log |
| --- | --- | --- | --- |
| `D:\yan1\agent\AutoSkill\imaging-101-maintainer\research_batch10_20261027\bio_med\oracle-env\Scripts\python.exe D:\yan1\agent\AutoSkill\imaging-101-maintainer\research_batch10_20261027\bio_med\capture_sscalc.py` | 0 | 1.371481 | sscalc_original_command.log |
| `D:\yan1\agent\AutoSkill\imaging-101-maintainer\research-runtime\Scripts\python.exe -m src.main` | 0 | 0.217088 | sscalc_cli_command.log |
| `D:\yan1\agent\AutoSkill\imaging-101-maintainer\research-runtime\Scripts\python.exe evaluation/judge.py --prediction output/reconstruction.npz --report output/judge_report.json` | 0 | 0.136873 | sscalc_judge_command.log |
| `D:\yan1\agent\AutoSkill\imaging-101-maintainer\research-runtime\Scripts\python.exe -m pytest test/ -v --basetemp=../../pytest_sscalc_final` | 0 | 1.780281 | sscalc_tests_command.log |
| `D:\yan1\agent\AutoSkill\imaging-101-maintainer\research-runtime\Scripts\python.exe -m jupyter nbconvert --to notebook --execute notebooks/protein_joint_cd_ir_structure.ipynb --inplace --ExecutePreprocessor.timeout=180` | 0 | 7.395023 | sscalc_notebook_command.log |

18tests passed. Native complete notebook exited0. Source signatures and all equations/factors/signs/array shapes/dtypes/units/caller paths reviewed against README and both plans; no proposed unimplemented method is described as runtime. Public source references contain no GitHub URL; visible arrays contain only target observations and independent calibration.

## Private archive index/SHA256

Archive root `D:/yan1/agent/AutoSkill/imaging-101-maintainer/research_batch10_20261027/bio_med/`. Original checkout in sscalc_archive/, original text outputs in sscalc_oracle/, original capture/log/records, paperHTML/supplementPDF, frozencontract and figureQA remain outside task. No suppliedPDF or source checkout is bundled.

| Artifact | SHA256 |
| --- | --- |
| sscalc.zip | d108875da2a1e4add9ce6a54f29546be3a77d3f3a0fcc6df2bfd17de4e9d5b11 |
| sscalc_paper.html | a8d267aa1181d48e97c5edc18bc8eb2edf1e6e13f577efa87a18b93542d47f67 |
| sscalc_supplement.pdf | 498a458130826792e4e7a37eb966de5fa72d347d9228da14a582f2254c248b81 |
| sscalc_oracle.npz | ebb1b0aea734868111e1b05f85bc35af07b68e9a00252893c43383b927b93655 |
| sscalc_oracle_first.npz | 2c169f76e1b3407f8d972f42cff9269f361e0e6296ccc3a0787f3973d51ea402 |
| sscalc_pre_extraction_contract.json | fe899443f15f22f4a6a89db91e8cdbb39afd18d52c7dd3e4f2583054b3948396 |
| capture_sscalc.py | ac5c296ee2e928698467801d5c2c1caf0fc08ab2fcb031b5748bf135c6077eb8 |
| sscalc_archive/SSCalcPy-7284986d0a0aaac1f4feaefeea56de1f5615851e/SSCalcPy_Release_3rdFebruary2025/Selcon3Py/SelconsFunction.py | d91da1339a2f44a2319d9643f0f9e81ff5bbdbb4b809e6dbbff606c07b1d0c45 |
| sscalc_archive/SSCalcPy-7284986d0a0aaac1f4feaefeea56de1f5615851e/SSCalcPy_Release_3rdFebruary2025/CD-IRRefDataAU/AU-A28_CD-IR_Aug24.txt | d102041b267252b57cff0351c58e111936c1f48772889b7362ee822eb8631158 |
| sscalc_archive/SSCalcPy-7284986d0a0aaac1f4feaefeea56de1f5615851e/SSCalcPy_Release_3rdFebruary2025/CD-IRRefDataAU/AU-A28_CD-IR2cm-1_Aug24.txt | c6a07dec708285aa5fee2b1ed11740f6b3399f8071836a5e1b12286ba4ca5750 |
| sscalc_archive/SSCalcPy-7284986d0a0aaac1f4feaefeea56de1f5615851e/SSCalcPy_Release_3rdFebruary2025/CD-IRRefDataAU/AU-F28_CD-IR_Aug24.txt | 253a561851f169e4bff9da8fbc59435988d2b8a6dc135c03eebef4c9f2e952cc |
| tasks/protein_joint_cd_ir_structure/data/raw_data.npz | be593d18e931b057ecddaeb32ffd84113c4c117f4ce2d456bb94f147c94d96f2 |
| tasks/protein_joint_cd_ir_structure/evaluation/baseline_reference.npz | e34c7e8647926948d48e10cc681eb75fac0034670dfe4b55a82d121ac9c41d67 |
| tasks/protein_joint_cd_ir_structure/notebooks/reference_outputs/upstream.npz | 9c1b94519fae597103094f875229776d573b79afdc88585d5c55a9aba0eeb261 |
| tasks/protein_joint_cd_ir_structure/notebooks/reference_outputs/cleaned_measurements.npz | e34c7e8647926948d48e10cc681eb75fac0034670dfe4b55a82d121ac9c41d67 |

## Gate checklist

| Gate | Status | Evidence |
| --- | --- | --- |
| Paper/revision/data/mapping | passed | Primary full paper/supplement, exact repo/hash/data chain above |
| Original and cleaned exact observations | passed |28original and28local comparisons, independent captures |
| Stage parity and independent mathematics | passed | All default stages and modes, mixture/projector/selection tests |
| Constants/units/dtypes | passed | Float32loader preserved, exact full/reduced grid identity |
| CLI/strict multi-output Judge | passed | Both fractions and refit blocks scored |
| Poor/malformed/evaluator-error rejection | passed | Missing, keys, shape,dtype,complex,NaN/Inf,undefinedconfig,reportwrite,foreigncwd |
| Public sufficiency/no target leakage | passed | All4targets/annotations excluded from24calibration; only measured arrays |
| Explicit math/minimal dependencies | passed | Source/call-role review, no domain runtime |
| Unit/parity/pipeline suite | passed |18tests |
| Active notebook/comparisons/figures | passed | Complete native execution;8figures individually inspected |
| Historical/original/fresh separation | passed | Historical paper claims not substituted; independent structural proxy labeled |
| README/plans/signatures/equations | passed | Semantic and interface review |
| Inventory/license/cleanup | passed | Template only, original license retained, caches external |

File-specific raw instrument calibration and same-condition structural truth are unavailable, explicitly disclosed scientific limitations. No mandatory execution or comparison gate remains failed/unverified. Parent review is still required before integration.


## Independent parent acceptance

CLI, standalone Judge, pytest, active nbconvert and dual-engine math rendering were independently rerun: all exit0. Parent notebook: 6 executed code cells, 8 computed PNG outputs, 25 formulas. Private evidence: research_batch10_20261027/acceptance/protein_joint_cd_ir_structure/execution.json and semantic_review.json. All panels actually inspected by parent; source/physics/GT roles and limitations semantically checked. Canonical notebook is this parent-executed artifact. No new source or notebook execution is inferred from cached upstream outputs.


## Public-contract scientific isolation 2026-10-29

A validated public statistical target for the discrete self-consistent solution ensemble is unresolved. Benchmark eligible=false. Scientific review is completed; blind-agent execution is a separate unexecuted gate and is not the reason for isolation. Removed README recipe, where present, is archived privately under benchmark_readiness_20261029/private_readme_recipes. Existing arrays/Judge thresholds unchanged.
