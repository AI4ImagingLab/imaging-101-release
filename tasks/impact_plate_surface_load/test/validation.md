# Provenance, independent evidence and completion gates

## Scope and immutable scientific contract

Domain mechanics. Recover a complete 250-frame spatial pressure field from real optically estimated steel-plate deflection; this is inverse dynamics, not displacement registration or forward prediction. The original280 real camera frames are acquired data; the public displacement is a derived observation estimate, not displacement truth. The selected output is the original sigma6-frame/20-square-Hermite-patch effective load at all valid spatial positions. No pixel-pressure ground truth is available. Hidden baseline is a separately executed original mechanics method. The independent hammer is a force measurement and is never relabeled pixel truth, public calibration or used for fitting.

Existing139 checked in batch index: particle_image_velocimetry targets velocity; finite_element_speckle_dic targets displacement; mdck_finite_thickness_traction targets cell-substrate traction; bayesian_monolayer_stress_mapping targets in-plane cell stress; checkerboard_water_surface_height targets water height. Here the unknown is transient external plate pressure governed by transverse plate bending and inertia. Parent reserved this new scope before task creation. Previous accepted POLCAM and old tasks are untouched.

## Sources and archive

Code: https://github.com/PolymerGuy/recolo , exact remotely verified API revision `05b14f0834fa675579eabdf43fac046259df19bb`, MIT. Private API record `../PolymerGuy_recolo_api.json`, archive `../recolo.zip`, unpacked `../upstream/recolo-05b14f0834fa675579eabdf43fac046259df19bb`. MIT copyright preserved in LICENSE and acquisition modification identified in NOTICE. This is a remotely verified revision, not a freshness claim based on local tracking.

Paper: Olufsen et al., published JOSS7(71),3980, 2022-03-07, DOI10.21105/joss.03980. Full two-page paper read. Author method Kaufmann et al., published Experimental Mechanics59,1203-1221(2019), DOI10.1007/s11340-019-00530-2; relevant full article virtual-work Eq4/5 and Hermite discretization read at publisher. JOSS PDF download returned403; browser-readable published source was used; do not claim a privately archived PDF. This access failure does not prevent the read source mathematics or original code validation.

Real data: https://doi.org/10.18710/TKS7SI , CC0, dataset API `../recolo_dataverse.json`. Complete downloaded archive `impact_dataset.zip` includes all280 grid images, independent hammer samples and metadata. No synthesized observation, fitted clock/amplitude, hidden pressure or target-derived calibration. Private root for paths in this section is `D:/yan1/agent/AutoSkill/imaging-101-maintainer/research_batch5_20261021/bio_med_mechanics/recolo`. `archive_index.json` records per-file SHA256/bytes including individual acquired images, archive, logs and captures; complete source archive hash and API records are indexed. Private source, capture scripts, original images, environments and logs do not ship.

## Independent original run and investigated acquisition defect

Original command: shared maintainer Python `../recolo_oracle.py` with private oracle-deps path placed before numerical imports. Exact original demodulation/integration/kinematics/Hermite/VFM functions were executed on all bound experiment images. Plot styling and source's download GUI are not part of numeric path. `oracle_run.json`, `oracle_status.json`, `oracle.log` record exit0,159.0332354s, peak RSS1106608128bytes. Stages: grid100.8206293s; slope integration13.6788734s; mechanics5.5975936s. Captures `observed_slopes.npz`, `observed_deflection.npz`, `oracle_pressure.npz`. Original Python3.11.11, NumPy1.26.4/SciPy1.13.1/skimage0.22.0, CPU Windows. Shared environment untouched. skimage0.24/0.26 reject original unwrap_phase(seed=0); unsuccessful0.24 attempt explicitly archived, not counted success. Final0.22 emits the seed deprecation FutureWarning; captured in original log. Private bs4,natsort,wget,psutil are acquisition/monitoring dependencies, not cleaned requirements. Thread counts one. There is no hidden model training or random seed in solver; original unwrap seed0 is declared acquisition provenance, not a hidden object seed.

Independent inspection found a genuine source defect in `slope_integration/disp_from_slopes.py::int2D`: its first row-direction derivative assigns both coefficients to flat_inds instead of assigning the second to flat_inds+n_y. Coefficients therefore cancel and the equation does not constrain the intended derivative. This is not merely displacement gauge. A private minimal correction changes exactly that second column index; original checkout remains unchanged. A linear field w=x gives maximum analytic reconstruction error5.329070518200751e-15 after correction. `../recolo_boundary_investigate.py`, `boundary_investigation.log/json`, `corrected_boundary_investigation.npz` preserve the test and all real-frame comparison. Corrected versus unmodified deflection relative L2 change0.0005111984261777664, maximum1.0928929301954034e-6m. Whole-field force NRMSE changes0.308873574 to0.309652527; local force0.091148518 to0.091147702. The small downstream difference does not excuse copying an independently known erroneous boundary equation.

Bound observation therefore uses minimally corrected optical integration, followed by UNCHANGED original kinematics and mechanics. `../recolo_corrected_capture.py` reruns original mechanics on that exact corrected bound input: exit0,12.6795902s including compressed capture, recorded `corrected_original_mechanics_run.json`. `corrected_oracle_full.npz` holds stages independently of cleaned code. This is explicitly a corrected-acquisition oracle, not unchanged whole-pipeline reproduction. Unmodified full-pipeline field is retained separately in the hidden baseline for a distinctly labeled comparison. The correction never references pressure/hammer truth and was determined by derivative index mathematics before selecting output quality.

Source geometry preserved: actual example E190GPa,rho7934kg/m3,distance1.385m; data text rounds rho7900 and distance1.37. README discloses discrepancies. Original first30 frames reference, processed30:280, local clock0..3.32ms, pixel spacing0.0030065919701586503m, row=x/column=y, negative displacement gradient, valid convolution, shift=False all preserved. Corner gauge and original sparse solve retain source conventions after the one explicit index correction.

## Equation / function / independent-test traceability

| Stage | Paper mathematics / upstream function | Local executable implementation | Independent evidence |
|---|---|---|---|
| Derived optical measurement | JOSS workflow; grid_method.detect_slopes; slope_integration.disp_from_slopes/int2D | Public NPZ only; no optical inverse in released mechanics boundary | Full real-image original run; analytic ramp and minimal corrected acquisition capture |
| Gaussian temporal measurement conditioning | Example sigma6, kinematic_fields_from_deflections | preprocessing.filter_deflection | Frozen entire250-frame original filtered data; independent constant check |
| Plate kinematics | Curvature minus second deflection gradient; inertia second time derivative; kinematic_fields_from_deflections | physics_model.kinematics | Frozen original slopes/curvatures/acceleration on frames0,80,140,249; analytic rigid slopes/sign |
| Isotropic rigidity | D11=Eh^3/[12(1-nu^2)],D12=nuD11; example plate | physics_model.bending_stiffness | Cubic thickness law, physical constant |
| Compact Hermite field | Published four-Hermite-element virtual displacement; virtual_fields.Hermite16.__hermite_16__ | physics_model.hermite_virtual_fields | Independently captured original arrays; centrosymmetry; direct polynomial derivatives |
| Dynamic virtual work | Method Eq4/5; solver_VFM.calc_pressure_thin_elastic_plate | physics_model.virtual_work_terms,pressure_virtual_work; solvers.pressure_from_fields | Original full pressure/stages; constant external-work and inertia exactness; quartic bending |
| Full inverse and output coordinates | Source complete time loop, valid moving patch/shift=False | solvers.reconstruct; main.run | Full250-frame upstream parity; foreign-working-directory end-to-end/Judge |
| Metrics and input contract | Explicit benchmark definitions | Independent evaluation/judge.py; visualization.ncc_nrmse | Known-vector metric test; malformed/poor/evaluator/report tests |

Independent quartic w=(x^4+y^4)/24 has continuum pressure2D11. Source midpoint Hermite quadrature analytically yields factor1-4/N^2, i.e.0.99 forN20. Local test verifies this exact discrete result and independently quantifies1% continuum bias (limit1.1% selected for this physical diagnostic). No discretization change is hidden. Constant acceleration gives exact rho*h*a and zero bending. These checks validate signs, material units, cross factors, normalization and lack of an extra spatial-area factor; matching copied upstream arrays alone would not establish these facts.

## Frozen fixtures and metrics

`test/fixtures/upstream_stages.npz`: complete bound deflection/filter, selected-frame kinematics, original virtual arrays and pressure; consumed by test/comparing_test/test_upstream.py only. Values captured from unchanged original mechanics before local extraction, not generated by cleaned functions. `frozen_tolerances.json` privately records pre-extraction acceptance: filteratol1e-16m, curvature rtol/atol1e-11, accelerationatol1e-9m/s2, virtual curvatureatol1e-9/m2, pressure rtol1e-10/atol1e-5Pa. These were not relaxed. After the independently justified acquisition correction the same tolerances were retained and fixtures recaptured from original mechanics on the new exact bound observation.

`evaluation/baseline_reference.npz` contains reconstruction (corrected-acquisition original mechanics), uncorrected_pipeline_reconstruction (unmodified full-pipeline comparator), and measurement_time/measurement_force (independent native hammer traces,243 samples). No ground_truth.npz is invented. Full parity test/Judge/notebook consume reconstruction; notebook consumes comparator; Judge/notebook consume measured force. `notebooks/reference_outputs/measurements.json` identifies all three numerical categories and their sources. Paper results are context, not substituted fresh values.

All pressure outputs batch1/time250/row60/column78 realfloat64Pa. Judge uses cosine NCC without mean subtraction and standard dynamic-range NRMSE. Numerical thresholds NCC>=0.999999999 and NRMSE<=1e-9 are nonzero roundoff tolerances, justified by original NumPy1.26/SciPy1.13 versus cleaned NumPy2.4/SciPy1.17 and explicit algebraic Hermite simplification. Fresh NRMSE5.013256813539885e-17, NCC0.9999999999999999. Independent physical metric force=sum(rows20:50,cols20:50)*Delta^2 on the source-fixed physical region and source clock, linearly interpolated hammer, no fitted scale/shift. Original corrected baseline error0.09114770181323001; boundary1.1*error=0.10026247199455302. Fresh force error0.09114770181323002. Region chosen in source example before this cleaning; full spatial field still receives parity metrics.

Core peak363.564543750032N; hammer365.52908593N. This independent integrated-force agreement does not certify pixel-pressure truth. Whole-field error0.30965252664975257 and strong late tail remain visible: differentiation and measurement boundaries create substantial artifacts. No quality claim is made for arbitrary unsmoothed contact pressure. Notebook fresh sigma3/6/9 and window12/20/28 are sensitivity experiments, not optimized canonical parameters; all are actually computed on250 frames.

## Clean runtime, notebook and review

CPU shared Python3.11.11/NumPy2.4.6/SciPy1.17.1/Matplotlib3.11.1. Only NumPy/SciPy are inverse runtime dependencies; Matplotlib is declared for maintainer visualization. No upstream packages imported by source/Judge/tests/notebook. `symbol_role_audit.json` records AST imports and zero forbidden-domain/public-link findings. Manual symbol/call-role review confirms explicit physical model, stiffness, local derivatives, cubic Hermite polynomials, normalization and closed-form inverse; SciPy provides general Gaussian filtering and convolution. No optimizer shortcut, opaque domain solver, cross-task import, hidden read, seed/object calibration or backend mutation. Closed-form method has no artificial iteration.

CLI/Judge/test/active-notebook commands, exit codes and exact timings: private execution_records.json. `python -m src.main`3.6806955s exit0; standalone Judge0.2303666s exit0; `python -m pytest test/ -v -p no:cacheprovider`36passed16.0251125s; full `nbconvert --execute`41.9957913s exit0 with300s timeout and explicitly selected shared kernel. Notebook actively reconstructs every bound frame and fresh sensitivity variants, evaluates strict Judge, then plots; no archived-as-fresh mode. Executed temporary notebook is private, final notebook retains reviewed outputs. output contains only reconstruction.npz and judge_report.json.

All five figure groups visually inspected PNG and corresponding SVG/PDF produced: optical observation/kinematics, pressure/bending/inertia, whole/core/hammer plus unmodified original comparison, Hermite/physical resolution, genuine parameter sensitivity. Colors use physical units, no hidden amplitude fitting; pressure panels share full extrema. Private figures/impact_plate_surface_load/numeric_sources.npz records exact plotted numeric arrays; metrics.json and hash_manifest.json preserve full precision and figure hashes. Figure boundaries/titles/colorbars checked; source-comparison RMSE explicitly differs from force error.

README/public-input sufficiency reviewed: every NPZkey/table, complete coordinate/output/metric convention, units/constants/valid-patch boundaries and fixed effective resolution present; no GitHub URL in public README/requirements. All actual truth/reference remains hidden. Both plans reviewed against actual signatures/defaults/call flow and equations, including negative curvatures,2xy factors, area cancellation, index center9.5, time edge convention and corrected acquisition stage. They describe only shipped runtime and explicitly private acquisition context. Dependencies pinned shared; no environment upgrade. Final file inventory follows template, source/data/archive/PDF/logs and temporary artifacts excluded. No generator because existing observations reused.

## Mandatory gate ledger

| Gate | Status | Evidence |
|---|---|---|
| Paper/revision/data hashes/equation mapping | passed | Sources, archive_index and traceability above |
| Original exact-bound and cleaned executions | passed | Unmodified full-pipeline and separately identified corrected acquisition?original mechanics; execution_records |
| Independent upstream parity and physical checks | passed | Frozen stages/full pressure; analytic ramp/quartic/inertia |
| Constants/units/precision/coordinates | passed | README and source constants tests; plan review |
| CLI and fresh standalone Judge | passed | Exit0/0, declared file/key/shape |
| Poor/error/Judge contracts | passed | Missing file/key, shape/dtype/complex,NaN/Inf,unavailable evaluator,report-write failure,poor prediction,foreign cwd tests |
| Public problem sufficient/no truth-calibration leak | passed | Manual array/README audit; solver reads only observation |
| Complete metadata/output and no extra metadataJSON | passed | README table and contract |
| Minimal dependencies/shared CPU compatibility | passed | Executed shared runtime; upstream isolated |
| Local executable core/no forbidden shortcuts | passed | AST/call-role audit, explicit source mathematics |
| Unit/parity/physics/pipeline tests | passed | 36passed |
| Active complete executed notebook | passed | nbconvert exit0, five genuine comparisons |
| Result labels/metric formulas/thresholds | passed | All three categories separated; nonzero numerical limits + independent force |
| README/plans semantic consistency | passed | Sign/factor/unit/call/default review |
| Final inventory/licensing/cleanup | passed | MIT/LICENSE/NOTICE; private archive; no temporary executed notebook shipped |

No failed or unverified mandatory gate remains. Known scientific limitations are preserved above and in notebook, not converted to claims of pixel-pressure accuracy.

## Bound file SHA256

- `data\raw_data.npz`: `bb0c94c78d25bab9e6d5a589d84f0fd55dc870642274072aa6484ea924339551`
- `evaluation\baseline_reference.npz`: `f9891d9f468b28431aa331824b981a2602109ae8f0c52713087cee0317bb44cc`
- `test\fixtures\upstream_stages.npz`: `6d008ae698c9992173c82b1552b6f57aa30c3f667fbffef650f71f46200e79fa`

## Parent review hardening

Threshold configuration now requires every mandatory key to contain a finite int/float excluding bool; NCC must lie within [-1,1] and error limits must be nonnegative. Overflow/invalid/divide exceptions during score arithmetic produce status error with empty metrics and exit2. Every calculated metric is checked finite. Added 11 malformed/missing/range/type config cases, finite1e308 prediction overflow, and valid tightened force threshold0 proving boundaries are applied. Final full tests36passed15.60s (process16.025s), CLI/Judge0 and active notebook41.996s all rerun. Numeric reconstruction and original oracle remain unchanged.


## Parent acceptance review

Parent independently ran36tests in15.90s, strict inventory/public-array/call-role audit errors[] and foreign-cwd standaloneJudge exit0. Read full README, both plans, complete core derivative/Hermite/virtual-work/closed-form inverse, standalone independent force metric and source provenance. Opened all5 actual scientific figure groups, retaining complete250frame optically measured deflection, signed pressure/bending/inertia, all3 whole/core/hammer experimental force curves and real unmodified-source comparison, exact Hermite physical-resolution fields, and genuinely executed filter/window sensitivities. Original complete280image real-data pipeline ran159s privately; linear-ramp proof identifies optical boundary-index defect, minimally corrected acquisition then unchanged original mechanics rerun12.68s on exact bound inputs. No target-dependent force fit or timing alignment. Fullfield parity dynamic-rangeNRMSE5.01e-17 and independent core-force NRMSE.09114770 pass nonzero numerical/physical thresholds; poor wholefield force.30965253 remains visible. No pixel-pressure truth is invented. Independent quartic bending exposes preserved1% midpoint bias and constant acceleration checks units/signs/normalization.36tests cover all strict threshold/type/finite/overflow/report contracts and actual named threshold use. Canonical active notebook rerun41.996s/5PNG/noerrors. All nativeframes and pressurepatches evaluated; output resolution and manual source ROI are declared. Exact source/data/paper licensing/provenance and corrected-vs-unmodified measurements are explicit, both plans semantically match actual module signatures and caller flow.
