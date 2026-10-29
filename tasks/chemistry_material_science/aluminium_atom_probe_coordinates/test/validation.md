# Atom-probe provenance and validation

Status: all mandatory gates passed; exact original and local numerical paths, active notebook, final inventory and review evidence follow. No physical atom-position ground truth is available.

## Scientific scope, source and acquisition

Selected published/software method: PyCCAPTGeiser atom-coordinate reconstruction, with Bas analytic projection retained as an active comparison. Published source paper Monajem etal.,MRT88(2025)3199–3210,DOI10.1002/jemt.70011. Version-of-recordPDF privately archived apt_PyCCAPT2025_version_of_record.pdf from https://d-nb.info/1389973727/34,SHA256 `cf5d9077414cc5a93fefde0b5a55691dc05b6441aa6f7abd7384b3fd5611e6e0`,2504124bytes. Full source repository https://github.com/mehrpad/pyccapt remotelyverifiedHEAD `69512f6fda75ed12519a36cff6e15437e072800f` on2026-10-22; no claim is based only on a local tracking revision. Original reconstruction.pySHA256 `499f50db9e590afd2183199a55f0646e48f3af3849d7fda1642a35bfc7a40912`.

Real data source: Monajem,Atom Probe Tomoghraphy Pure Aluminium Dataset,v2,DOI10.5281/zenodo.14673955,CC-BY4.0. Source calibratedHDF5 `2382_Jan-10-2025_15-12_NiC9_Al_calibrated.h5`,size1786414500bytes,repository-declared checksum `md5:bd9305e29999c4936d92ace42616e05f`. Whole source HDF5 was NOT downloaded or independently whole-file hashed: exact source version metadata and checksum are recorded, while selected numerical observations have an independent SHA256. HTTP206 range-reading accessed18874368bytes. The source contains16540789processed events; the bound acquisition is the initial consecutive200000-event prefix, preserving all events and full detector field in that window, rather than species/spatial/quality filtering or lower-resolution synthesis. Reader transfers are indexed in apt_data_capture.log. Event order starts at the original first event, so no discarded earlier-ion removal history is assumed.

Retained source columns: x_det(cm),y_det(cm),high_voltage(V). Source stored x(nm),y(nm),z(nm),mass/charge and timing columns are explicitly EXCLUDED from solver input and NOT used to tune calibration. No independent truth is assigned to these old coordinates. Selected-input archive apt_bound_observations.npzSHA256 `1c6692ba7858608c4bbc0cb471a703cd369ba3a1ae2027f4deae482139fdb507`; packaged raw_data.npzSHA256 `c22a7a3451796459cb04262140af622daff0e509c1074a99664ca17f53f507ad`. Exact selected-array equality is checked. Public arrays preserve all native float64cm/V values with zero conversion error.

Instrument/model assumptions were fixed BEFORE original capture: L110mm,kf4,efficiency.7,icf1.4 from author interface defaults in tutorials_helpers/helper_3d_reconstruction.py and helper_data_loader.py; F19V/nm,density60.2atoms/nm³ from original files/field_density_table.h5Al entry. No source reconstructed coordinates are consulted in fixing these values. The parameters are presented as fixed author conventions and their uncertainty is studied, not passed off as independent metrological truth. The observed maximumdetectorradius defines the explicit author analysis-area convention. The acquisition subset cannot establish the full specimen shape.

Deduplication: this is field-evaporation detector-hit inverse geometry to ordered3Datomic positions. Existing STEM density tomography, EBSD orientation imaging and AFM surface-height inversion use different unknowns/acquisition operators. No existing atom-probe inversion task exists.

## Original run before local extraction

Privatecommand oracle-env/Scripts/python.exe apt_oracle.py,CPU1OMP/OpenBLAS/MKLthread. Python3.11.11,NumPy2.4.6. Originalunchangedfull reconstruction.py imported viaimportlib,then original atom_probe_recons_from_detector_Geiser_et_al and atom_probe_recons_Bas_et_al called on ALL200000bound events. Exit0; Geiseractualruntime0.01467889999912586s,Bas0.012232499997480772s. Originaloutputs and separately run nondegenerate12event constructorcases captured inapt_upstream.npz before local implementation. Warnings:none. Fullparameters,sourcehash,processversions and importcompatibilityrecord inapt_original_run.json/log.

Importcompatibility is explicitly limited to unused Plotly/GUI helpers. Every placeholderfunction raises if called; the imported reconstruction numerical functions and originalcart2pol/pol2cart remain unchanged and execute their originalNumPy code. apt_oracle.py records the complete placeholdermodule/symbollist. This is a selected numerical-function execution with disclosed optionalUI import compatibility,not a claim that the complete interactivePyCCAPT GUI was run. Cleanedcode doesnot import these helpers or upstreamdomaincode. Sharedruntime uses onlyNumPyandMatplotlib,with no upstreampackages installed or sharedupgrades.

## Traceability and independent checks

| Published/reference stage | Originalfile/function | Localfunction | Evidence |
|---|---|---|---|
| Field-dependent specimenradius | reconstruction.py/Geiser | radius_evolution | originalsmall/fullpositions,unitcheck |
| Detector polar and imagecompression | cart2pol,Geiser | launch_geometry | nondegenerateoriginalsmallcase,independentinverseprojection |
| Lateral cap inverse | Geiser | reconstruct | originalfull200000eventcoordinateparity |
| Atomicvolume-depth accumulation | Geiser | depth_increments,reconstruct | sum(area*dz)=N/(rho*eta),positivefirstincrement |
| AlternativeSIprojection | Bas | bas_reconstruct | original12eventcase;fullfreshnotebookcomparison |
| Independentdetectorforward | correctedangleidentity | project_to_detector | analyticroundtrip;notcopiedinverseoperations |
| Metricconventions | benchmarkequations | Judge.score,visualization.metrics | independent[1,0]vs[1,2]knowncase |

Prespecifiedfloat64parityrtol2e-13/atol2e-13nm; independentprojectionrtol1e-13/atol1e-14cm;volumeconservationrtol1e-13. Frozenupstreamfixturecontains12real,nonuniform,off-axisdetectorhits andvaryingions,not a uniform or cleaned-self-generated case. Fullreferenceagreement is independentlytested. Test-first missing-modulefailure archivedapt_red_tests.log; finalpassingtests cannot be inferred fromanimportalone.

The source currentBasfunction documents an older detectorcm->m error that inflatedarea/depth. Checkedrevision uses1e-2; cleanedcode preserves it and validatesunits. We do not claim unchangedreproductionofthatoldererroneousrevision. TheGeiserbranch maintains detectorradii inmm and specimenradii innm,with mixed-unitmagnificationexplicitlytracked; convertingL to nm without changingAD would be anerror.

## Assets, licensing and archive consumers

Maintainerarchive ../imaging-101-maintainer/research_batch6_20261022/groups/physics_materials/ contains pyccapt.zip,completecheckout,livecommit/repositorymetadata,apt_paper_record.json/PDF,pyccapt_data_record.json,apt_bound_data_record.json,apt_data_capture.log,capture_apt_data.py,apt_oracle.py,apt_original_run.json/log,apt_upstream.npz and finalnotebook/plotnumericalexperiments. LicensecodeGPL3 ispreserved task/LICENSE; source dataccby4attribution inREADME. NoPDF,upstreamcheckout,downloadarchiveorcapturescript isinthefinaltask.

Hidden evaluation/baseline_reference.npz/reconstruction holds the independentlyexecutedoriginalGeiser(N,3)coordinates. It isNOTphysicaltruth. notebooks/reference_outputs/upstream_bas.npz/positions_nm holds originalBascoordinates,used onlyforthe explicitlylabelled originalBasnotebookcomparison. test/fixtures/upstream.npz holds input_small_x,input_small_y,input_small_voltage,output_small_geiser,output_small_bas,consumed bytest_upstream.py. Freshmeasurements mustbecomputed locally; nohistoricalpapermetric isrelabelled as thisrun.

FixedmetricboundariesNCC>=.999999999,NRMSE<=1e-10 are nonzero numericaltolerancesfora methodreference,not mechanicallyderivedfromzero baselineerror. Everyeventcoordinateisscoredwithoutalignment. CalibrationaccuracyrequiresadditionalindependentmeasurementsandisnotimpliedbyJudgeagreement. Finalrepeat/precisionandJudgequalityrejectionmeasurementsfollow.

## Final execution, semantic review and mandatory gates

The unchanged original numerical functions were independently rerun on the exact full observation prefix: every saved stage/final array is bitwise identical (apt_original_repeat.json/log). Two independent cleaned runs are also bitwise equal. The fresh cleaned result matches the original Geiser array exactly. Rounding coordinates to eight decimal places in nm gives the independently measured serialization sensitivity recorded below. This finite numerical precision experiment supports a nonzero NRMSE tolerance; the threshold is not a zero self-error rule, a statement of atomic-position accuracy or permission for alignment.

The full shared-runtime `python -m pytest test/ -v --basetemp=<private archive>/apt-pytest` run passed17 tests (apt_core_tests.log). The private test temporary directory avoids an existing Windows default-Temp access problem. Original small-case Geiser and Bas, full Geiser output, analytical detector reprojection, ion-volume conservation, physical unit signs, known independent metric cases, all malformed Judge inputs, poor predictions, evaluator and report-write failures, and the standard CLI are covered. `python -m src.main` followed by the standalone Judge passed. No upstream computational-imaging dependency is installed in the cleaned environment.

The six-chapter notebook was fully executed using `python -m jupyter nbconvert --execute --to notebook ... --ExecutePreprocessor.timeout=180 --ExecutePreprocessor.kernel_name=imaging101-research`. Actual kernel executable is `D:/yan1/agent/AutoSkill/imaging-101-maintainer/research-runtime/Scripts/python.exe`, Python3.11.11, NumPy2.4.6. The reconstruction itself takes around0.02s;180s covers startup, all full-data sensitivity runs and figures. Nonfatal Windows ZMQ selector-thread and local-kernel TCP notices are archived in apt_nbconvert_final.log; every code cell executed without errors. Executed outputs are retained only in the canonical notebook; the temporary output notebook was removed.

All six PNG scientific figures were opened and inspected: full detector/voltage; original/fresh3D geometry and all-event error; orthogonal all-event density histograms; actual Bas/Geiser model comparisons; six full-data efficiency/compression reconstructions; and independent detector/volume residuals. Axes, units, labels, count colorbars, legends and common original/fresh views are readable. Signed residual colors use symmetric RdBu_r with neutral zero. For signed depth-model difference, sign is represented by the vertical axis; hexagon color denotes positive ion counts with a labeled sequential colorbar. Point-view decimation is explicitly display-only; every scientific metric and numerical reconstruction uses all200000 events. PNG/SVG/PDF exports are private in apt_figures; complete underlying numeric source is apt_experiments.npz and actual fresh measurement provenance is apt_experiments.json. No cached result was called fresh.

README model units, source constants, exact signatures/defaults and both plans were semantically reviewed against executable code. The Geiser radius, corrected angle, mixed-unit magnification, cap sign, first-ion cumulative depth, Bas SI conversion, finite-domain validation, metric normalization and unaligned event order agree. Actual caller flow has no optimizer or synthetic generator and no hidden-source imports. The source licence and data attribution are preserved. Solver-visible README/raw/requirements suffice independently; no hidden atomic coordinates, privileged source coordinates, mass labels, target-derived calibration or GitHub URLs appear there.

| Mandatory gate | Status | Evidence |
|---|---|---|
| Paper, exact live revision, hashes, mapping | passed | Records, archived version of record, traceability above |
| Original and fresh exact-observation runs | passed | Original and repeat logs; full-array parity; active notebook |
| Independent parity and physics | passed | Frozen12-event cases; full Geiser; projection and volume identities |
| Constants, units, coordinates, precision | passed | Source default table and unit review; serialization experiment |
| NPZ CLI and standalone Judge | passed | output/reconstruction.npz and judge_report.json |
| Judge malformed/quality/evaluator failures | passed | Unit contract tests; known metric cases |
| Public sufficiency and no leakage | passed | Visible allowlist and archive column separation reviewed |
| Minimal dependencies and explicit local math | passed | NumPy/Matplotlib only; CPU execution; symbol/call-role audit |
| Complete tests | passed |17 passed in apt_core_tests.log |
| Active notebook and scientific comparisons | passed | Canonical executed notebook,6 inspected figures |
| Result provenance and nonzero thresholds | passed | Original/fresh separation; repeated runs and precision record |
| README/plans/source consistency | passed | Semantic/signature review above |
| Inventory, archive and notices | passed | apt_final_inventory.json; source GPL3 and data CC-BY4 |

Existing real observations are reused, so no generate_data.py is shipped. No physical atom-coordinate truth is invented. Every reference/fixture has the documented notebook or test consumer above. No PDF, repository checkout, archive or temporary notebook is inside the final task.

Serialization experiment: {"ncc": 0.9999999999999998, "nrmse": 7.774576404901805e-11}; fresh runtime and independent physics values: {"parity": {"Geiser fresh vs original": {"ncc": 1.0000000000000002, "nrmse": 0.0}, "Bas fresh vs original": {"ncc": 1.0, "nrmse": 0.0}}, "Geiser_seconds": 0.020244500003173016, "Bas_seconds": 0.010956800004350953, "reprojection_max_cm": 2.6645352591003757e-15, "volume_nm3": 4746.084480303749, "expected_volume_nm3": 4746.084480303749, "judge": {"status": "pass", "passed": true, "metrics": {"ncc": 1.0000000000000002, "nrmse": 0.0}, "thresholds": {"ncc": {"comparison": ">=", "boundary": 0.999999999}, "nrmse": {"comparison": "<=", "boundary": 1e-10}}, "errors": []}, "actual_kernel": "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research-runtime\\Scripts\\python.exe", "numpy": "2.4.6"}
