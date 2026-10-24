# Muon scattering field: provenance and validation

Status: all mandatory gates passed; final22-test handoff suite, fully executed notebook, independent geometric qualification and visual review are recorded below. This is a qualified corrected reproduction of the same published POCA geometry. It is NOT an unchanged-source reproduction and NOT a physical-density truth benchmark.

## Scientific scope and acquisition

Selected source https://github.com/MaximeLagrange/muograph, exact remote HEAD `626b86d7bee3808a0ce47a0d9f0dfbd0bab6913a`, verified live by repository and commit API records muograph_repo.json/muograph_commit.json on2026-10-22. Complete original checkout and download archive are private. Source and bundled data are distributed with the repository's AGPL-3.0 licence; its full LICENCE is retained as task/LICENSE. No new third-party data licence is invented. Author attribution appears in the README.

Scientific references: Giammanco etal.,iScience28(2025)112094,DOI10.1016/j.isci.2025.112094; selected archived version arXiv2405.10417v1, *Cosmic rays for imaging cultural heritage objects*. AppendixA describes the effective vertex as the closest approach of actual incoming/outgoing lines and associates it with their geometric scattering angle; Section3 describes angular-width material discrimination. These definitions support the three minimal geometric corrections below. The current author software's unweighted angular-RMS statistic is the selected branch; the paper's normalized POCA count-density figures are retained as an explicitly different count diagnostic, not silently equated with RMS units. Related author paper Lagrange etal.,JAP138(2025)104902,DOI10.1063/5.0273541 is cited for context. Its university PDF endpoint timed out and was NOT falsely claimed as archived. The successfully acquired arXiv paper version and hash are recorded below.

Source observations are `muograph/data/Marble_statue_80x40x40_scattering_bronze_bar/Statue_scale1_scattering.zip`, decompressed privately as muon_statue.csv. This is a GEANT4v10.07/CRY acquisition, not real detector data. Author specs.txt states5million generated muons on a1m² source surface, about8h, and only complete six-detector traversals retained. Every522963 recorded event is bundled, without subsampling, event-quality selection, added noise or event loss. The source specimen includes unknown material contrast; its material labels/shape, source energies and reconstructed vertices are excluded from visible input. No physical voxel truth is available. Public only detector_hits float32(1,522963,6,3) mm, exactly equal to the original Hits.reco_hits convention; no original material label or position field is supplied.

Data discrepancy investigated: source specs describes detectors as 'infinitly thin', but stored z values vary within about1mm below the nominal upper face. The full native XYZ coordinates are preserved; no snapping or synthetic planarization is performed. The original Hits path casts source CSV coordinates to float32, and the same preprocessing is preserved exactly. No extra noise is added. This is distinguished from optional author notebook examples that add1mm smearing,98% efficiency and angular cuts; those configurations are not this bound acquisition.

Deduplication: a cosmic-particle six-track scattering acquisition maps to a3D angular material-contrast volume. Existing magnetic field/current, ultrasound, optical scattering and neutron/diffraction tasks use different observations and unknowns. No prior muon-scattering volume task exists in the159-task catalog.

## Original run, qualification and independent error investigation

Before any cleaned mathematical implementation, `oracle-env/Scripts/python.exe muon_oracle.py` executed the complete unchanged author Hits, Tracking, TrackingMST, Volume and POCA classes on the EXACT full source CSV. CPU only, CUDA_VISIBLE_DEVICES=-1 and OMP/MKL/OpenBLAS threads1, PyTorch threads1. No source import/UI shim was used. Original exit0, full outputs and stage arrays in muon_upstream.npz; process log muon_original_run.log and parameter/environment record muon_original_run.json. Author package dependencies were installed only in the private oracle environment; the cleaned environment was not upgraded or given domain reconstruction packages.

Independent source audit discovered three actual mathematical discrepancies, quantified on every bound event before local implementation:

1. The first SVD right-singular vector represents an undirected fitted line. Source changed only a positive z component, whereas preserving the line requires multiplying the ENTIRE vector by-1. Direct comparison with actual three-hit line geometry found2 incoming tracks with cross-direction mismatch above1e-3.
2. Source angular numerator was sum(abs(u_i*v_i)), whereas the geometric angle between downward vectors uses sum(u_i*v_i). On the source fitted directions,3661 events differ by more than1mrad; maximum difference1.0078092813491821rad. The independent test with u=(.6,0,-.8),v=(-.6,0,-.8) has angle acos(.28), whereas the source returns approximately0.
3. Source 'parallel' mask required ALL three absolute cross-product components to exceed1e-7. A nonparallel test uses norm(cross)>=1e-7. The componentwise test discards24965 additional valid nonparallel tracks. Vertical/skew lines with a zero cross component are explicit counterexamples.

Source-only results are never called physically correct merely because of parity. Parent review authorized restoring the same published line/angle/nonparallel geometry. Exactly three mathematical source lines were corrected in a separate private checkout muograph_qualified, with unmodified original preserved. The complete unified diff is muon_qualified.patch, hashes in muon_qualified_changes.json. No solver, discretization, POCA midpoint, voxel box, clipping rule or data observation was replaced. `muon_qualified_oracle.py` executed that qualified author code on the same exact full CSV BEFORE local extraction; exit0, frozen muon_qualified_upstream.npz and muon_qualified_stages.npz. Independent analytic skew-line cases and12nondegenerate actual hit events were captured using author functions before extraction. Predetermined tolerances are in muon_predeclared_tolerances.json. A later complete qualified original rerun also passed and is bitwise equal for EVERY saved array, not only the image (muon_qualified_repeat.json/log/npz).

## Equation / function / validation traceability

| Required stage | Original / qualified source function | Local function | Validation |
|---|---|---|---|
| Float32 native detector hits | Hits.get_hits_from_df/get_reco_hits_from_gen_hits | preprocessing.load_data | Exact retained arrays; native12-event cases; no added smearing |
| Orthogonal three-hit line | Tracking.get_tracks_points_from_multi_hits | fit_tracks | Frozen actual hit directions/centroids; analytic downward line residual |
| Geometric angle | TrackingMST.compute_dtheta_from_tracks,qualified numerator | scattering_angles | Frozen angles; opposite transverse signs; independent atan2 |
| Nonparallel criterion | POCA.compute_parallel_mask,qualified cross norm | nonparallel_mask | Frozen actual mask; vertical-line counterexamples |
| Common-perpendicular midpoint | POCA.compute_poca_points | closest_approach | Frozen actual/analytic midpoints; independent2×2Gram solve/perpendicular connector |
| Inclusive box | POCA.compute_mask_in_voi | in_volume | Full qualified event count; native coordinates |
| Voxel interval assignment | POCA.assign_voxel_to_pocas | assign_voxels | Full original count array;19distributed frozen POCA/voxel cases; internal/global boundaries |
| Voxel angular statistics | POCA.get_xyz_voxel_pred | aggregate_rms | Full qualified field; empty voxels; count conservation; direct known RMS case |
| Historical source audit | Same source functions,uncorrected mathematics | reconstruct correction switches | Fresh full historical field matches unchanged author output exactly |
| Metrics | Benchmark cosine/range equations | Judge.score and visualization.metrics | Independently known two-element case,zero prediction |

Source voxel assignment uses inclusive bounds on both sides, then repeated advanced assignment. At internal shared boundaries the observed last assignment selects the larger cell index, despite a misleading 'first' comment in the original function. Local sorted-edge lookup preserves that measured rule; it does not silently choose a different convention. No target-derived geometric mask is introduced.

Prespecified stage tolerances are recorded in the approach plan and private JSON. Full qualified/historical field comparisons are exact in this runtime. Full event-count array is exact. Small-case angle/POCA tests and selected full-data cases pass the stated float32 tolerances; no threshold was relaxed. The native observation table is reused, so no generator is shipped.

## Numerical backend and environment

Actual Windows runtime: research-runtime/Scripts/python.exe,Python3.11.11,NumPy2.4.6,PyTorch2.9.1+cu128,Matplotlib3.11.1; all reconstructions validated on CPU with one thread. muon_environment.txt records platform and complete torch numerical-backend build settings. Oracle dependencies (pandas,uproot,awkward,seaborn,fastprogress and related reading/plotting helpers) are private only. Cleaned runtime uses only NumPy,PyTorch and Matplotlib; all task-specific geometry and aggregation are local executable functions.

One repeated Windows standalone run exposed a PyTorch/MKL SGESDD batch-workspace assertion (argument12) on finite data at original200000-event execution chunks; this failure is preserved in muon_core_tests.log. Nine independent full incoming-track experiments (3each at50000,100000,200000) all subsequently passed and were bitwise identical to the source, recorded in muon_svd_backend_audit.json. The local default uses50000-event execution chunks to reduce batch workspace. This preserves every event and exact per-line mathematics, with no SVD-library substitution or data change. Three subsequent independent CLI processes passed and exactly reproduced the qualified field (muon_cli_repeats.json). The final notebook and test suite use this actual default. No claim is made that the upstream backend assertion can never recur on another platform.

## Scientific results, metrics and limitations

Hidden evaluation/baseline_reference.npz/reconstruction is the independently run qualified author float32(1,36,80,40) angular-RMS field, NOT physical truth. notebooks/reference_outputs/unchanged_original.npz stores original scattering_rms and counts for explicit source-error comparisons. Test fixtures contain the captured qualified native/analytic stages, distributed full POCA indices/coordinates and complete original count array; they are consumed by comparing_test/test_upstream.py and unit_test/test_geometry.py. No unused complete source checkout, PDF, cache-preview notebook or archive is shipped.

NCC is uncentered cosine similarity and NRMSE is RMS error divided by the full reference dynamic range. Both use every voxel with fixed axes and no fitted gain, shift, crop, smoothing or alignment. Boundaries NCC>=.999999,NRMSE<=5e-6 are finite numerical reproduction tolerances, not a mechanically zero self-error threshold. Repeated complete qualified-original and local runs are exact; eight-decimal-radian serialization sensitivity is independently measured below. Poor all-zero and extreme constant fields fail, malformed arrays returnerror2. The Judge independently loads NPZ with allow_pickle=False, validates exact shape/float32/finite values, and tests missing keys/files,wrong dtype/shape/complex/NaN/Inf,hidden asset/configuration failures and report-write failures. Foreign-working-directory CLI paths are tested.

The fresh standard field retains220601 in-volume interactions. Original source retains217513; orientation-only correction217517; geometric-angle correction217517; cross-norm correction220601. The original field differs from the qualified standard by NCC.9882097842218399 and dynamic-rangeNRMSE.017784747674236177. This intentional physical correction is far outside the extraction tolerance and is explicitly visualized, not hidden by relaxed evaluation. The source's field is relative angular contrast with sparsity, single-interaction and unknown-momentum limitations, not a density measurement.

The active notebook covers problem,data,physics,method,results,conclusion and computes all full-data correction ablations. Seven actual scientific figures cover observed detector geometry/depth,original/fresh/residual,symmetric correction audit,orthogonal sections,POCA-count versus angular statistics,upper-tail sensitivity,and independent connector/angle precision. All7 PNG panels were opened and visually inspected. Crowded audit titles were shortened and replaced by a shared common colorbar; the log colorbar correctly labels counts on a logarithmic scale. Signed residual fields use a symmetric diverging RdBu_r scale with neutral zero. Every fresh method/ablation is actually calculated; archived arrays are original comparisons only. Private PNG/SVG/PDF and numeric sources are muon_figures and muon_experiments.npz/json.

The canonical notebook was executed through nbconvert with timeout240s and kernel imaging101-research,whose JSON selects the actual research-runtime executable. Its fresh full reconstruction takes around3s;240s accommodates all full ablations,figures and startup. Actual kernel executable/NumPy/Torch versions are printed and saved in the experiment record. Nonfatal Windows ZMQ selector-thread and local-kernel TCP warnings are retained in muon_nbconvert_final.log. All code cells executed with zero error outputs. The temporary executed copy was transferred to the canonical notebook then removed from output/.

README,approach/design plans and executable signatures/defaults were semantically reviewed. Signed directions,angle convention,unit conversion,grid axes,200000-oracle versus50000-local execution chunks,clipping quantiles,reference identity and absence of physical ground truth agree. The README exposes only complete acquisition metadata,public input and required output. No GitHub URLs,privileged energies/material labels,source vertices,hidden object coordinates or target-derived calibration appear in the solver-visible allowlist. No shared environment changes or cross-task imports are required.

## Private archive index

Maintainer-only archive ../imaging-101-maintainer/research_batch6_20261022/groups/physics_materials/ contains complete muograph original and qualified checkouts;muograph.zip;remote repository/commit JSON;source specs;muon_statue.csv;paper PDF/record;original and qualified capture scripts/logs/arrays;qualified3-line patch/hashes;source audit;predeclared tolerances;backend audit;CLI repeats;original complete rerun;notebook export generator;PNG/SVG/PDF;numeric experiment NPZ/JSON;final task hashes. Source/capture scripts are not imported by any task runtime,test,Judge or notebook.

Archived paper: {"url": "https://arxiv.org/pdf/2405.10417v1", "sha256": "f15d72d3038739fc30696722e4dbdc805d5e5c4172f1b160dd6f51f129981f48", "bytes": 5625891, "version": "arXiv2405.10417v1", "published_doi": "10.1016/j.isci.2025.112094"}

Original full runtime seconds: 449.1811306000018; qualified full runtime seconds: 463.4466281000059; repeated qualified runtime seconds: 438.23181659998954

Source CSV SHA256: c65130a1e9a5f4e37faf751fdef2b4d5f5fcf7cd9926bb721316612289474129; packaged raw NPZ SHA256: a4d65cefe9fc02a2353d5f21c0d37bad04f49978c75be7504c0d9730ccf90ab4

Qualified source file hashes: {"muograph/tracking/tracking.py": {"original_sha256": "d62b4290453682b5b4a2d553461d47e838ba2f44f149a9ed81f8dec9c0842670", "qualified_sha256": "cee7758cb12fa8a86276accd860d42f65f0757e0e9a4f39d0235c5c1a14810c1"}, "muograph/reconstruction/poca.py": {"original_sha256": "61ada08955d99a1348a6d01ceab2c8a2ba648774c5e19d20030ee70889601c8c", "qualified_sha256": "5d30941281a528dcc4359bc39f6155c3179e07982e8ab4a6d250140bdc260df9"}}

Eight-decimal-radian serialization sensitivity: {"ncc": 0.9999999999999953, "nrmse": 1.1005536636359077e-08}

Full fresh notebook measurements: {"fresh_seconds": 2.7989331999997376, "qualified_parity": {"ncc": 0.9999999999999999, "nrmse": 0.0}, "unchanged_parity": {"ncc": 1.0, "nrmse": 0.0}, "state_statistics": [{"method": "Unchanged source", "seconds": 2.8851570000115316, "retained": 217513, "ncc": 0.9882097842218399, "nrmse": 0.017784747674236177}, {"method": "Whole-vector orientation", "seconds": 2.8904853999993065, "retained": 217517, "ncc": 0.9882097681315782, "nrmse": 0.017784739780011456}, {"method": "Geometric angle", "seconds": 2.9210567999980412, "retained": 217517, "ncc": 0.9999725920167648, "nrmse": 0.000811849522053233}, {"method": "Geometric nonparallel norm", "seconds": 2.7989331999997376, "retained": 220601, "ncc": 0.9999999999999999, "nrmse": 0.0}], "sensitivity": [{"ncc": 0.9771494387689375, "nrmse": 0.02626549800538647}, {"ncc": 0.9999999999999999, "nrmse": 0.0}, {"ncc": 0.9977852900154606, "nrmse": 0.007770656420873896}], "physics": {"independent_lines_checked": 858, "midpoint_max_mm": 0.005728335569937775, "connector_dot_u_max_mm": 6.385285689081144e-09, "connector_dot_v_max_mm": 6.385300430761243e-09, "acos_vs_atan_max_rad": 3.893007511343409e-05}, "judge": {"status": "pass", "passed": true, "metrics": {"ncc": 0.9999999999999999, "nrmse": 0.0}, "thresholds": {"ncc": {"comparison": ">=", "boundary": 0.999999}, "nrmse": {"comparison": "<=", "boundary": 5e-06}}, "errors": []}, "actual_kernel": "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research-runtime\\Scripts\\python.exe", "numpy": "2.4.6", "torch": "2.9.1+cu128"}

## Final mandatory-gate ledger

Final command `python -m pytest test/ -v --basetemp=<private group archive>/muon-pytest-handoff` passed22 tests in11.68s (muon_final_tests.log), after the full-count/distributed POCA fixtures were added. `python evaluation/judge.py --prediction output/reconstruction.npz --report output/judge_report.json` passed the fresh declared output. All commands used the actual shared runtime on CPU with one OMP/MKL/OpenBLAS/PyTorch thread; a private pytest directory avoids an existing default Windows Temp access error. No shared installs or upgrades occurred. The actual canonical notebook includes8 executed code cells, zero errors and7 inspected scientific figures. Temporary executed notebook is absent.

| Mandatory gate | Status | Evidence |
|---|---|---|
| Paper version,exact live revision,data hashes,stage mapping | passed | Remote records,arXiv PDF/hash,source hashes,traceability table |
| Unchanged original exact-observation run | passed | Complete unchanged muon_oracle.py run,449.18s,exit0 |
| Investigated source errors and qualified same-method oracle | passed | Independent geometric audit,3-line diff,full qualified original463.45s,complete repeated run exact |
| Frozen stage parity and independent mathematics | passed | Native/analytic fixtures,full field/counts/selected POCA tests,independent2×2Gram geometry |
| Constants,units,precision,coordinates | passed | Native full XYZ preserved,source z discrepancy disclosed,voxel conventions tested |
| Standard CLI,NPZ output and independent Judge | passed | Three exact CLI repeats,foreign-cwd Judge,output report |
| Poor/malformed/evaluator/report-write rejection | passed | Unit contract tests and independent known metrics |
| Visible sufficiency and no privileged input/URL | passed | Raw/README/environment allowlist scan,source label/energy/vertex exclusion |
| Shared CPU runtime and local explicit mathematics | passed | NumPy/PyTorch/Matplotlib primitives only,source/call-role audit,no upstream imports |
| All tests | passed |22 passed,muon_final_tests.log |
| Active six-chapter notebook and comparisons | passed | Canonical executed notebook,actual full corrections/sensitivities,7 inspected figures |
| Provenance/metric/threshold consistency | passed | Separate original/qualified/fresh identity,repeat/serialization evidence,no relaxed threshold |
| README/plans/source semantic and signature consistency | passed | Equation/unit/default/caller-flow review above |
| Inventory,archive,notices,cleanup | passed | muon_final_inventory.json;AGPL licence;no temporary notebook/PDF/archive in task |

This task explicitly documents a qualified corrected published-method reproduction. Mandatory original execution is preserved even though it contains investigated mathematical errors; fresh standard parity is evaluated against the independently run qualified source and independent geometry, not against a known incorrect original formula. Historical source parity is separately exact and actively visualized.
