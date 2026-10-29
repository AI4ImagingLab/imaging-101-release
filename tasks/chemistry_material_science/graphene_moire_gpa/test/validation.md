# Provenance and validation

## Bound observation, source and licence

Real AC-LEEM image of supported twisted bilayer graphene, de Jong et al., Nature Communications13,70(2022), journal article DOI10.1038/s41467-021-27646-1, data DOI10.4121/16843510. Dataset licence CC BY4.0 was verified through the current repository API metadata. The source is a stitched, recorded instrument image, not a simulation or rendered paper figure. Selected file is `20200715_154404_0.66um_495.9_sweep-STAGE_X-STAGE_Y_highres_highE_stitch_v10_2020-11-20_1843_sobel_5_bw_200.tif`,57523011bytes, SHA2568ad63a14de35a49c8a5ee64cd9872e84f69618c685b1058607500a363d1f91c7, native uint16 shape5383x5343. Exact crop rows3000:3512,columns3200:3712 is preserved without resampling. Calibration0.88nm/pixel is verbatim from author Figure_dislocation.py for this exact image; high-energy stacking setting37eV is described in paper. The filename's495.9 is instrument acquisition context, not substituted for calibrated landing energy.

Public NPZ SHA2562a20d891c78bc14b5467df8d3eae20ab5795ebbaf9bcd348d7651ed0838e95c5; frozen fixture SHA2567c9555364052d58bd7a83448ee8e0716fbde8d675257d8ae726e3f083c6dc25b. `reference_vectors` are original extract_primary_ks image peaks, period search10–100pixels: observation-derived reference waves only. No independent unstrained region or hidden target is used. Displacement is relative moire inverse deformation, never relabelled absolute atomic strain.

Original algorithm https://github.com/TAdeJong/pyGPA, remotely verified commitc16cb1ed53722f74907e69125f74da5ab45c138c; author figure code https://github.com/TAdeJong/figures-imaging-moire-deformations, remotely verified commitfec3f675a143529775a4166bab565185e3beb323. Exact code archives and licence are private; algorithm licence and data attribution are retained in NOTICE. Paper version is2022 journal article; original published analysis roles and scope were read from author scripts. No claim of remote freshness beyond2026-10-20 query.

## Archive index and original execution

All private paths below are relative to maintainer archive `research_batch2_20261020/material_mechanics/`, outside final task: pyGPA.zip + expanded commit; graphene_figures.zip + expanded author figure source; graphene_highres.tif; graphene_data_metadata.json with CC BY4.0/download URLs; explore_gpa.py and capture_graphene.py; graphene_oracle.npz and graphene_oracle.log; oracle-env/; graphene_executed.ipynb; private PNG/SVG/PDF exports and figure numeric trace. Original scripts/logs/checkouts are never imported by runtime or tests.

Command `oracle-env/Scripts/python.exe capture_graphene.py` completed exit0 **before local implementation**,7.444102300098166s CPU one thread. It imports unmodified original GPA functions, extracts visible carriers, runs extract_displacement_field(...,return_gs=True,wfr_func=wfr2), and freezes512-square image, three carriers, all lock-in fields/selected-carrier maps and inverse displacement. Parameters are original defaults sigma33,kw=mean(carrier_norm)/2.5,kstep=kw/3,10PCG steps. No simulation seed is used.

Environment Windows x64, Python3.11.11, NumPy2.4.6,SciPy1.17.1,Numba0.68.0/llvmlite0.50.0,pyGPA exact source, moisan2011 exact archived source, latticegen0.0.5 and private dependency set. Environment freeze archived separately. CPU NUMBA_NUM_THREADS=OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=1; no GPU. Shared cleaned runtime uses only NumPy/SciPy/Matplotlib and has no pyGPA domain library.

## Mathematical discrepancies investigated

Original precomp_Poissonscaling exchanges N/M denominators on rectangles. Independent Neumann eigenmode derivation gives row cos(pi*i/N),column cos(pi*j/M). This selected512-square case is unaffected, and local implementation rejects rectangles rather than silently extending the erroneous convention. The analytic affine-field test independently validates the actual selected discretization. No corrected rectangular experiment is claimed as unchanged reproduction.

Original wfr2_grad prints a factor-of2pi warning and is not selected. The actual bound method uses wfr2 followed by reconstruct_u_inv_from_phases, which forms K=2pi*k explicitly, wraps finite differences and applies weighted least squares. The local port preserves that exact caller role. Weighted projection uses source leading-pixel weights and phase integration rewraps Cartesian gradients as source does; these are documented conventions, not silently changed.

## Equation-stage traceability

| Stage | Original function | Local function | Independent validation |
| --- | --- | --- | --- |
| Visible-image carriers | extract_primary_ks | supplied public observation-derived vectors | exact frozen values; no hidden calibration |
| Gaussian spatial demodulation | GPA/optGPA | physics_model.lockin | analytic cosine yields0.5complex DC |
| Adaptive strongest carrier and phase compensation | wfr2 | adaptive_lockin | frozen complex fields and exact selected vectors |
| Weighted phase-gradient inversion | myweighed_lstsq/reconstruct_u_inv_from_phases | weighted_projection | known3x2 analytic linear inverse and original output |
| Weighted Neumann normal operator | phase_unwrap_prediff/applyQ | divergence/integrate_gradient | affine displacement integration and zero-gradient gauge |
| Ten-step DCT-preconditioned CG | phase_unwrap_prediff | integrate_gradient | original full pixel-displacement fixture |
| Spatial unit conversion | author NMPERPIXEL0.88 | preprocessing.NM_PER_PIXEL and reconstruct | declared constant and exact scaling |
| Metrics and strict NPZ interface | independently known arrays | Judge/visualization.agreement | contract cases and hand-computable metrics |

Tolerances: lock-in rtol1e-12/atol1e-10 counts, entire physical displacement rtol1e-10/atol1e-10nm; selected carriers require exact equality. Fresh maximum difference2.1316282072803006e-14nm, dynamic-range NRMSE5.336739591742845e-17. Explicit2x2 normal equations match original per-pixel SVD on the bound conditioning. No threshold relaxation concealed a failed stage.

## Hidden references and measured results

`evaluation/baseline_reference.npz` and notebook reference upstream.npz key reconstruction float64(1,2,512,512)nm come exclusively from original output_u multiplied by0.88. Actual displacement truth does not exist. `test/fixtures/upstream.npz` additionally contains original pixel-unit output_u, complex lock-in and adaptive selected vectors; each has documented comparison consumers. Notebook fresh_measurements.npz labels actively computed outputs and alternate settings independently. No historical paper numerical result is used as this run's evidence.

NCC is uncentred cosine; NRMSE is RMS difference/dynamic range, no crop/scale/alignment. Nonzero tolerance1e-6 is more than float64 rerun discrepancy yet rejects zero/ablation outputs. Reference agreement proves extraction fidelity, not physical accuracy. Adaptation/window sensitivity is reported directly from fresh measurements in the notebook.

## Gate ledger

| Mandatory gate | Status | Evidence |
| --- | --- | --- |
| Source/paper/data/license/equation mapping | passed | above and NOTICE |
| Original exact bound run before extraction | passed | original log and independently frozen stages |
| Cleaned local mathematics/no domain imports | passed | explicit FFT,weighted2x2 inversion,DCT-PCG |
| Preserved constants/precision/coordinates | passed | raw uint16 unchanged,exact carriers,0.88nm conversion |
| Independent parity/physics | passed | test/comparing_test/test_numerics.py |
| Strict Judge malformed/poor/evaluator/report failure | passed | unit_test/test_contract.py |
| Public contract/no truth or GitHub leakage | passed | README and visible NPZ audit |
| Tests | passed |16passed (including added Cartesian-wrap audit) via shared Python,-m pytest test/ -q,private basetemp |
| Active complete notebook | passed | nbconvert exit0; fresh3.0741338999941945s, research kernel180s timeout |
| Both plans match actual source/math/flow | passed | source signature,weight/sign/2pi/unit/caller review |
| Final inventory/archive/licence | passed | exact template and NOTICE |

Commands: shared Python -m src.main; evaluation/judge.py --prediction output/reconstruction.npz --report output/judge_report.json; -m pytest test/ -q --basetemp=<private unique directory>; -m jupyter nbconvert --to notebook --execute notebooks/graphene_moire_gpa.ipynb --ExecutePreprocessor.kernel_name=research --ExecutePreprocessor.timeout=180 --output graphene_executed.ipynb --output-dir=<private archive>. JUPYTER_PATH points to explicit shared-Python research kernel, avoiding default Anaconda kernel.

Independent Cartesian-wrap audit: gx max10.404523375153985pixel,869/523264 entries>pi; gy max10.257946672066998pixel,679/523264>pi. All occur in downweighted66pixel edge; interior gx max0.620618059858323 and gy0.4983523090338092pixel have no wrapping. Private no-rewrap integration yields max6.16786621776555e-11nm change, NRMSE1.4099902782765644e-13, interior80pixel max1.680433570072637e-12nm. This operation is not dimensionally valid in general; exact source reproduction is justified only because edge contributions are squared downweight1e-12 and demonstrated negligible for this selected observation. New independent RHS-perturbation check is in comparing tests.

Visual QA passed: all three final figures rendered, inspected for calibrated axes, readable signed scales, Fourier reference annotations, and numerical difference scales; no clipping or placeholder panels. Final PNG/SVG/PDF+numeric-trace manifest is private at final_figures/graphene_moire_gpa/manifest.json under the indexed maintainer archive.

Final coordinate QA: imshow extent uses pixel/cell boundaries (sample endpoints plus/minus half spacing), with displayed limits set to declared sample-centre endpoints; Fourier-bin extents likewise use half-bin boundaries. Canonical notebook and final exports are reexecuted after this display-only correction.


## Independent parent acceptance

Parent reviewed the implemented equations, numerical conventions, shipped signatures/plans, independent upstream evidence, physics checks and all 3 actual scientific figure groups. Parent strict inventory/public-input/import-role audit returned no errors and independently executed the standalone Judge from a foreign working directory (exit 0). Parent verified 16 distinct passing test cases; evidence is archived in the second-batch parent audit/test records. Effective moire displacement, not absolute atomic strain; square-grid scope and negligible weak-edge gradient rewrap quantified.
# Public mathematical contract revision 2 — 2026-10-29

The benchmark acceptance semantics changed from agreement with the selected ten-PCG-step reference to the explicitly public derived-observation weighted-gradient inverse. This is not a claim of unchanged reproduction criteria or physical displacement truth. Original reference arrays and observations remain byte-for-byte untouched. Version 1 metrics are privately preserved at `benchmark_readiness_20261029/graphene_metrics_v1.json` in the maintainer archive.

The independent evaluator directly constructs the periodic Fourier Gaussian window, carrier selection, wrapped differences, local weighted normal systems and integration normal-equation residual. It imports no solver. Relative stationarity boundary is 0.001 and zero-mean gauge boundary 1e-8 nm; these are numerical accuracy budgets for the stated least-squares problem rather than image-similarity tolerances. The existing independently validated ten-step result has maximum relative stationarity 0.0002294369237466319. A more converged 1000-budget PCG control reaches 9.187447443762273e-10 in approximately 6.60 s including independent scoring. Zero displacement gives 1 and reversed displacement gives 1.999995723415685, both rejected. An independent integrable polynomial-field unit case verifies exact normal-equation satisfaction, zero/reversed rejection and the physical 0.88 conversion.

Private evidence: `benchmark_readiness_20261029/graphene_gradient_controls.json`, `graphene_controls.log`, `graphene_precise_v2_report.json`, `graphene_pytest_v2.log`. The precise control was generated solely for validation, is not bundled public data, and is not substituted for the existing solver output. Standalone CLI accepts it. `python -m pytest test/ -q --basetemp=<private-directory>`: **17 passed in 14.10 s**, including upstream parity/physics and revised evaluator contracts. An initial test attempt encountered a permission error in the global Windows pytest temporary directory; the isolated private temp directory resolved it. No packages were installed.

