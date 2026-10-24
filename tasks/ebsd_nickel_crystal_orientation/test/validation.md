# EBSD provenance and validation

Status: mandatory cleaning gates passed; ready for independent parent audit. This is a real-data orientation inverse with a qualified standard-method reference, not measured orientation truth.

## Frozen scope, paper, source and licenses

Paper: Zachary T. Varley, Gregory S. Rohrer and Marc De Graef, **Accelerating dictionary indexing of electron backscatter diffraction patterns with PCA and quantization**, Scientific Reports16,4382, version of record **2026-01-04**, DOI https://doi.org/10.1038/s41598-025-34391-8. The DOI's025 is not the publication year. Complete primary XML pcadi_paper.xml and extracted pcadi_paper_fulltext.txt are private; publication metadata and primary text were read, including methods/experiments. This selects source2degree102448orientations,512PC,30x30 internal detector,4refinement rounds,2000real scan points. It does not reproduce the paper1degree DI method reference,1024PC,quantization or all gain experiments. Direct DI at the same selected2degree resolution is separately executed as a scientific comparator.

Original code https://github.com/ZacharyVarley/pcadi, **88f676e6f44812900b3c873692aa1a1c9e48b746**. Remote API commit and exact revision raw utils.py were actually verified (pcadi_verified_commit.json,verify_ebsd_remote.py); not an inferred local tracking revision. pcadi_utils.py is the full193KB source, kept privately. MIT2025ZacharyVarley copyright and license are preserved in task LICENSE/NOTICE. Selected explicit mathematics, not an opaque framework, is extracted locally. Paper CC BY-NC-ND material/figures is not redistributed.

Real data: https://doi.org/10.5281/zenodo.7498632, official record ni1_2, explicit **CC BY4.0**. Private97995078byte ZIP ni1_0 matches record MD5 7393a03afe5d52aec56dfc62e5cefdc3. Pattern.dat is uint8 **149x200x60x60**,29800scan points; bound native scan crop rows50:90,columns60:110 is40x50=2000points. No scan resize or data augmentation. Pitch1.5micrometre, origin(90,75)micrometre,20keV,70degree sample tilt,0degree detector tilt,0dB gain,3500microsecond exposure come from Setting.txt. Not the unrelated55x75 dataset. All2000native60x60patterns are public. Internal2xdetector binning gives30x30 only after loading. The static background is computed solely from those public2000patterns; no outside-ROI pattern or unused calibration BMP is used.

Independent nickel calibration: https://kikuchipy.org/en/stable/reference/generated/kikuchipy.data.nickel_ebsd_master_pattern_small.html explicitly licenses this simulated EMsoft data **CC BY4.0** (actually read primary provider page, ebsd_primary_license_record.json). Exact raw calibration repository revision **fc55fa86470f1510f5598d3ba89d957dedd4c3b3**, src/kikuchipy/data/emsoft_ebsd_master_pattern/ni_mc_mp_20kv_uint8_gzip_opts9.h5, remote bytes reverified, SHA256 **8a7c1fb471d9ce750f0332a154e87cf41eed7529be508548e0c0f51ec6f92bc2**. Northern/southern401x401uint8 Lambert arrays at20keV are public calibration, not target-derived orientation estimates. Independent reader inspection confirms no vertical flip in Lambert representation. No kikuchipy/EMsoft reconstruction code is imported by task runtime/tests/Judge/notebook.

All private artifacts below are under the maintainer-only material_physics directory outside this task; ebsd_archive_index.json records exact file sizes/hashes. Supplied full source, PDFs/XML, download archives, original scripts and logs are not bundled in the task.

## Original runs before extraction

Every run uses the exact bound2000patterns and independent master. CPU one thread, seed20261021 fixes dictionary permutation only. Original source imports its pure Torch monofile privately; the cleaned task never imports this file. Commands are the shared Python executable followed by each named private wrapper. Environment/version/backend evidence is ebsd_numeric_environment.txt; Python3.11.11, NumPy2.4.6,Torch2.9.1+cu128 CPU, OpenMP/MKL/OpenBLAS threads1. No training or GPU is involved.

| Capture | Actual execution | Evidence |
|---|---|---|
| Unmodified original100PC precursor | Initial53.563s; independently repeated actual exit0 in60.3479532s, all2000outputs identical | pcadi_original_screen.*,pcadi_unmodified_verified.*,pcadi_unmodified_verified_run.json |
| Qualified corrected original100PC precursor | Actual exit0,49.9268753s | pcadi_corrected_oracle.*,pcadi_corrected_run.json |
| Independent original stages | Original preprocessing, covariance, orientations, projections and calibrated geometry captured before extraction | freeze_ebsd_stages.py,pcadi_original_stages.npz/json |
| Original8000refinement centers | Actual25.4154468s; four rounds for every2000current | verify_ebsd_refinement.py,ebsd_refinement_center_check.npz/json |
| Qualified corrected original **direct DI** | Actual exit0,34.9016617s including capture; all2000,102448orientations,716uncompressed matching pixels,4rounds | pcadi_direct_original.*,pcadi_direct_original_run.json |

Original direct DI uses actual source dictionary_index_orientations, not PCA with an identity surrogate. Its initial capture stdout inherited two unused PCA-only metadata fields from the wrapper; final JSON removes components/learning_degrees and explicitly records this capture-metadata correction. Original raw stdout remains archived. Actual executed direct-call code, not those unused labels, establishes the algorithm.

### Qualified correction and source discrepancies

The unmodified EBSDGeometry detector coordinates use linspace(0,30,30), whose adjacent spacing is30/29. Independent Bruker half-pixel geometry proves30detector centers should be arange30: with a=i+.5-L*PCy,b=j+.5-L*PCx,D=L*PCz and phi=-70degrees, r=(-sin(phi)a+cos(phi)D,b,-cos(phi)a-sin(phi)D). At the principal point the ray is D*(cos(phi),0,-sin(phi)); independent tests verify rigid rotation and unit adjacent spacing. Native60center pairs correspond to coarse30centers with the same ray direction. Parent approved only replacing these endpoint coordinates; no PC fitting, source-file rewrite or inverse algorithm change. The unmodified actual run remains separately plotted and frozen. This is **qualified reproduction**, never called unchanged reproduction.

Source Bruker PC(.4221,.2179,.4954) differs from paper TSL PC(.4210,.7794,.5049), independently converted to Bruker(.4210,.2206,.5049). A complete active reconstruction at the paper PC is plotted; neither is chosen by hidden-orientation fitting.

Source right-action FZ map is not physical symmetry invariance for sample-to-crystal q. Exact order: pcadi_utils.py5073offset cu2qu;5077offset/current product;5080right-map;5085apply mapped candidate to rays;5096master interpolation;5109dot;5117argmax;5123select **that same scored q**;5133set orientations. There is no post-score replacement. Retained as original candidate heuristic, not a claim of physical equivalence. Actual8000center proposals all unchanged, max component delta0. Original selected512PC mean raw dot by iteration0..4:27.0248527527,29.5760116577,30.2430877686,31.0321750641,31.2823963165.180tiny per-pixel increments are negative down to-7.62939453125e-6 from finite-precision reduction; strict pointwise monotonicity is not claimed. Independent left/right physical-pattern counterexample is retained. Judge symmetry is independently generated24proper **left** crystal actions plus q/-q.

Source TorchScript nominalfloat64sqrt(.5) constants retain FP32 promotion error1.21e-8. Local float64 cubic group is mathematically normalized, fixture tolerance2e-8; actual float32 runtime group remains bitwise equal. Original master tensor aliases in-place normalization, so its initially captured master key was not raw. The stage-capture IO was corrected by rereading independent original HDF5 raw arrays; public master and all original inverse runs use those raw data. This was capture IO, not an inverse-method change.

## Paper/source/local/test traceability

| Required stage | Primary method/source | Local implementation | Independent evidence |
|---|---|---|---|
| Forward diffraction dictionary and unit rotation | Paper DI methods; bruker_geometry_to_SE33214,EBSDGeometry3321,MasterPattern4407,rosca_lambert_side_by_side1778,qu_apply202 | physics_model detector_rays/interpolate_master,orientations qu_apply | Original ray/projection fixtures; independent principal-ray,rotation,Lambert and cubic left/right cases |
| Experimental and independent master preprocessing | ExperimentPatterns3792,clahe_grayscale3678 | preprocessing prepare_patterns/prepare_master/clahe_grayscale | All2000original stage arrays; independently written NumPy evaluator agrees within4e-7 |
| Quaternion/homochoric parameterization | Paper equations1/2; source qu_apply202 and cubochoric/homochoric conversion kernels | orientations cu2ho/ho2ax/cu2qu/qu_apply | Frozen original conversions,independent rotation/norm tests |
| Covariance accumulation and PCA decomposition | Paper equations3/4; compute_pca_components_covmat5342 | solvers update_covariance/learn_components | Frozen original updates,independent batch covariance,eigenspace and full pipeline |
| Explained dictionary variance | Paper equations6/7 | visualization.review cumulative covariance eigenvalue fraction | Actual fresh eigenvalues and numeric_sources.npz cumulative curve |
| Projected dictionary dot matching | Paper compressed DI; pca_dictionary_index_orientations5440,ChunkedKNN4720 | solvers index_patterns | Separate original100PC/selected512PC coarse/final/projector fixtures and all2000outputs |
| Full uncompressed direct DI comparator | Paper direct-DI method reference; dictionary_index_orientations5140 | solvers index_direct | Separately run original direct coarse/refined fixture; both arrays bitwise equal locally |
| Local cubochoric refinement | orientation_grid_refinement4962,get_local_orientation_grid4905,ori_to_fz_laue2461 | orientations conversions/cubic_reduce,solvers refine | Frozen27offsets,8000actual centers,per-round raw dots,full output |
| Geometric/calibration robustness | Paper experimental PC and source PC | active review alternative-PC full inverse;25PC sensitivity | Actual fresh comparisons,all plotted arrays in numeric_sources.npz |

Paper equation5 is the alternative Oja orthogonalized stochastic update, not the selected covariance branch; equation8 generates added Poisson-noise sensitivity data, not these reused real0dB observations. Neither is claimed as executed. PCA covariance uses explicit Welford/Chan updates,lower-triangle float32eigh and selected512eigenvectors; original matching is an unnormalized dot, not silently renamed NCC. Queries are not additionally centered by the learned feature mean. Original DI and PCA branches differ in normalization order: direct DI normalizes900rays before rotation,simulates all900without renormalizing rotated rays,then masks716; PCA branch masks first and normalizes during interpolation. Both local branches preserve their original conventions. Dictionary interpolation align_corners=False; refinement align_corners=True,border padding,all900pixels. No artificial optimizer substitution or domain-library shortcut.

## References, evaluator and tolerances

No actual orientation truth exists. evaluation/baseline_reference.npz is the qualified corrected original512PC method result, not truth. If an actual ground_truth.npz is supplied, it has priority; malformed truth errors rather than falling back. All required output is reconstruction float32(1,40,50,4),scalar-first active sample-to-crystal quaternion. Norm must already be within1e-5 of1 (FP32 representation); arbitrary scale,zero,tiny,2q,complex and nonfinite inputs are errors. Actual original512/direct norm error max1.409e-7 and100PC1.468e-7;1e-5 is a conservative declared input-validity tolerance, separate from the much stricter orientation/forward scoring. Only after validation is FP64 normalization used for stable angles. No cropping,spatialregistration,reference-dependent gain or hidden mask.

Disorientation is2acos(max absolute quaternion dot over24left crystal symmetries),degrees. Independent selected512PC one-FP32-ULP perturbation max7.2445096183544e-6degree and self roundoff3.4150945850063755e-6degree justify nonzero max1e-4/mean1e-5degree tolerances (ebsd_512_assessment.json/npz (100PC precursor in ebsd_independent_evaluation.json/npz)). These are numerical reproduction tolerance, not physical orientation accuracy. Independent public-only NumPy CLAHE/Bruker/Lambert/bilinear forward cosine is maximized over the same24representatives to preserve declared physical equivalence despite finite-grid calibrated intensities. Original512PC mean0.4512641988756253; fixed boundary0.99times this =0.446751556886869. This modest agreement and its full quantiles/residuals are exposed; no external orientation accuracy is inferred. Single actual source-representative normalized forward residual plots differ intentionally from the evaluator's symmetry-max cosine.

Judge imports no src/solver/upstream code, anchors assets at its own file,allow_pickle=False,validates all shape/dtype/unit/finite/config limits,returns pass/fail/error with0/1/2 and finite JSON. Poor orientations/spatial permutation fail. Invalid inputs/config/missing assets/report writes produce descriptive errors and empty metrics. Tests include foreign current directory,GT priority,physical left symmetry,q sign,nonunit scales,tiny norm and independent known angles.

## Actual execution, notebook and figures

Final main/Judge/tests and active nbconvert runs are summarized below from captured actual return codes. The shared development kernel imaging101-batch5-material explicitly invokes the shared research-runtime/Scripts/python.exe; canonical notebook prints executable,NumPy and Torch versions. No Python3global kernel substitute. Timeout600s per cell is justified by full active512PC,100PC,25PC,paper-PC and direct-DI comparisons, all on real observations. The canonical notebook contains8executed code cells and **6embedded PNG** scientific groups; no cache-preview path substitutes for reconstruction.

Six groups: native/processed real patterns; original unmodified/qualified/fresh orientation maps and independent consistency; actual PCA basis/spectrum; paired observed/forward/residual patterns; active component/refinement/PC sensitivity and actual timing; original/fresh direct DI versus PCA and matching-stage timing. Pole color is explicitly defined cubic-invariant explanatory color, not a claim of an external-library IPF or orientation truth. Scan pixels retain physical centers; extent(89.25,164.25,134.25,74.25)micrometre with upper raster origin. Comparators are computed fresh, original captures separately labeled. Matching-only direct/PCA timing excludes PCA learning/refinement and is not presented as full-paper speedup. Figures are outside task at material_physics/figures/ebsd_nickel_crystal_orientation; six PNG/SVG/PDF sets, numeric_sources.npz,fullprecision metrics.json and SHA256manifest. No paper image copied.

Plans were semantically checked against source: quaternion action/sign,Bruker dimensions/half-pixel coordinates,Lambert north/south layout,border/interpolation align conventions,covariance normalization,dot normalization,candidate ordering,stopping4rounds,stage timing,all signatures/defaults/returns and actual caller paths agree. README supplies every public key and physical parameter; it contains no hidden truth/reference values or GitHub URL. Requirements use only shared pinned NumPy,Torch and Matplotlib;Matplotlib is for review. No installation/upgrade is required.

## Gate record

All mandatory gates **passed**: primary paper/exact remote code/data/license/hash; actual original and qualified/local full runs; independent original fixtures and physical checks;units/coordinates/precision;standard CLI and standalone Judge;malformed/poor/evaluator failure contracts;public sufficiency/no target-derived calibration/no leaks;minimal local mathematics;complete unit/parity/pipeline suite;actively executed canonical notebook with actual comparisons;nonzero measured numerical tolerance;README/two plans/source semantic agreement;template/private archive/license audit. Parent acceptance is a separate independent review pending this handoff. No failed or unverified mandatory gate is concealed. Actual orientation truth/absolute physical accuracy and full-paper experiments are unavailable and not claimed.

## Selected512PC improvement and preserved comparisons

Before any local512 output was evaluated, ebsd_512_assessment.json fixed max1e-4/mean1e-5degree numerical boundaries and public-forward boundary0.446751556886869. Independent original source512 run exit0 in64.308427s, same qualified endpoint correction and all other parameters. Additional unmodified512 actual run exit0 in67.9227157s preserves original coordinates, so endpoint effects are compared at equal component count. Original512 refinement instrumentation exit0 in21.1168814s covers all8000actual centers, all unchanged, exact finalq agreement,180tiny negative numerical dot increments (min-7.62939453125e-6). All these captures are frozen before selected local512 parity. upstream_pipeline512.npz and upstream_refinement512_check.npz are separate from consumed100PC precursor fixtures.

Under the same independent public-only model, original100/512/direct mean cosine is0.3831627454946539/0.4512641988756253/0.45129242259574187. Their5%quantiles are0.10424949/0.36520837/0.36186387. This independent observation consistency, rather than source self-similarity, supports upgrading the principal-component count. Against direct DI,100PC mean angular difference9.400671degree drops to512PC0.263746degree;512median0,95%0.554337degree,maximum59.83258degree. The large tail remains visible; direct DI is also a method estimate, not actual truth. Full figures retain freshly computed100PC and25PC negative paths, complete independently specified paper-PC inverse, direct DI, original/fresh parity,512PCA learning and matching costs. No total speedup or physical orientation accuracy is inferred. ebsd_100pc_executed.ipynb preserves the earlier active precursor privately; it is not the final canonical notebook.

## Bound array hashes

| File | SHA256 |
|---|---|
| data/raw_data.npz | cad7ae39c828178c9045ae2dcdf152d802396564bce6eaf6cf19ec7a12e16c60 |
| evaluation/baseline_reference.npz | 05cb5d39e8d2575f1df47b58e07628dfe54b66ae97106c0feb0141f186baafc2 |
| test/fixtures/upstream_direct.npz | cc405082da9e551db39abeb08a2d4de308b1becbc6f13ff02a4d1dabbe222cb5 |
| test/fixtures/upstream_pipeline.npz | 176a94a290ed27f1b9af4e1cec906c7ff9e754236a6bbba70f5996eade47b3c2 |
| test/fixtures/upstream_pipeline512.npz | c90dc9b4c9d47760863da77e2d2b6b11b3083f1b6bcbf3bbdf5e007685fe8266 |
| test/fixtures/upstream_refinement512_check.npz | 89d80a31ccde52a93c70ac3a020c59ef3a44c23dae8b45051038f93f1b3eb67e |
| test/fixtures/upstream_refinement_check.npz | 772ff63d27550f3a11f0384f9c8dd2c4d5453eeeab319e392849ae6542317363 |
| test/fixtures/upstream_stages.npz | f5e7b189fed5b6ac587c7601d0720fadc1809d165519d4f30b3e2f72bb8fd9cb |
| notebooks/reference_outputs/upstream_methods.npz | 0eea476aee3290f0e0cb8e397003b30819281a7d72c7712bb0f4467d5a5303b9 |

## Final captured commands

```json
{
  "main": {
    "command": [
      "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research-runtime\\Scripts\\python.exe",
      "-m",
      "src.main"
    ],
    "seconds": 51.526327100000344,
    "exit_code": 0
  },
  "judge": {
    "command": [
      "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research-runtime\\Scripts\\python.exe",
      "evaluation/judge.py",
      "--prediction",
      "output/reconstruction.npz",
      "--report",
      "output/judge_report.json"
    ],
    "seconds": 11.105458099998941,
    "exit_code": 0
  },
  "tests": {
    "command": [
      "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research-runtime\\Scripts\\python.exe",
      "-m",
      "pytest",
      "test/",
      "-q",
      "-p",
      "no:cacheprovider",
      "--basetemp",
      "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research_batch5_20261021\\material_physics\\ebsd-pytest-final"
    ],
    "seconds": 188.2424628000008,
    "exit_code": 0
  }
}
```

Notebook command and result: ```json
{
  "command": [
    "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research-runtime\\Scripts\\python.exe",
    "-m",
    "jupyter",
    "nbconvert",
    "--to",
    "notebook",
    "--execute",
    "notebooks/ebsd_nickel_crystal_orientation.ipynb",
    "--ExecutePreprocessor.timeout=600",
    "--output",
    "ebsd_executed.ipynb",
    "--output-dir",
    "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research_batch5_20261021\\material_physics"
  ],
  "seconds": 290.91377230000217,
  "exit_code": 0
}
```

Notebook SHA256: 0eedc046197257801d96cbd7be1921b15092fda8c9b449db6f7723731c6a537a

Actual fresh comparison summary: ```json
{
  "judge": {
    "status": "pass",
    "passed": true,
    "metrics": {
      "max_disorientation_deg": 2.4148365394514667e-06,
      "mean_disorientation_deg": 1.098291091440116e-07,
      "mean_forward_cosine": 0.45126419887562524
    },
    "thresholds": {
      "max_disorientation_deg": {
        "comparison": "<=",
        "boundary": 0.0001
      },
      "mean_disorientation_deg": {
        "comparison": "<=",
        "boundary": 1e-05
      },
      "mean_forward_cosine": {
        "comparison": ">=",
        "boundary": 0.446751556886869
      }
    },
    "errors": []
  },
  "fresh_baseline_seconds": 53.37600189999648,
  "dictionary_count": 102448,
  "comparison25_reused_basis_seconds": 37.70995440000115,
  "paper_pc_full_seconds": 57.47309210000094,
  "paper_pc_bruker": [
    0.421,
    0.22060000000000002,
    0.5049
  ],
  "mean_endpoint_change_deg": 2.2046839656047776,
  "max_local_original_difference_deg": 3.4150945850063755e-06,
  "mean_refinement_change_deg": 1.1786442082099846,
  "mean25pc_change_deg": 37.12135209303002,
  "mean_paperpc_change_deg": 0.9865818890477074,
  "independent_forward_cosine_quantiles": [
    0.14394776344900714,
    0.36520836898596587,
    0.45781933793442553,
    0.5100429001875936,
    0.5590685087069286
  ],
  "reference_is_not_truth": true,
  "comparison100_reused_basis_seconds": 39.44037389999721,
  "mean100pc_change_deg": 9.40512530018365,
  "fresh100_vs_original_max_deg": 3.4150945850063755e-06,
  "max_coarse_local_original_difference_deg": 2.9575586669421963e-06,
  "direct_index_seconds": 16.756080099999963,
  "direct_refinement_seconds": 25.83375910000177,
  "direct_max_original_difference_deg": 3.4150945850063755e-06,
  "direct_vs_pca_mean_deg": 0.26374626298902926,
  "pca_stage_seconds": {
    "preprocessing": 0.12345319999440107,
    "pca_learning": 11.243761300000187,
    "indexing": 16.665925700006483,
    "refinement": 25.342780899998615
  }
}
```


## Parent acceptance review

Parent independently reviewed the selected 512-component PCA dictionary indexing method, exact source revision and 2026 publication date, native 2000-point real scan crop, independently simulated Lambert calibration, and public-only background. The explicit Bruker pixel-center endpoint qualification is independently justified and unmodified-original comparisons remain visible. Source right-action FZ reduction is correctly identified as a genuinely scored candidate heuristic, while physical symmetry and the independent Judge use left action. Source, plans, scalar-first quaternion/units contract and CPU operation orders agree. The final complete 44-test independent parent suite passed in189.01s; independent foreign-directory Judge exited0 and contract audit found no errors. Parent opened all six final executed scientific groups, including full direct dictionary indexing and its independently frozen original parity, source/local 512PC, actual 100PC/25PC negative comparisons, alternative-PC inversion, independent public-pattern consistency and separately measured learning/matching/refinement costs. The improved512PC baseline has public cosine .4512642 versus direct .4512924; source/local endpoint disorientation remains at float32 representation roundoff. The direct-comparison tail up to about59.83degrees is retained, no orientation truth is invented, and no overall acceleration or absolute orientation accuracy is claimed. Complete standard CLI, standalone Judge and eight-cell active notebook (290.914s) have actual execution records. Numerical source/tests/references are frozen and all required scientific and packaging gates pass.
