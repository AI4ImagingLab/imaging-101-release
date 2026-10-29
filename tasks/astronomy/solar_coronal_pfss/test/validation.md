# Provenance and completed validation

## Bound problem fixed before extraction

Reconstruct all three signed face-centered components of a current-free coronal magnetic field from the full native GONG synoptic photospheric radial map. The complete 180x360 measured image and its native big-endian float32 precision are preserved. The shell has 35 log-radius cells, inner radius1 and radial-field outer surface2.5 solar radii. This is a conditional potential-field extrapolation, not true measured coronal ground truth. No synthetic generator, target-derived calibration, target model weight or hidden acquisition is used. Only the observation arithmetic mean is removed to enforce magnetic flux balance.

Paper: Stansby, Yeates, Badman (2020), pfsspy: A Python package for potential field source surface modelling, JOSS5(54),2732; https://doi.org/10.21105/joss.02732 . The complete author implementation was remotely verified at commit f493e2cdc653e541425f01318ec2f35645ee36d0, https://github.com/dstansby/pfsspy/tree/f493e2cdc653e541425f01318ec2f35645ee36d0 . API commit plus codeload archive verification is recorded in sources.json; a failed/hanging git transport was not represented as a successful fresh checkout. Source ZIP and complete original source are private. The published algorithm is a separable spherical finite-volume solve; no iterative surrogate is substituted.

Data: author sample_data.py GONG sample, https://gong2.nso.edu/oQR/zqs/202009/mrzqs200901/mrzqs200901t1304c2234_022.fits.gz . The SHA256 equals the author-declared sample hash. It is the September1,2020 13:04UTC Carrington2234 GONG map using7391 inputs. Full header coordinate metadata remain archived in the FITS. The slightly rounded FITS CDELT2=.0111111 and exact model increment2/180 are separately identified in README. Longitude left edge22degrees, Gauss units. Native values are unchanged in public magnetogram.

## Licenses and notices

Original pfsspy code is GPL-3.0; its complete license is retained as task LICENSE and copyright/adaptation notices appear in extracted modules. NSO's official data page links the image use policy, https://nso.edu/data/ and https://nso.edu/about/image-use-policy/ , which releases NSO images under CC BY4.0 with credit. Both pages are archived privately as nso_data_policy_link.html and nso_image_policy.html. Required acknowledgment: Data were acquired by GONG instruments operated by NISP/NSO/AURA/NSF with contribution from NOAA. The task README and every observation figure identify NSO/AURA/NSF; source and full credit are preserved. Repository code licensing is not used as a substitute for observation rights.

## Independent original execution before local mathematics

Private archive root: D:\yan1\agent\AutoSkill\imaging-101-maintainer\research_batch3_20261020\batch3\astro_earth_physics . It contains sources.json, pfsspy.zip, archives/pfsspy-f493e2cdc653e541425f01318ec2f35645ee36d0/, gong.fits.gz, capture_pfss.py, pfss_oracle.npz/json/log and pfss_stages.npz. The independent capture script imports only the unmodified archived original and domain dependencies in oracle-env. It reads the exact full final measured map, converts to float64 and subtracts the observed arithmetic mean, then calls pfsspy.Input and pfsspy.pfss. Original intermediate locals are captured on the original function return with sys.setprofile; no cleaned function generates fixture expectations. The full original vector field is captured directly from Output.bc and transposed to declared radius-first axes. Full edge-integrated potentials remain private, with only documented probes in shipped fixtures.

Command: oracle-env/Scripts/python.exe capture_pfss.py, working at private archive root. Actual Python subprocess return0 was recorded; stdout/stderr are in pfss_oracle.log. Runtime and environment are in pfss_oracle.json. It ran before build_pfss.py/local extraction. A repeat independently ran the identical original command and reproduced the exact NPZ SHA256. Warning: SunPy reports missing optional extras and FITS does not parse literal Gauss as a standard unit string; numeric values are confirmed by the FITS header, algorithm and original field scale, and the warning does not change data. Oracle environment: Windows10 build26200, CPython3.11.11, NumPy2.4.6, SunPy7.0.5, Astropy8.0.2; private domain libraries do not enter cleaned requirements. All BLAS/OpenMP thread limits1.

Acceptance tolerances were fixed before extraction: stage matrices and potential probes rtol1e-10,atol1e-10; all full field components rtol1e-9,atol1e-10; independent continuum dipole relativeL2<.03. Baseline is the original standard method, never named truth. No ground_truth.npz exists.

## Equation and implementation traceability

| Mathematical stage | Original file/function | Local function | Independent evidence |
|---|---|---|---|
| Flux balance, full sine-latitude/log-radius grid | pfsspy/grid.py Grid and pfsspy/pfss.py pfss | preprocessing.balance_flux, physics_model.grid | exact original balanced input; README constant checks |
| Spherical edge factors, Fourier longitude eigenvalues and latitude matrix | pfsspy/pfss.py pfss Fs,Fp,mu,A | solvers.solve | upstream_stages all matrices and factors |
| Growing/decaying radial recurrence and source-surface coefficients | pfsspy/pfss.py pfss Zlp,Zlm,psir,psi | solvers.solve | captured psi probes; full vector parity; analytic dipole |
| Edge-integrated vector potential | pfsspy/pfss.py pfss als,alp | solvers.solve | original als/alp probes |
| Discrete curl, areas, southward polar sign and pole regularity | pfsspy/output.py Output._common_b, Output.bc | physics_model.field_from_potential, area_factors | every original field element; boundary/dipole/linearity/seam tests |
| Physical finite-volume flux conservation | independent divergence theorem | physics_model.flux_divergence | axisymmetric and non-axisymmetric independent test; actual native imbalance |
| Evaluation | independently defined standard metrics | evaluation/judge.py score, visualization.metrics | known two-element cases, poor/malformed/foreign-cwd/evaluator contract tests |

An independent diagnostic review caught an area interpretation issue: the original uses adjacent-face surrogate areas solely to define a finite polar field at a zero-area coordinate singularity. Physical flux diagnostics must use zero true polar face area, not that surrogate. The reconstruction mathematics and all upstream parity fields were unchanged; flux_divergence now explicitly sets pole fluxes to zero and the independent non-axisymmetric conservation test exercises this distinction. There is no hidden cosmetic residual fix. Other source conventions preserved include original Nyquist handling, constant-monopole omission, Fourier normalization, opposite-longitude polar regularity and all staggered component shapes.

## Asset consumers, metrics and result provenance

data/raw_data.npz contains only measured magnetogram(1,180,360). test/fixtures/upstream_stages.npz contains output_balanced,A,Fs,Fp,mu and psi/als/alp probes, consumed by comparing_test/test_parity.py. evaluation/baseline_reference.npz contains the three original full fields, consumed by that comparison and standalone Judge. notebooks/reference_outputs/upstream_measurements.json identifies original run/data/environment and is used for provenance review. No redundant answer-array copy is shipped. output/reconstruction.npz is freshly computed by the final CLI, and judge_report.json is the generated report.

NCC is uncentered flattened cosine; NRMSE is RMS error divided by reference dynamic range. Every full component is separately evaluated with NCC>=.999999 andNRMSE<=1e-6. The nonzero boundaries permit finite-precision/backend variation and float32 submission, rather than mechanically1.1times zero self-error. Independent original reruns were identical. Strict stage and full parity are much tighter than Judge thresholds; analytic and conservation tests independently check the physical model. There is no cropping, phase/sign alignment, rescaling or masked component omission. Historical paper numbers are not represented as this run.

Fresh measurements and exact numeric trace live privately at final_figures/solar_coronal_pfss/fresh_measurements.json and numerical_trace.npz; all PNG/SVG/PDF exports are indexed by manifest_pfss.json. Measurements:

```json
{
  "fresh_runtime_seconds": 0.5895124000962824,
  "removed_measured_mean_gauss": -0.11282645460453997,
  "original_agreement": {
    "radial": {
      "ncc": 1.0,
      "nrmse": 0.0
    },
    "polar": {
      "ncc": 0.9999999999999998,
      "nrmse": 4.864039120988965e-19
    },
    "azimuthal": {
      "ncc": 1.0,
      "nrmse": 1.749816391793141e-19
    }
  },
  "max_cell_flux_imbalance_G_Rsun2": 3.469446951953614e-18,
  "max_inner_boundary_error_G": 4.81439776933712e-10,
  "max_shell_signed_flux_G_Rsun2": 4.423544863740858e-16,
  "source_unsigned_flux_G_Rsun2": 4.288133603899058,
  "source_radius_unsigned_flux_G_Rsun2": {
    "2.0": 5.080458627546665,
    "2.5": 4.288133603899054,
    "3.0": 3.69380440436813
  },
  "half_longitude_source_agreement": {
    "ncc": 0.9999317153492367,
    "nrmse": 0.0026187357048137743
  }
}
```

Notebook actively recomputes the complete bound native vector from cleaned src, reads independently captured original fields only for comparison, and evaluates the fresh result. It also freshly solves source surfaces2,2.5,3 and an actual every-other-longitude acquisition subset; the benchmark output remains full native. All vector components are shown with clearly labeled averages to cell centers. Five evidence figures cover observation/reconstruction/original/residual, complete vector, conservation/radial decay, boundary model and acquisition sensitivity. Signed99th-percentile display limits are labeled; arrays and full-range parity residual are never clipped. The original repository has no single mandatory benchmark-comparison notebook for this selected native observation; unrelated field-line/tracing/interpolation examples are outside the defined inverse output, not replaced by invented comparisons.

## Final verification and semantic review

Shared clean runtime is CPython3.11.11, NumPy2.4.6, Matplotlib3.11.1 on WindowsCPU. Only these actual runtime dependencies are pinned in requirements. No installation/upgrades or domain package imports occur in cleaned source, tests, Judge or notebook. FFT, eigenvalues and arrays are generic NumPy operations with local task mathematics. No source imports upstream source, fixture, checkpoint or another task. main imports do not select/change a plotting backend.

Final actual commands, runtimes and return codes follow; logs pfss_final_0.log through pfss_final_3.log are private. Full notebook timeout300sec is adequate relative to measured10sec CPU execution; private imaging-research kernel explicitly points to shared clean interpreter, JUPYTER_PATH is private. The canonical notebook retains executed outputs and figures are explicitly displayed once. Temporary execution copies remain private, not output/.

```json
[
  {
    "command": [
      "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research-runtime\\Scripts\\python.exe",
      "-m",
      "src.main"
    ],
    "runtime_seconds": 2.317517099902034,
    "returncode": 0
  },
  {
    "command": [
      "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research-runtime\\Scripts\\python.exe",
      "evaluation/judge.py",
      "--prediction",
      "output/reconstruction.npz",
      "--report",
      "output/judge_report.json"
    ],
    "runtime_seconds": 0.7186304000206292,
    "returncode": 0
  },
  {
    "command": [
      "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research-runtime\\Scripts\\python.exe",
      "-m",
      "pytest",
      "test/",
      "-v",
      "-p",
      "no:cacheprovider",
      "--basetemp",
      "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research_batch3_20261020\\batch3\\astro_earth_physics\\tmp_pfss_final"
    ],
    "runtime_seconds": 7.769816400017589,
    "returncode": 0
  },
  {
    "command": [
      "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research-runtime\\Scripts\\python.exe",
      "-m",
      "jupyter",
      "nbconvert",
      "--to",
      "notebook",
      "--execute",
      "notebooks/solar_coronal_pfss.ipynb",
      "--ExecutePreprocessor.timeout=300",
      "--output",
      "solar_coronal_pfss.ipynb",
      "--output-dir",
      "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research_batch3_20261020\\batch3\\astro_earth_physics\\executed"
    ],
    "runtime_seconds": 10.441562799969688,
    "returncode": 0
  }
]
```

Plans were semantically reviewed against source: no missing optimization objective/update is implied for this closed-form linear eigensystem solve; root signs, recurrence ratios, DFT scaling, Gauss conversion, southward polar sign, radial/log-coordinate arrays, exact staggered dimensions, source-radius defaults and CLI/notebook/evaluator flows agree. README independently supplies all input/acquisition/output conventions and never exposes hidden assets or GitHub URLs. Licensing and all fixture consumers were reviewed. Limitations remain current-free/static model, uncertain polar observations and time-combined synoptic acquisition. Exact agreement with a published standard solver is not empirical verification of the true corona.

| Mandatory gate | Status |
|---|---|
| Paper, exact remote commit, licensed measured data, hashes, traceability | passed |
| Independent original exact-bound full run and repeat before extraction | passed |
| Independent fixtures, all-stage/full parity, analytic and conservation checks | passed |
| Constants, signs, units, precision and coordinates | passed |
| Fresh standard CLI and strict independent Judge | passed |
| Judge malformed, missing, nonfinite, poor, evaluator, report-write and foreign-cwd cases | passed |
| Public allowlist suffices; no target/GitHub leaks | passed |
| Clean minimal CPU runtime and local mathematics | passed |
| Full11-test suite | passed |
| Active full nbconvert and computed scientific comparisons | passed |
| Identified original/fresh metrics, nonzero justified thresholds | passed |
| README and both plans semantic review | passed |
| Template inventory, licenses and final actual figure inspection | passed after recorded image review |

## Hash and archive index

```json
{
  "task": {
    "LICENSE": "4c2ffab9b609744a94ef3d68a5e5944ff928c6f8f358061049397cf80da71ea3",
    "NOTICE": "7f2c8ce4f2208537b109dda489647fe127c13728bcd67c21df5159c46550ecfb",
    "README.md": "260d324a48ca65f1724538a16ce197f89aa84f2c89f07b0cf64f2def8abd4a0f",
    "requirements.txt": "75618fbc20ccc5860a4243cda7dca3734342f9a46ac084a2d4c96164404ea787",
    "data\\raw_data.npz": "99994a41446d75a87aac3b0f0b0a172ad6e2037e28d18b8f62a8b79e0fecb22c",
    "evaluation\\baseline_reference.npz": "b4255926c4a35d7576e4b9ace9280fc26713bb1ef5495b2bbfe1967e8457f520",
    "evaluation\\judge.py": "c1810cf7c14b9c5d57314f5fc72ad079468eb03e5a86e0d6c6d46a71bb219b5f",
    "evaluation\\metrics.json": "764639ff4bebd4e1da5c5ba6d50c7fb8879d76b6fbc45b6c8fc7994f839d5c8f",
    "notebooks\\solar_coronal_pfss.ipynb": "c71b484a72ec7097e620ca68b2ad613f9913bba7f75053a9e8748b160bd4934a",
    "notebooks\\reference_outputs\\upstream_measurements.json": "5f8bf0594a6cf28a7a6aa41b8596b2b6f461b513b5b7378e6107114ef8280a3b",
    "output\\judge_report.json": "cb42ec29c0f1d2f8cead7baad411331aae1476d45949cc274ac9fd5490419ae6",
    "output\\reconstruction.npz": "6b936353230ded829f05842ea070853432fcf56cce4bbf06873aada2c2d4ee7d",
    "plan\\approach.md": "9470662a1586517b568c4cdcfd66005fa86ac2a2fb27c62daef0f9ad55bf6a24",
    "plan\\design.md": "8a4f51c675f569d206efedd269614cd5fb7b4b4cd0fc8eab150220ccecc66a29",
    "src\\main.py": "a0bef86a199f0d63bdf3e8733ef5af01703e99a34094249bae514b467bf4a51a",
    "src\\physics_model.py": "e4a572f7bd231d2a5b24e351456ecfe9e99d0799ab1aa682425ce57e7f41bffc",
    "src\\preprocessing.py": "b500d3e3b5bc30b279739a40ea67aebb6b644d065969db92fa52ae2de8497b37",
    "src\\solvers.py": "af448d95381b43ec92317fb8f63f370060a799fb01586a495693067de6837091",
    "src\\visualization.py": "37ee31da69ce7f437ad79049f84a1622ff8f9e7cf5e29345c7eb2e68cb179973",
    "src\\__init__.py": "35e3ada96411f79d26a0c174db7d4d77e8c60f15b1d1cb9e2f408e840bb1b1b3",
    "test\\comparing_test\\test_parity.py": "38b3599445e2c19fb1ca41057fbf091f7969b90e03c6dcfbac2f76c80f4c36e6",
    "test\\fixtures\\upstream_stages.npz": "c23dfe5230ccb8c3832cef840b460d370dd0f3f1c0680a21349aaaf9947df882",
    "test\\unit_test\\test_judge.py": "56dffea0e624840eb9db4734f4ea669d776ef971165d6b95d6b4eaa1ba0a2d34",
    "test\\unit_test\\test_physics.py": "a38dd7679235969dc84568796cc0a2aaf8955cdbf101b33dc10125391f4b8b2c"
  },
  "private": {
    "gong.fits.gz": "aad927d8f617f32b72255b862c4910f13640fc7ca13edf98288cd0735a2db6a0",
    "pfsspy.zip": "3e98e4c73801a2bda085f8005404bcc03428991bec51cdf64e1bfac7f650a6ce",
    "capture_pfss.py": "d0eed70f15aef11aafb09a48e79b7bb8ed2c858c91b3b2ee8f3bc8e5672e7b8a",
    "pfss_oracle.npz": "becd601626d1f679e4f27ab459dbe50490376034085fe8f82a627eb97fc2fb28",
    "pfss_oracle.json": "5f8bf0594a6cf28a7a6aa41b8596b2b6f461b513b5b7378e6107114ef8280a3b",
    "pfss_stages.npz": "c23dfe5230ccb8c3832cef840b460d370dd0f3f1c0680a21349aaaf9947df882",
    "nso_image_policy.html": "aec10d173d770e12232a2f469d08b155b2fccfdc5cf70f2cae509fb6d3d38e42",
    "nso_data_policy_link.html": "b2a495dcb896fe607183cc16b93817113d7bd5462216606cde59ff55f1e48513"
  }
}
```

Task writes are frozen for parent review after this validation and inventory. Parent independent acceptance is a separate gate and is not claimed here.


## Independent parent acceptance

Passed: parent independently ran 11 distinct tests in the cleaned shared environment and the standalone Judge from a foreign working directory. Source/public-interface/dependency/call-role/executed-notebook/inventory audit has no errors. Reviewed spherical finite-volume eigenproblem, Fourier normalization, all staggered field components, potential curl and southward colatitude sign, native FITS precision and legitimate observed flux balancing. Independent dipole, linearity/seam and true-zero-polar-area divergence checks pass. All five actual scientific figures were opened, including source-radius and angular-sampling sensitivities. Conditional potential-field agreement is not claimed as observed-corona accuracy. No figure or array was accepted on an unexecuted preview. Parent raw test/audit evidence and final file hashes are indexed under `D:\yan1\agent\AutoSkill\imaging-101-maintainer\research_batch3_20261020`. Final source hashes are refreshed after this acceptance-only note.
