# Authoritative provenance and validation

## Bound measured experiment and published method

Domain physics; task `terahertz_waveform_noise_calibration`. Unknowns are all 267 latent waveform samples, all fifty scan amplitudes and delays (first amplitude/delay gauge fixed), and three shared noise amplitudes. Total free dimension is 368. The six required outputs additionally retain every fitted secondary waveform and model standard deviation. This is an instrument-noise/drift inverse, distinct from a material's refractive-index or absorption inversion.

Primary paper: Laleh Mohtashemi et al., **Maximum-likelihood parameter estimation in terahertz time-domain spectroscopy**, Optics Express 29, 4912–4926 (2021), https://doi.org/10.1364/OE.417724. Complete author-hosted publication PDF was retrieved from https://www.sfu.ca/lux/publication/mohtashemi-2021/mohtashemi-2021.pdf and archived as `method_2021.pdf`; `method_2021.txt` is its extracted full text. Section 5 explicitly specifies secondary waveforms, first-scan gauge, secondary-waveform-dependent variance, joint negative likelihood and finite-sample correction (paper equations 31–41). This is the selected noise-method scope; transfer-function/sample fitting and the paper's other simulations are not claimed experiments of this task.

Software publication: J. Posada Loaiza et al., **THzTools: data analysis software for terahertz time-domain spectroscopy**, JOSS 9(104), 7542 (2024), https://doi.org/10.21105/joss.07542, full PDF `software_2024.pdf`. Exact remotely verified source is https://github.com/dodge-research-group/thztools/tree/b4ecea908921fdd30d996b8eb12513a777e3807a . `remote_commit.json`, complete `upstream.zip` and `source.json` are under `math_physics/thztools`, obtained before local extraction. This is a live verified code commit, not an unverified local tracking revision.

Measured observations are the author's **2017-03-20 Air Scans/Normal**, all 50 native Scan*.thz files and all 267 samples per file, from the MIT-licensed v1.0.2 MATLAB deposit https://doi.org/10.5281/zenodo.4876388 . Deposited archive MD5 **766d6da2cb2642351e8d5c77f51aaec3** matches the independently checked downloaded archive. The source `dataexp.m` specifies gain 5e7 V/A and voltage ×1e12/gain → pA. Private `native_data_verification.json` verifies every supplied sample and all 50 native hashes against this conversion. No synthetic observations, hidden seed, object, target-derived calibration, crop, denoising, baseline subtraction or reduced precision is used.

Historical-method distinction: the 2021 archived MATLAB script selected the first 256 samples and approximated the variance with a common waveform, while the modern selected Python implementation evaluates each scaled/shifted secondary waveform as in the paper's general likelihood. Here the complete modern joint method uses all 267 native samples. No historical paper table is relabeled reproduction on these full bound observations.

Code and data are MIT licensed. The Python-source original notice is preserved in `LICENSE`; the data deposit's original 2020 author copyright and full MIT text are preserved in `NOTICE`, with the exact voltage/pA adaptation. Paper PDFs stay private and retain their separate publication terms. Zenodo's API request returned HTTP 403 during this agent's metadata fetch; the accessible primary landing page, actual license within the archive and matching publicly displayed archive MD5 establish the verified deposit identity without pretending that the API fetch succeeded.

## Independent original execution and stopping diagnosis

Before local extraction, the parent executed the unmodified published `thz.noisefit` implementation on the exact full `(267,50)` calibrated observations, with dt=0.05003462406015038 ps, mu0=mean(X,axis=1), est_mu=False, workers=1 and min_options={maxiter:10000,gtol:1e-6}. The source parser was observed through a capture-only wrapper that calls the original function unchanged; no numerical function/file was modified. Original driver is private `capture_thz_oracle.py`, full array capture `upstream_air_scans.npz`, report `oracle_air_scans.json`. Parent runtime was approximately 11.23 seconds; Python exit status zero. All initialization, scales, objective, analytic gradient, final waveform/drifts/noise, scaled iterate/gradient, inverse curvature and marginal errors were frozen before extraction. Additional original `scaleshift` and `NoiseModel.noise_amp` numerical fixtures were captured in `extra_upstream.npz` before local mathematics.

**Strict optimizer gtol: NOT MET.** The original canonical BFGS diagnostic is success=False, “Desired error not necessarily achieved due to precision loss.” Its NLL is 5066.818699894164, iteration count 169 and maximum scaled gradient 1.996988099860724e-5. The source success flag is not changed to true, and gtol/maxiter/initialization/output parity boundaries are not relaxed. This is distinct from the mandatory original-program execution gate, which completed successfully and returned all numerical quantities.

Before local implementation, `investigate_original.py` independently checked the original analytic gradient against centered directional objective differences and differentiated the original gradient using oracle-only numdifftools. The Hessian is positive definite. A diagnostic Newton step demonstrates negligible remaining likelihood improvement and near-zero gradient at an adjacent stationary point. Parent explicitly accepted this nearstationarity evidence while requiring canonical BFGS, its false flag and unchanged criteria to be retained. The Newton point is **never** a submitted reconstruction, hidden baseline, initialization replacement or invented truth.

Independent pre-extraction original-function audit:
```json
{
  "original_success": false,
  "original_message": "Desired error not necessarily achieved due to precision loss.",
  "dimension": 368,
  "n_observations": 13350,
  "nll": 5066.818699894164,
  "gradient_inf": 1.996988099860724e-05,
  "gradient_inf_per_observation": 1.4958712358507296e-09,
  "hessian_min_eigenvalue": 0.07012178740495172,
  "hessian_max_eigenvalue": 250.844884455233,
  "newton_decrement": 5.41453695287039e-11,
  "newton_nll": 5066.818699894139,
  "newton_gain": 2.4556356947869062e-11,
  "newton_scaled_step_max": 4.1078779023493e-06,
  "newton_physical_step_max": 3.254983056698759e-07,
  "newton_gradient_inf": 5.569605063668968e-12,
  "directional_gradient_checks": [
    {
      "point": "initial",
      "direction": 0,
      "analytic": 5.535790616514265,
      "finite": 5.5357906294375425,
      "absolute_error": 1.2923277559195867e-08
    },
    {
      "point": "initial",
      "direction": 1,
      "analytic": 6.815162772797111,
      "finite": 6.815162774273631,
      "absolute_error": 1.476520239407364e-09
    },
    {
      "point": "initial",
      "direction": 2,
      "analytic": -23.611153032599713,
      "finite": -23.611153049387212,
      "absolute_error": 1.6787499390602534e-08
    },
    {
      "point": "initial",
      "direction": 3,
      "analytic": 35.03904312788795,
      "finite": 35.03904312356099,
      "absolute_error": 4.326963676248852e-09
    },
    {
      "point": "initial",
      "direction": 4,
      "analytic": -28.42318504413153,
      "finite": -28.42318506554875,
      "absolute_error": 2.1417221773845085e-08
    },
    {
      "point": "final",
      "direction": 0,
      "analytic": -5.264965224391086e-06,
      "finite": -5.2600626077037305e-06,
      "absolute_error": 4.902616687355283e-09
    },
    {
      "point": "final",
      "direction": 1,
      "analytic": 1.2974178797154682e-06,
      "finite": 1.2823875294998288e-06,
      "absolute_error": 1.503035021563939e-08
    },
    {
      "point": "final",
      "direction": 2,
      "analytic": 7.937005284276569e-07,
      "finite": 7.980816008057445e-07,
      "absolute_error": 4.381072378087623e-09
    },
    {
      "point": "final",
      "direction": 3,
      "analytic": -8.944695625089942e-07,
      "finite": -8.981260180007666e-07,
      "absolute_error": 3.656455491772479e-09
    },
    {
      "point": "final",
      "direction": 4,
      "analytic": -2.4942127190398375e-06,
      "finite": -2.5324879970867187e-06,
      "absolute_error": 3.8275278046881175e-08
    }
  ]
}
```

A second complete unchanged original `thz.noisefit` execution after extraction ran privately as `<math_physics/oracle-env/python> material_mechanics/thz_noise/repeat_original.py`, Python exit 0, and repeated **every captured final quantity bit-identically**. Its actual false flag is retained:
```json
{
  "runtime_s": 12.145849499851465,
  "success": false,
  "message": "Desired error not necessarily achieved due to precision loss.",
  "relative_differences": {
    "output_mu": 0.0,
    "output_a": 0.0,
    "output_eta": 0.0,
    "output_noise": 0.0,
    "output_final_theta": 0.0,
    "output_final_gradient": 0.0,
    "output_fval": 0.0,
    "output_hess_inv": 0.0,
    "output_err_mu": 0.0,
    "output_err_a": 0.0,
    "output_err_eta": 0.0
  }
}
```

The original oracle is isolated under `math_physics/oracle-env`; only it has thztools and numdifftools. `oracle_environment.txt` pins its packages. Clean shared runtime is Windows/Python 3.11.11, NumPy 2.4.6, SciPy 1.17.1, Matplotlib 3.11.1, CPU; OMP/OPENBLAS/MKL threads and FFT workers are one. `clean_environment.json` records numerical backend and confirms both domain/oracle packages are absent from clean runtime. No environment upgrade or per-task domain package installation occurred. Physical and commit headroom stayed above 6 GiB; arrays/fits are small CPU workloads.

## Scope, frozen criteria and traceability

`success_criteria.json` is the parent's pre-extraction original binding; `frozen_scope.json` additionally fixes all six output keys, shapes/units and the honest canonical stopping policy before local implementation. Original stage tolerances are rtol=1e-6,atol=1e-7; parser and analytic stage tests use tighter recorded tolerances where appropriate. Every final quantity must meet relative Euclidean error ≤1e-6, separately for the three differently dimensioned noise components. No criterion was widened. All six current output errors are exactly zero. Original repeatability and independent nearstationarity measurements justify a nonzero numerical comparison allowance without claiming physical truth.

| Paper stage / mathematical content | Original source function | Local function | Independent validation |
| --- | --- | --- | --- |
| Known instrument gain and unchanged measured records | MATLAB dataexp.m / thzload.m | Private lossless conversion; preprocessing.load_data | native_data_verification.json and public metadata test |
| Secondary waveform and first-scan gauge, paper 31–33 | scaleshift; _nll_common | physics_model.shift_waveforms / decode / common | Original full forward fixture; integer-delay, sinusoid and adjoint tests |
| Spectral derivative, paper 12 and 34 | NoiseModel.noise_var; _nll_common | physics_model.derivative / variance_components | Analytic sinusoid, skew-adjoint and constant-signal noise limits |
| Heteroscedastic joint likelihood, paper 35–37 | _nll_noisefit / _nll_common | physics_model.objective / common | Original initial/final NLL and independently derived finite differences |
| Scales, regression noise initialization and latent mean | _parse_noisefit_input | physics_model.initialize | Every original initial parameter/scale/cost/gradient fixture |
| Full waveform/amplitude/delay/logvariance analytic gradient | _jac_noisefit, selected est_mu=False branch | physics_model.gradient | Original initial/final gradient; independent directional and adjoint checks |
| Joint BFGS updates and source stopping | noisefit -> general SciPy minimize BFGS | solvers.reconstruct | Full original waveform, all drifts/noise, NLL/stopping state parity |
| Finite-M amplitude correction, paper 41 | _parse_noisefit_output | physics_model.decode_outputs | Known sqrt(M/(M-1)) case and all three original output amplitudes |
| Observed inverse curvature and marginal-error conventions | _parse_noisefit_output: numdifftools.Jacobian(jac) | physics_model.curvature: explicit five-point derivative of analytic gradient | Original independent Hessian, inverse curvature and every saved marginal error |
| Real acquisition prefixes and additive-model comparison | New controlled notebook experiments | visualization.experiments / make_figures | Actively executed complete fits, exact plot-source NPZ and inspected PNG/SVG/PDF |

The source `_hess_noisefit` analytic helper is not called by the selected original output path; its large unused alternate implementation is not represented as shipped runtime. The original method obtains curvature numerically from the analytic gradient. Local code implements that numerical derivative explicitly with a five-point centered formula, using permitted general primitives and no numdifftools import.

## Data/reference/fixture identity and consumers

Public observation SHA256: `c0150fb6b5a831d6560677779a55cee397bdb7edf802f8c501c7f8bd3c6bcd7b`.
Independent original full reference SHA256: `6aeec1837012b7f435b200b9e382cbe78a1ef4135f7e178c3c56b3d52988cd25`.
Actual latent/noise truth is unavailable. `evaluation/baseline_reference.npz` contains only the six declared original standard-method arrays, consumed by the standalone Judge, full-pipeline test and explicitly labeled original notebook comparison. No `ground_truth.npz` is invented. `upstream_measurements.json` identifies original code/data/environment/parameters and precision investigation; `fresh_measurements.json` is the actual active notebook's separate fresh reconstruction/experiment/runtime/environment record. No archived output is substituted for a fresh fit.

`test/fixtures/upstream_air_scans.npz` consumers: initialization test covers initial theta, NLL, analytic gradient, all three regression sigmas, mu0/a0/eta0, all scale arrays/scalars, every fix/est_mu flag and single FFT worker. It also evaluates the original final theta's objective and gradient. Full-pipeline test covers output_mu/output_a/output_eta/output_noise/output_fval and actual stopping state. Curvature test covers output_hess_inv and output_err_mu/output_err_a/output_err_eta. `extra_upstream.npz` contains independently executed original expected_waveforms/noise_std and both are checked against the fresh full pipeline. `diagnostic_stages.npz` contains original gradient, independent original Hessian, diagnostic Newton step/theta; the curvature test checks each, positive definiteness and trial stationarity. Every fixture is independently sourced and has a documented consumer.

Private `archive_index.json` lists absolute paths, sizes and SHA256 for source ZIPs/commit evidence, native data archive, paper assets, capture drivers, numerical captures, environment specifications and logs. Complete source checkouts, original `.thz` files, downloads/PDFs, capture scripts and original logs stay outside tasks. The numerical trace of each final figure includes all raw traces, both original/fresh output arrays, all real-prefix results/curvature, additive-only result, variance components, standardized residuals, QQ quantiles, actual objective histories and latent-minus-mean difference. Figures are original-code measurements/fresh-code measurements rather than historical paper plots.

## Interpretation, discrepancies and uncertainty conventions

1. The source canonical strict gtol remains unmet and the precision-loss flag is false. Independent FD/SPD/nearby Newton diagnostics establish numerical quality without changing the selected result or claiming success=True.
2. The 2021 published general model uses each secondary waveform's variance; the archived MATLAB approximation and 256-sample scope differ from the selected modern full-window implementation. These are documented version/acquisition differences rather than unchanged historical numerical reproduction claims.
3. Finite-M correction multiplies all noise amplitudes by sqrt(M/(M-1)). It is approximate; residual incidental-parameter bias may remain. NLL is evaluated at uncorrected MLE variances while returned noise_std uses the corrected output model, and these are never conflated.
4. The source returned inverse-curvature scale uses positive waveform-deviation scale, although the physical waveform map has a minus sign. Marginal variances are unaffected; local diagnostics match this source-coordinate convention and do not present its off-diagonal entries as physical covariance. Noise marginal standard errors retain the source's additional finite-M multiplication. Plotted errors are joint observed-curvature approximations conditional on model/data, including nuisance coupling, not verified coverage intervals or physical ground truth.
5. Prefixes of 10/25/50 acquired scans are correlated datasets and retain all time samples. The additive-only ablation fixes the other two variances and is never substituted for the selected full three-noise inverse. Residual QQ and variance curves are qualitative fit diagnostics; fitted-residual dependence precludes an unqualified iid normality p-value.
6. Periodic band-limited shifts/derivatives and conditional independent Gaussian noise are explicit assumptions; unmodeled colored noise/nonstationarity and finite-window effects may remain. The measured photocurrent is in pA, not falsely labeled absolute electric field.

Fresh active notebook measurements:
```json
{
  "method": "Fresh selected full joint BFGS; independent Hessian diagnostics do not replace it",
  "sample_counts": {
    "50": {
      "noise": [
        0.6276475127579297,
        0.01567228705405409,
        0.0017742194019303627
      ],
      "err_noise": [
        0.005422872747771413,
        0.00032103509261619586,
        6.620475445837538e-05
      ],
      "success": false,
      "message": "Desired error not necessarily achieved due to precision loss.",
      "nll": 5066.818699894164,
      "gradient_inf_per_observation": 1.4958712358507296e-09
    },
    "10": {
      "noise": [
        0.6412372728776576,
        0.017198299713686475,
        0.0016948723309495079
      ],
      "err_noise": [
        0.01297519746046912,
        0.0007941052788519615,
        0.0001691086770660335
      ],
      "success": true,
      "message": "Optimization terminated successfully.",
      "nll": 984.6057047614399,
      "gradient_inf_per_observation": 2.40176885008837e-10
    },
    "25": {
      "noise": [
        0.6341735276811076,
        0.01564453921131416,
        0.0017840354395570392
      ],
      "err_noise": [
        0.007850514036402906,
        0.0004734807104887612,
        9.988431600729244e-05
      ],
      "success": false,
      "message": "Desired error not necessarily achieved due to precision loss.",
      "nll": 2514.4824394278676,
      "gradient_inf_per_observation": 5.735665291478191e-10
    }
  },
  "additive": {
    "nll": 12016.96915982116,
    "success": false,
    "message": "Desired error not necessarily achieved due to precision loss.",
    "noise": [
      1.5071928111298538,
      1.9483315667254413e-22,
      9.748403748563124e-24
    ]
  },
  "experiment_runtime_s": 5.7844247000757605,
  "reconstruction_runtime_s": 1.2754376998636872,
  "selected_diagnostic": {
    "success": false,
    "message": "Desired error not necessarily achieved due to precision loss.",
    "iterations": 169,
    "nll": 5066.818699894164,
    "gradient_inf": 1.996988099860724e-05,
    "gradient_inf_per_observation": 1.4958712358507296e-09
  },
  "stationarity_audit": {
    "actual_success": false,
    "scaled_gradient_max": 1.996988099860724e-05,
    "gradient_max_per_observation": 1.4958712358507296e-09,
    "hessian_min_eigenvalue": 0.0701217874034354,
    "newton_decrement_diagnostic_only": 5.414536952946593e-11,
    "possible_nll_gain_diagnostic_only": 2.4556356947869062e-11,
    "trial_gradient_max_diagnostic_only": 5.527477382586661e-12
  },
  "upstream_relative_errors": {
    "reconstruction": 0.0,
    "amplitude": 0.0,
    "delay": 0.0,
    "noise": 0.0,
    "expected_waveforms": 0.0,
    "noise_std": 0.0
  },
  "raw_data_sha256": "c0150fb6b5a831d6560677779a55cee397bdb7edf802f8c501c7f8bd3c6bcd7b",
  "environment": {
    "python": "3.11.11 (main, Mar 17 2025, 21:01:29) [MSC v.1943 64 bit (AMD64)]",
    "numpy": "2.4.6",
    "platform": "Windows-10-10.0.26200-SP0"
  }
}
```

## Final commands and gate status

- PASS primary method/software/data/license identity, live exact commit, full source/data hashes and equation/function/test mapping.
- PASS original full program on exact observations before local extraction, independently frozen numerical stages, separate complete repeat exit 0 and bit-identical outputs. **Canonical optimizer strict gtol: NOT MET / precision-loss false flag preserved**, independently investigated as detailed above.
- PASS explicit local acquisition/derivative/likelihood/Jacobian/scaling/BFGS/gauge/finite-M code; no domain/oracle package or cross-task import. Private import_audit.json records actual symbols/call roles. Source has no matplotlib.use; importing main preserves backend.
- PASS all native 50×267 measurements retained, exact author gain conversion, no synthetic benchmark data/target-derived calibration/hidden seed/GitHub URL leak. README alone provides forward model, units, scalars and full output contract.
- PASS `python -m src.main` and standalone `python evaluation/judge.py --prediction output/reconstruction.npz --report output/judge_report.json`; all required arrays pass with relative errors zero. CPU logs cli.log/judge.log are archived.
- PASS `python -m pytest test/ -v -p no:cacheprovider --basetemp <private-thz_noise/pytest_tmp>`: **25 passed**. Includes stage/physics/full pipeline, metric values, every required array's quality failure, missing/malformed/nonfinite input, finite enormous-prediction metric overflow, missing evaluator assets, truth priority, foreign cwd, invalid metric configurations and report-write failure. Overflow produces error JSON with empty metrics and exit 2 rather than a failed JSON write.
- PASS complete active `jupyter nbconvert --execute` with explicit clean imaging101-material kernelspec, timeout 1800 seconds; full reconstruction, real-prefix fits, curvature, additive ablation, independent Judge and fresh stationarity audit all execute. Canonical notebook contains zero errors and three unique scientific figure displays; no cache-preview result supplies reconstruction.
- PASS actual final PNG visual review for axes/units/errorbars/glyphs/legends and no hidden clipping; latent-minus-mean inset relocated/reopened to avoid covering the waveform. All three figures have private PNG/SVG/PDF and exact NPZ+fresh JSON sources/hashes.
- PASS README and both plans semantically reviewed against source signatures/defaults/shapes, Gaussian constant/objective normalization, log/physical variance conversion, Fourier signs, first-trace gauge, scale factors, gradient adjoints, stopping behavior and actual optional/standard flow.
- PASS final inventory: output contains only reconstruction NPZ and Judge report; no cache, duplicate source arrays without consumers, temporary executed notebook, source archive, standalone PDF/PNG/MAT/NPY or redundant report directory in the task. Minimal shared pinned requirements validated.
- Parent independent acceptance: PASSED; see the independently written parent acceptance and current parent audit below.

Notebook execution record:
```json
{
  "returncode": 0,
  "runtime_s": 12.032587799942121,
  "errors": 0,
  "figure_displays": 3
}
```


## Independent parent acceptance

Passed: parent independently ran 25 distinct tests in the cleaned shared environment and the standalone Judge from a foreign working directory. Source/public-interface/dependency/call-role/executed-notebook/inventory audit has no errors. Reviewed full latent waveform/50 amplitude-delay gauges and heteroscedastic additive/amplitude/timing likelihood, analytic gradient, finite-M bias scaling and all six outputs. Independently reproduced source false precision-loss stop is explicitly retained, with separate directional-gradient/SPD/Newton-decrement evidence of near-stationarity. Diagnostic curvature does not replace the submitted estimator; its coordinate-sign and extra finite-M error conventions are disclosed. All three actual scientific figures were opened, including prefix joint refits and additive-noise ablation. No figure or array was accepted on an unexecuted preview. Parent raw test/audit evidence and final file hashes are indexed under `D:\yan1\agent\AutoSkill\imaging-101-maintainer\research_batch3_20261020`. Final source hashes are refreshed after this acceptance-only note.


### Public source-link separation, 2026-10-29
Removed the direct public URL https://doi.org/10.5281/zenodo.4876388 from the solver README. Role: confirmed mixed MATLAB functions and measured-data archive. The scientific citation, dataset attribution and license remain; complete source/data provenance stays in this maintainer-only validation record. The released NPZ supplies every observation/calibration input, so no software archive is required by the solver contract.


## Current benchmark contract v2 (2026-10-29)

The v2 evaluator uses the independently evaluated mean heteroscedastic Gaussian negative log likelihood, with predeclared excess tolerance 1e-6 over an attainable fit. Reported noise amplitudes retain the sqrt(50/49) finite-sample correction; likelihood variances undo it. Expected waveforms and corrected noise standard deviations must satisfy the public periodic Fourier model to relative L2 1e-7. Period-equivalent delays are accepted. This certifies a conditional statistical fit, not an independently calibrated noiseless waveform. Independent synthetic drift/variance and periodic-alias controls pass; inflated consistent noise is rejected. The cleaned reconstruction algorithm is unchanged.

Earlier method-identity metrics are historical v1 reports, superseded for current benchmark acceptance. Full command stdout, native exits, controls and immutable old assets are indexed in the external benchmark_readiness_20261029 maintainer archive.
