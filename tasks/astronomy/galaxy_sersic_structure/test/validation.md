# Galaxy Sersic structure: authoritative validation

Status: all mandatory cleaning gates passed locally; frozen for independent parent review. This document distinguishes local success from physical truth and from parent acceptance. No actual intrinsic-galaxy truth exists. One earlier unexplained native process failure is retained below, with all later successful executions identified separately.

## Bound problem, source and rights

Unknowns are the pixel-integrated intrinsic extended galaxy light distribution and eight physical Sersic/sky parameters in each of g/r/z, with independently calibrated PSFs and inverse variances. Selected method: original PixelRenderer, Gaussian likelihood, independent MAP with flat sky and explicit manual priors. This is different from existing point-source stellar photometry and source-map inversions. It does not claim to reproduce the author's separate Student-T joint multiband posterior demonstration.

Original repository https://github.com/pysersic/pysersic, remote commit `12978a7a5eeee8c0cba3eadda0b64c25257e3ac2` verified through GitHub commit metadata and revision-bound source archive before extraction; see sources.json and complete private archive `archives/pysersic-12978a7a5eeee8c0cba3eadda0b64c25257e3ac2`. No remote freshness beyond that verification is claimed. Source code is MIT; task LICENSE preserves original notice. Primary paper: Imad Pasha and Tim B. Miller, *PySersic: A Tool for Bayesian Inference of Galaxy Structural Parameters*, JOSS 8(89), 5703, published 15 September 2023, DOI https://doi.org/10.21105/joss.05703. Full paper PDF/HTML/text and rendered first page remain private.

Real author-supplied DECaLS/Legacy Surveys DR9 south FITS `cutout_59.1067_-30.4848.fits` and `copsf_59.1067_-30.4848.fits` are bundled in that exact source revision. All g/r/z bands use the author's native central rows/columns68:188 of the256x256 image,120x120 pixels,0.262arcsec/pixel; no resize or synthetic observation. Each PSF is63x63. Official Legacy Surveys policy https://www.legacysurvey.org/acknowledgment/ licenses images CC BY4.0 and requests credit `Legacy Surveys / D. Lang (Perimeter Institute)`, retained in README and actual image figures. Units and calibrated-image conventions were checked against https://www.legacysurvey.org/dr9/description/ and official coadd image documentation. Private archived rights/unit/stack pages are hashed below. PSFs are legitimate author-supplied observation calibration, not inferred from hidden structure. Original FITS hashes are in oracle metadata and index below.

The public mask is generated once from observed r-band data by original SEP3RMS segmentation, retaining segment at(60,60) and masking other nonzero segments; same mask across bands. It contains no hidden answer. All observations/inverse variances/PSFs/mask are supplied as batch-first NPZ arrays. No hidden target, fitted calibration or synthetic seed is public. README scalar metadata matches explicit source constants. An independent visible mask defines excluded pixels, not a post-hoc error mask.

## Original run before local implementation

Private oracle environment `D:\yan1\agent\AutoSkill\imaging-101-maintainer\research_batch3_20261020\batch3\astro_earth_physics\oracle-env` contains original PySersic dependencies (JAX, NumPyro, Astropy and SEP); none is installed into or imported by cleaned runtime. Command: oracle-env/Scripts/python.exe capture_galaxy.py. This invokes unmodified original FitSingle.find_MAP, PixelRenderer and gaussian_loss on the exact bundled three observations and calibrations, with seeds20261020/21/22. Original source files were not edited. Exit0, runtime 29.025781100s, Windows CPU, Python3.11.11 (main, Mar 17 2025, 21:01:29) [MSC v.1943 64 bit (AMD64)], NumPy2.4.6, JAX0.10.2, float32, OMP/OPENBLAS/MKL threads1, XLA preallocation disabled. `galaxy_oracle.log` records actual optimizer output; no cached author notebook result substitutes for it. Captured original arrays are galaxy_oracle.npz, with stages in galaxy_stages.npz produced by a second isolated original renderer/Gaussian-loss/JAX value_and_grad invocation. Those scripts and `galaxy_acceptance_before_extraction.json` existed before local mathematics. Original figure/training demonstrations outside this bound independent-MAP problem are explicitly outside scope.

Manual physical priors were fixed before original execution: xc,yc~Normal(60,5pixel), sky~Normal(0,.01nanomaggies/pixel), uniform flux[0,300],major re[.5,30],n[.65,8],ellipticity[0,.9],theta[0,pi]. The sky prior is in capture_galaxy.py; oracle JSON's priors subdictionary lists the seven profile quantities only. Gaussian negative loglikelihood includes half squared standardized residual, log RMS and half log2pi for every unmasked pixel. Fixed-stage objective adds weak position/sky quadratic priors, omitting irrelevant prior constants in both independently computed stage comparisons. Renderer uses12x12 central pixels with12x12 Gauss-Legendre quadrature, full120x120 FFT convolution and original approximate PSF phase constant3.1415.

Independent original seeds20261030/31/32 were rerun by rerun_galaxy_original.py, exit0; galaxy_rerun.npz/json/log preserve them. Maximum intrinsic DR-NRMSE between original seeds is1.61567e-7. Maximum parameter differences in order(xc,yc,F,re,n,e,theta,sky) were approximately(3.81e-6,0,8.39e-5,3.58e-5,2.50e-5,1.31e-6,5.96e-7,1.20e-8). These measure original stochastic initialization variability, not local accuracy or ground truth.

## Mathematical discrepancies and adaptations

The rendered primary JOSS paper's introductory Sersic display prints a positive exponent without the required negative half-light constant. Literal use would grow with radius and fail the flux integral. Original rendering.py/render_sersic_2d uses the physically normalized decaying profile exp[-bn(z^(1/n)-1)], which the local implementation extracts. This is an explicit paper/code discrepancy; we do not claim literal parity with that printed introductory expression. Independent continuous radial integration verifies total flux and half-light fraction for n=1, while finite differences verify all eight gradients. Original analytic bn approximation, not a replacement special-function inverse, is preserved.

Original PSF centering uses3.1415 rather than exact pi. Its63pixel kernel phase implies a residual axis shift31*(1-3.1415/pi)=0.000914269162267pixel. This quantified approximation is retained for source numerical parity and declared publicly; it is not asserted to be an exact centering identity. PSF transfer stages and total discrete convolution flux are independently checked. No hidden image alignment corrects this difference.

Standard local initialization now preserves the exact original SVI.init optimizer-unconstrained prior median draws at the original three seeds. capture_galaxy_initialization.py independently invokes original AutoDelta/SVI.init and one stable_update, without local imports or changing original source. galaxy_initialization.json/log preserve all24 latent/physical starting values and first warm optimizer state. These target-independent pure-prior constants are embedded as INITIAL_LATENT in local source; no fitted answer or hidden calibration enters initialization. A new independent test verifies exact latent constants, physical transformations and all24 first-warm-step latent outputs against original fixture upstream_initialization.npz. Warm step is restored, round0 retains its pre-step best loss, and rounds1/2 alone reset best loss to infinity. All comparisons retain pre-step loss/post-step best state just as original stable_update. The stop iteration is excluded from original-style history. Prior constants and optimizer states are not estimated from the target.

Earlier local59.5geometric-center runs and the omitted warm update were rejected during parent review as failing initialization fidelity; abandoned_geometric_initialization privately preserves their logs and figures. They are not canonical results and are not completion evidence for the current source. Same original initialization is supplied to every measured and corresponding hypothetical-PSF band fit. Float32 generic backend/order differences can still yield finite convergence differences; actual final differences are recorded in fresh_measurements.json, not rounded or aligned away.

## Equation/stage traceability and independent tolerances

| Mathematical stage | Exact upstream file/function | Local executable function | Independent case/consumer |
|---|---|---|---|
| Normalized decaying Sersic, elliptical rotation, analytic bn | pysersic/rendering.py render_sersic_2d | physics_model.sersic | fixed original images; continuous total flux/half-light integral |
| Central pixel quadrature | rendering.py PixelRenderer.render_int_sersic | PixelModel.intrinsic | fixed three-band original intrinsic stages |
| Measured PSF padding, Fourier shift and periodic convolution | rendering.py PixelRenderer and renderer PSF_fft | PixelModel.__init__/observed | all3 original complex transfer/convolved stages; independent flux preservation |
| Masked Gaussian density and physical prior | loss.py gaussian_loss; original priors.py | physics_model.objective | original objective and all8 JAX gradients; independent central finite differences |
| Affine/sigmoid supports | original priors.py/AutoDelta support transforms | solvers.physical_parameters | full original MAP comparisons and all declared physical output columns |
| Three-round Adam and best-state/patience | pysersic.py FitSingle.find_MAP and training utilities | solvers.fit_band/reconstruct | complete independent original three-band arrays, fresh full-pipeline test |
| Metrics/output | benchmark-defined, not paper scores | independent evaluation/judge.py; visualization.metrics | known orthogonal vectors, allband/parameter failures, strict contract cases |

Frozen before extraction: fixed image/PSF rtol2e-5,atol2e-7; gradient rtol5e-4,atol.01; objective rtol2e-5,atol.01; final centroid.02pixel,flux.5%relative, image DR-NRMSE.001. Final additional physical column tolerances re/n.005,e/theta.002,sky2e-6 are explicit in metrics.json, substantially tighter than measurement uncertainty but permit finite backend/initialization convergence. Images require uncentered NCC>=.999 and RMS error/reference dynamic range<=.001 in every band. All24parameters are separately scored with no gauge alignment, rescaling, omitted band or magnitude coercion. Baseline is independently reproduced standard method, never truth; no ground_truth.npz is invented. Nonzero numerical tolerances, seed reruns, fixed-stage gradients and observation residual checks are used instead of vacuous self-comparison thresholds.

## Fresh cleaned runs and result provenance

Validated shared research-runtime Python is documented in galaxy_clean_environment.json: NumPy2.4.6, Torch2.9.1+cu128 used on CPU, Matplotlib3.11.1, one numerical thread; Torch build/backend configuration and NumPy backend are archived. requirements.txt is the runtime subset; SciPy is an independent quadrature test development dependency. No upstream/domain library or cross-task import occurs in source, tests, Judge or notebook; upstream files/capture scripts are never imported by task runtime. NumPy supplies arrays/FFT/Legendre nodes, Torch supplies generic FFT/autodiff/Adam and Matplotlib supplies figures. Core profile, quadrature, acquisition convolution, objective, support transformations, initialization and optimizer schedule remain local readable code.

Standard python -m src.main: exit0,54.753619s. Standalone Judge on fresh output: exit0. Full python -m pytest test/ -v: all13 passed,56.557024s. Full canonical active notebook/export orchestration: exit0,223.283987s with600s/cell timeout. Before these final fidelity runs, standalone Judge promoted metric accumulation to float64 to reject overflow and ensure finite JSON for valid large submissions; all8 Judge tests passed in1.424229s, including1e30finitefloat32 prediction, negative/missing thresholds, missing keys/files, wrong shape/type, NaN/Inf, every required parameter/band corruption, actual-truth precedence, missing hidden assets, foreign cwd and report-write failure. Solver mathematics unchanged. Final original-initialization/warm-update and label-corrected notebook rerun executed all9fits and current Judge, exported actual PNG/SVG/PDF directly from live figures, exit0,223.038711s. Canonical executed notebook retains these actual outputs; a private final export-only cell is removed from released notebook because it contains maintainer paths. The relative-root setup is restored without changing mathematics.

All three baseline fits and six hypothetical PSF-variant fits are freshly computed in the notebook. Final raw evidence is final_figures/galaxy_sersic_structure/numerical_trace.npz and fresh_measurements.json, with actual histories, original/fresh parameters/maps/convolved outputs, raw calibration, radial profiles, variant PSFs and full variant refits. Original runtime29.0258s and fresh baseline runtime53.056120s are labeled separately. Perband intrinsic comparison measurements are [{"ncc": 0.999999999732583, "nrmse": 1.453683133791556e-06}, {"ncc": 0.9999999863472755, "nrmse": 8.716335021060914e-06}, {"ncc": 0.9999999996752418, "nrmse": 1.1959933150254598e-06}]. All24 absolute parameter differences are [[[0.0, 3.814697265625e-06, 0.0001697540283203125, 6.151199340820312e-05, 5.0127506256103516e-05, 1.6391277313232422e-06, 4.76837158203125e-07, 4.0068698581308126e-08], [3.814697265625e-06, 4.1961669921875e-05, 0.0023136138916015625, 0.0006012916564941406, 0.00041431188583374023, 1.627206802368164e-05, 7.152557373046875e-07, 1.6562262317165732e-07], [3.814697265625e-06, 3.814697265625e-06, 0.000576019287109375, 0.00012111663818359375, 7.236003875732422e-05, 2.0265579223632812e-06, 2.384185791015625e-07, 4.1341991163790226e-08]]]. Unmasked standardized residual mean squares are[0.900238573551178, 0.9388998746871948, 0.8146639466285706]; public observed-neighbor masked fraction0.0455555555556. These are measurement-fit diagnostics, not proof of intrinsic truth. Source smoothness/structured residuals, diagonal coadd noise, finite quadrature region and circular image boundary remain limitations. Notebook/visualization and Judge both accumulate metrics in float64 using the same declared cosine formula, independently checked against known cases. Neither applies mean subtraction, clipping or image scaling to evaluation.

Five actual exported PNGs were opened and visually checked: measured/original/fresh/residual3x4; intrinsic/full-scale difference2x3; actual radial profiles and residual distributions; real optimization and wavelength-dependent structural parameters; complete seeing/PSF refit sensitivity. Physical arcsec axes, nanomaggy units, observed mask, shared perband image range99.5%, residual display±5RMS, full symmetric numerical differences and image credit are explicit. Long intrinsic titles were shortened after first inspection to avoid colorbar overlap, then actively re-executed. The sensitivity alternatives are clearly hypothetical deltaPSF/extra0.5pixelblur calibrations, not substituted real measurements or claimed truth.

## Earlier native failure, investigation and limits

One initial standard CLI attempt exited3221225477(0xC0000005) after39.2367571s with no Python traceback or diagnostic stderr; galaxy_native_failure_0.log is retained even though empty except the stderr separator. No mathematical change was made based on this unconfirmed cause. A faulthandler-enabled private perband diagnostic then completed all3fits (g9490updates,r4321,z2953), followed by successful complete standard CLI, all13tests, full active notebook, independent all9fit figure export, and final active notebook/export. The native crash did not recur in those executions. Its root cause is unresolved; we do not claim it was fixed or attribute it to memory without evidence. Resource scheduling used numerical threads1, no simultaneous own heavy process, and parent-controlled physical/commit-free memory>=6GiB. Parent independent acceptance must consider this transparently recorded platform incident.

## Semantic/allowlist/file review

README and both plans were reviewed against actual source signatures, caller flow, Gaussian signs/log constants, elliptical flux normalization, pixel coordinate units, priors, latent transforms, Adam moment/state restoration and stopping, FFT sign/padding, output physical order and every Judge tolerance. Plans explicitly identify the initialization adaptation, source approximation and historical broader examples. README alone plus raw NPZ/runtime subset defines the physical problem; no hidden array/target-derived calibration is required. Visible README and requirements contain no GitHub URL; all public raw keys appear in the exact four-column batch-first data table. All required outputs are in one NPZ. Hidden arrays have actual consumers: upstream_stages.npz→stage tests and original observed notebook comparison; baseline_reference.npz→Judge/full pipeline/notebook; upstream_measurements.json→original identity/runtime display. No unused fixture/reference copy or synthetic generator is retained. Main/library import never selects Matplotlib backend.

Final task inventory follows the canonical layout. Cache directories are moved privately before freezing, and no temporary executed notebooks/screenshots/source archive/PDF remain in task. All five PNG/SVG/PDF and exact numeric sources are indexed in manifest_galaxy.json. Parent may append independent acceptance notes; resulting final task inventory is then owned by parent's final audit.

## Mandatory gate ledger

| Gate | Status | Evidence |
|---|---|---|
| Paper/revision/data hashes and equation mapping | passed | above; private exact commit/archive and hashes below |
| Exact original and cleaned observations, independent original execution | passed | capture scripts/oracle log/NPZ; complete CLI logs |
| Upstream fixture parity and independent physics | passed |13tests, stage gradients, analytic integrals/finite differences |
| Constants/units/conventions | passed | source-plan review, exact original initialization/warm update, explicit tiny phase approximation |
| Standard CLI and fresh standalone Judge | passed | galaxy_fidelity_runs and last_judge_checks |
| Full output/strict invalid-input/evaluator contracts | passed |8Judge tests including foreign cwd/reportwrite |
| Public allowlist sufficiency and no hidden/GitHub leak | passed | visible key/metadata and source/import audit |
| Minimal environment/local mathematics | passed | clean environment, no domain imports |
| Full tests | passed |13passed including original initialization/warm-update parity |
| Active reconstruction notebook and computed comparisons | passed | prior archived nbconvert plus final corrected active notebook/export complete rerun |
| Historical/original/fresh identity, metrics and limits | passed | independent labels/no fabricated truth |
| Semantic README/plans/math/signature review | passed | review above |
| Final inventory/rights/actual figure QA | passed | task hash inventory/private figure manifest |

The earlier native failure remains a documented unresolved cause, not an unexecuted mandatory gate or a claimed fixed defect; successful subsequent exact full runs are the completion evidence.

## Private archive index and SHA256

Archive root: `D:\yan1\agent\AutoSkill\imaging-101-maintainer\research_batch3_20261020\batch3\astro_earth_physics`. Full source checkout and original FITS stay under archives/pysersic-12978a7a5eeee8c0cba3eadda0b64c25257e3ac2; private capture scripts and logs never enter the task. The following list freezes consumed provenance artifacts, not every failed candidate in the group's broader research archive.

| Private artifact | SHA256 |
|---|---|
| sources.json | 58b35d7b442079c90e016e7456c07e3856686ba5100a6b9f4ed0c1079dbe3fe5 |
| galaxy_paper.pdf | dcd71d7bea4ff88f4091c0dd3080f4ada90a8f850b6d8a0ecfc8a6151440d647 |
| galaxy_paper.html | cd62813f489f4905187edf8dcc4292129c445504239d0d87108759e77b61b7a9 |
| galaxy_paper_text.txt | b556d2463ca270b364c03a460c968109c74bff0f09f475ad1c698f0645f8e564 |
| legacy_image_rights.html | adb1c5b41ad8aee7544a23d1b499b317d97c82222b4f69f65efa3ad13f6a9a1c |
| legacy_image_units.html | 593edb3efb130286523ef2426370ea9cbde4cc7dc2d5b511f03282340cdc6b24 |
| legacy_image_stacks.html | d0b51d66529cb4c62db7e8ae1df22d6976879f46dcd62b4e6993729b42674c85 |
| capture_galaxy.py | e42a65fed5b6e3876b3ada09ee4352c6de7265b89d0d0122309bb84415019473 |
| capture_galaxy_stages.py | dea19927dbc1e86f11933b432bb6cffb476622eabc0d663c66f35a5488131037 |
| galaxy_oracle.npz | e05b308854b73d44b1cb24ca57e52abc580247a1873e1a3eacbad2bf47354fb6 |
| galaxy_oracle.json | f437a1ac761bb58811ad6bb3ca4f5046f518199c1a5b7cb6751e56d5df7519f3 |
| galaxy_oracle.log | 6d8b06c035d6a59499e5336dc2e252f98ccd31bba304c6264a9d3b455c0e8ba2 |
| galaxy_stages.npz | b3d360869a0fa867e884ce293bd6796c956f42a40570505c01d3584fa5199143 |
| galaxy_acceptance_before_extraction.json | d58b51ac5027a441bea01af2effc42b89c4b9ef12450bf53aa3bfb1140ae6612 |
| rerun_galaxy_original.py | 04c221f7426ea96fb4fec9bfe715b062c45dcae12ae56877b2cc49ce274f5e0b |
| galaxy_rerun.npz | a759c910b3c90f856a7b1aef181fc15e14ea649fad69212bd4868431fd8c45c4 |
| galaxy_rerun.json | ba3e467ee6d9a758fb206a6e8de08f2868547db539b896fa090658695db68381 |
| galaxy_rerun.log | c2e1686b19d5efebeb4f3e5d842fe9831a2fda062c57f79b5c6147067b12203e |
| galaxy_native_failure_0.log | 1643f6abc16fad2d66c6268cab1c723e3fcd3a05afd75615b8fb99e26fd58011 |
| debug_galaxy.py | 35d71865fe5a5353c77b3115d43827077d021290176302b22bf57b20c6cf7f14 |
| galaxy_final_runs.json | 6682b0c66ac00b8f4edd43def8ae06f129ab43efeee9ca9ee914c9d7982027ad |
| galaxy_final_0.log | 1643f6abc16fad2d66c6268cab1c723e3fcd3a05afd75615b8fb99e26fd58011 |
| galaxy_final_1.log | 1643f6abc16fad2d66c6268cab1c723e3fcd3a05afd75615b8fb99e26fd58011 |
| galaxy_final_2.log | a688436a67475aa471a1eb68b9c24e55e4240988b1cbd4670a84e481907451e6 |
| galaxy_final_3.log | d4782628e9815586d4c0ef2e2d06a40eddfa81eb0630df3c88442d185002d7d8 |
| galaxy_last_judge_checks.json | 137bf41d6efe6fb2c2fae828671af77d005cb2aefa098d2997c15d600eb2783f |
| galaxy_last_judge_0.log | 1643f6abc16fad2d66c6268cab1c723e3fcd3a05afd75615b8fb99e26fd58011 |
| galaxy_last_judge_1.log | 31d9127593a578f2a64faecd16532c7a49c6e27798a76627eb068fb2372b16ea |
| finalize_galaxy_notebook.py | c058814d40b1460304f2e9d5bbee93d2ad0d175a1cb9b8c02a70f57268705166 |
| galaxy_visual_final_nb.json | d8983af2b4d49e49b54738ce43710e923b400844414bb6a3f3262aa4947b2ebe |
| galaxy_visual_final_nb.log | 27b152f2785801f84b5fc7acd490d14545db6ad28739b43f811c1b714c11d62e |
| galaxy_visual_contract.md | 0be6f9b48d9b36b8a0aebeee743555dbf7008170d26a6aa4fa599cadd5f52398 |
| manifest_galaxy.json | b9af18ec269e5518aa77496d29212fe5bb37a56fc34dea50edff6cc536bf9af5 |
| archives\pysersic-12978a7a5eeee8c0cba3eadda0b64c25257e3ac2\examples\examp_gals\cutout_59.1067_-30.4848.fits | 0758cfadfa77774461b8c5fd456a96e03f54e9dfd843046a0fb89fd45d180122 |
| archives\pysersic-12978a7a5eeee8c0cba3eadda0b64c25257e3ac2\examples\examp_gals\copsf_59.1067_-30.4848.fits | da78a138b90396bb95fef514ff22031414e6a7151d46eed980939c44c00b870f |
| archives\pysersic-12978a7a5eeee8c0cba3eadda0b64c25257e3ac2\LICENSE | 0a24c8ee20a51b522e446410dff149aa86611d41c6adaf5a9efa30bed7e69ba7 |
| capture_galaxy_initialization.py | c6846e2c13f393eec90d36a3d0f066e8bba4c22070143533d2726a5c88ca910f |
| galaxy_initialization.json | ede447a361cd4dce0e7ad3eebf5f7fdceb16afeeea4be6ddba166ff2b6ad791a |
| galaxy_initialization.log | 485f9c7217c3bb48aed72f12e9babbefc5b68a6e8e42d50e4275338e6bbf0877 |
| prepare_galaxy_fidelity.py | cf82b8e72b53b9410173ec67dec395871c0b7dae5008bb7f324ceec0dfbe3970 |
| validate_galaxy_fidelity.py | ca21d8c9eb9542314be3ce23d9342990ed4059f22f29e7ae2a7a4f5250fef9fe |
| galaxy_fidelity_runs.json | 3c345874cf43a78cc1b30b1de66301d2ee243daee75f781e300865c5a0b879a6 |
| galaxy_fidelity_0.log | 1643f6abc16fad2d66c6268cab1c723e3fcd3a05afd75615b8fb99e26fd58011 |
| galaxy_fidelity_1.log | 1643f6abc16fad2d66c6268cab1c723e3fcd3a05afd75615b8fb99e26fd58011 |
| galaxy_fidelity_2.log | 20883386089f84e12d564dc1c8f8dd977dd81afccc5af727b34e1af06c676b54 |
| galaxy_fidelity_3.log | 56b1ef2f18efa0390ef37e4482c6b32df861c4574b9038708bb7daf48be5b648 |
| pysersic.zip | f924e290d6656e0a5e3d9e88d8d2a27b0ee5b4b722959b16c873388a3f014bef |
| galaxy_clean_environment.json | df00c7fc301db41b9bfe6dd02dd01a3904898e3ff10883b6544a87d9b8f971c1 |

## Bound released/hidden arrays

| Task array/evidence | SHA256 |
|---|---|
| data\raw_data.npz | ddd95d20b2d78da129e0e7e611563678ab5695d6de4d9cfc30c7dffd68d759a4 |
| test\fixtures\upstream_stages.npz | 82454549de4ce8e979c439828c7b15755f334d249141c9ffdd9b9b815dcb0542 |
| evaluation\baseline_reference.npz | 62eadefe20ba8706b9198c134333028a2c6ea35c30a3fb9a45ff6966c153407f |
| notebooks\reference_outputs\upstream_measurements.json | f437a1ac761bb58811ad6bb3ca4f5046f518199c1a5b7cb6751e56d5df7519f3 |
| output\reconstruction.npz | 148e422af8cea64828893c440779b1ae7ac558cfad565c35d58781bf3af16dd5 |

Current independent initialization fixture SHA256: `979238b349997c947830a1b891110202d480afba3f89e67d769b1e3bf40b5468`. Consumed by test/comparing_test/test_initialization.py; original private capture provenance above.


## Independent parent acceptance

Passed: parent independently ran 13 distinct tests in the cleaned shared environment and the standalone Judge from a foreign working directory. Source/public-interface/dependency/call-role/executed-notebook/inventory audit has no errors. Reviewed three-band pixel-integrated flux-normalized Sersic rendering, central Gauss quadrature, calibrated Fourier PSF conventions, observed-only neighbor mask, likelihood/physical priors and all24 parameters. Parent rejected the changed geometric-center initialization: the shipped code now preserves independently captured original pure-prior seeds, exact initial unconstrained variables, first warm Adam update and original best-state/patience/history semantics; extra independent fixtures and full tests verify them. The actual preprint/JOSS erroneous profile printing and source3.1415 centering approximation are independently quantified, not silently copied or called exact. All five actual final figures were opened, including measured normalized residuals and six actually refitted PSF controls. Earlier isolated native failure and abandoned adaptation remain privately documented; repeated complete corrected commands passed without claiming an identified OS fix. No figure or array was accepted on an unexecuted preview. Parent raw test/audit evidence and final file hashes are indexed under `D:\yan1\agent\AutoSkill\imaging-101-maintainer\research_batch3_20261020`. Final source hashes are refreshed after this acceptance-only note.
