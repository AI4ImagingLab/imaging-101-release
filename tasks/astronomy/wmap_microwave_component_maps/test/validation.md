# Validation and provenance

## Bound source, paper version and data

Published method: Stompor,Leach,Stivoli,Baccigalupi, Maximum likelihood algorithm for parametric component separation in cosmic microwave background experiments, MNRAS392,216–232(2009),DOI10.1111/j.1365-2966.2008.14023.x. Actually read paper version: arXiv0804.2645v2, revised2009-01-21, private19-page PDF `stompor2009_v2.pdf`,SHA256 `12d0051b351afea588e9697f5b580633dbf0075009c73c2ae70ca41a5ff9edd5`, obtained from https://arxiv.org/pdf/0804.2645v2 . Directly reviewed section3 equations2,4–8, deriving the fixed-spectrum generalized least-squares map-amplitude inverse, and equation1 for modified-blackbody scaling. This task selects the fixed-spectrum linear-amplitude case; it does not claim reproduction of the paper's free-spectral-parameter EBEX simulation,noise levels,sky patch or reported cosmological results. Foreground constants follow the selected FGBuster default model rather than replacing the paper's illustrative parameters without disclosure. Complete extracted text is private `stompor2009_v2.txt`.

Original author implementation https://github.com/fgbuster/fgbuster, exact live remote revision40ef92cd913411222b90434b6d68e24a682f5ffe verified2026-10-22 through GitHubAPI; complete metadata/commit response and source archive are private `fgbuster__fgbuster_remote.json` and `fgbuster__fgbuster.zip`. GPL3 original notices are preserved in local physics/solver docstrings and LICENSE. FGBuster is the author-maintained implementation of this formalism. Core spectra come from component_model.CMB,Dust,Synchrotron; amplitude inversion uses algebra.comp_sep fixed-matrix branch,algebra.Wd and its reduced SVD. No import stubs or edited upstream formulas are used for the oracle. Unmodified modules are loaded directly from the archived source with importlib to avoid importing unrelated optional full-sky plotting/generation dependencies.

Real observation source: NASA LAMBDA WMAP DR5 nine-year1degree-smoothed native HEALPix band temperature maps,v5,release2012-11-11. Public product page https://lambda.gsfc.nasa.gov/product/wmap/dr5/maps_band_smth_r9_i_9yr_get.html . Five exact URLs,source byte counts and SHA256 hashes are in private `wmap_downloads.json`; native source tables contain TEMPERATURE(float32,mK thermodynamic) and N_OBS. Only TEMPERATURE is bound. Native source NSIDE512,NESTED is preserved during nearest-pixel sampling of a fixed48x64direction patch: requested longitude linspace(70,102,64),latitude linspace(12,36,48),degrees. Private Astropy-HEALPix converts those requested directions to native indices; it does not interpolate or convolve observations. Data are native source float32 values promoted exactly to float64 and multiplied by1000to change mK to microkelvin. This does not claim added precision. The requested direction arrays locate the native pixel containing each direction; they are not relabeled exact native pixel-center positions. No foreground-reduced image,component template,hidden answer,seed or reconstructed target is supplied. Native source maps already share1degree smoothing; this operation and its pixel correlations are not attributed to local reconstruction. No new data generator is needed or shipped.

NASA's separate flat-Mollweide reprojected FITS files were initially inspected and explicitly rejected because their header warns of changed noise properties and discourages CMB analysis. They are not the observation source and remain only in the private research archive. The final task uses original native HEALPix tables. Numerical frequency values22.8,33,40.7,60.8,93.5GHz are the explicitly adopted conventional effective-frequency approximation,not a claim of exact9-year bandpass integration. Native file FREQ identifies band names; its archived values are not overwritten. README explains the monochromatic approximation and residual model discrepancies. Frequency-center drift/color corrections are not inferred from hidden sky components.

Bound public NPZ SHA256 `675f8a2a529acc211ba1f273a25ffc782782312b971664fbf6cf22591fce9369`. There is no independently measured component truth: evaluation/baseline_reference.npz contains three independently captured standard-method maps,all(1,48,64),float64,microkelvin,keys reconstruction(CMB),synchrotron22.8GHz,dust93.5GHz. They are not ground truth. All three quantities are independently evaluated. The upstream and fresh notebook files are separately named; each is consumed by the notebook comparisons and source-data visualization exports. Frozen fixture contains unchanged measurements/frequencies and oracle mixing matrix,maps,prediction,unit-noise amplitude covariance,residual. Its sole automated consumer is test/comparing_test/test_upstream.py.

## Original run before local extraction

Private `capture_wmap.py` reads the exact native sampled observations,then invokes unmodified original CMB/Dust/Synchrotron.eval and algebra.comp_sep(A,data,None,None,None). The fixed-spectrum column order is CMB,synchrotron22.8GHz,dust93.5GHz. Original parameters: synchrotron index−3,dust emissivity1.54,dust temperature20K,Tcmb2.7255K,thermodynamic units,equal channel weights,no iterative optimization or regularization. Source numerical constants are preserved: Planck h6.62607015e-34J s,Boltzmann k1.380649e-23J/K. Run command research-runtime/Scripts/python.exe capture_wmap.py;exit0,wall0.1179556s; original stdout `wmap_oracle.log`,arrays `wmap_oracle.npz`,environment/ranges/residuals `wmap_oracle.json`. The local method was implemented after these independent arrays were frozen. No cached paper notebook establishes this gate. Independent local maximum map difference=5.6843418860808015e-13microkelvin; output parity is measured,not inferred from import success.

Numerical environment: Python3.11.11,NumPy2.4.6,Windows CPU,threadsOMP/OPENBLAS/MKL1. Oracle-only dependencies live under private oracle_deps: Astropy8.0.2,Astropy-HEALPix2.0.1,SymPy,Numdifftools and their reader dependencies. Cleaned runtime depends only on NumPy and Matplotlib. No shared package was installed or upgraded. The actual executed notebook kernel environment is printed in its firstcell and stored privately as wmap_notebook_kernel_environment.json; executable/module paths verify shared research-runtime rather than relying only on the launcher. No GPU. Deterministic operations,no seed.

## Equation/function/test traceability

| Stage | Upstream source | Local function | Independent validation |
|---|---|---|---|
| Thermodynamic/Rayleigh-Jeans conversion g | component_model K_RJ2K_CMB | physics_model.rj_to_cmb | low-frequency limit;source mixing fixture |
| Fixed dust/synchrotron spectral laws | component_model.Dust,Synchrotron.eval | physics_model.mixing_matrix | source matrix fixture;unit reference-frequency amplitudes |
| Linear spectral acquisition d=A m | paperEq2;algebra._As_svd | physics_model.forward,adjoint | analytic noiseless multicomponent recovery;adjoint dot product |
| Map amplitudes GLS inverse | paperEq8;algebra.comp_sep,Wd | solvers.separate,reconstruct | independently captured full maps/prediction/covariance |
| Equal-channel residual objective | paperEq4 withN=I | normal-equation stationarity test | scaled A^T residual criterion and gain linearity |
| Unit-noise response covariance | algebra._invAtNA_svd | solvers.separate | source covariance and independent inverse(A^T A) |
| Metrics/Judge | benchmark explicit definitions | evaluation/judge.py;visualization.compare | known vectors,zero,constant rejection;all-component contract tests |

Parity tolerances: spectral matrix rtol2e-14,atol2e-15;maps/predictions rtol1e-11,atol2e-10microK;covariance rtol/atol1e-12. The current local/oracle difference is5.7e-13microK maximum. Operator/inverse analytic tests use2e-14 and adjoint identity1e-13. Stationarity is checked as max|A^T(A m-d)|/(norm(A)*maxpixelnorm(d))<5e-15, a relative float64 arithmetic criterion. An initial1e-12microK unscaled stationarity assertion detected1.65e-12at a few high-amplitude sky pixels; inspection showed roundoff below1e-15of acquisition scale,not numerical divergence. The test was corrected to the scale-aware criterion,with no change to solver or quality boundaries. A Windows locale-dependent README read was corrected to explicit UTF8. Neither issue was hidden as scientific success before correction.

## Metrics and physical interpretation

Each component has its own cosine NCC and dynamic-range NRMSE. Every NCC>=.999999 and every NRMSE<=1e-7is required. This nonzero numerical tolerance is appropriate to deterministic,well-conditioned float64 reruns and tightly rejects zero or doubled individual components; no tiny physical amplitude is concealed under another component's dynamic range. Source matrix condition number6.4394839405070314; observation RMS116.12279679619404microK,standard forward residual RMS2.8100211877644163microK. Model consistency is nonzero even when method parity is nearly exact. Unknown smoothed-noise covariance and foreground/bandpass model discrepancy prevent treating this RMS as unit-noise reduced chi-square or independent physical accuracy. Negative amplitudes remain visible and valid. No center crop,alignment,gain fitting,mean subtraction or clipping occurs. Judge checks all three arrays before emitting any metric,handles unavailable assets as evaluator errors,and emits finite JSON only.

## Executed validation commands

| Command | Exit | Wall seconds | Private log |
|---|---|---|---|
| `-m src.main` | 0 | 0.1320829999895068 | wmap_validation_0.log |
| `evaluation/judge.py --prediction output/reconstruction.npz --report output/judge_report.json` | 0 | 0.12772320001386106 | wmap_validation_1.log |
| `-m pytest test/ -v --basetemp D:\yan1\agent\AutoSkill\imaging-101-maintainer\research_batch6_20261022\groups\astro_earth\wmap_pytest_final` | 0 | 0.582737900011125 | wmap_validation_2.log |
| `-m jupyter nbconvert --to notebook --execute notebooks/wmap_microwave_component_maps.ipynb --output wmap_executed.ipynb --output-dir D:\yan1\agent\AutoSkill\imaging-101-maintainer\research_batch6_20261022\groups\astro_earth --ExecutePreprocessor.timeout=180` | 0 | 5.569053700004588 | wmap_validation_3.log |

Complete16-test suite passes. It covers frozen-source stages,all-output full pipeline,noiseless analytic recovery,adjoint identity,reference-frequency laws,temperature conversion limit,unit-noise covariance,stationarity,physical gain linearity,README constants,known metrics,valid reference,zero and doubled quantities,missing file/key,wrong shape,float32,complex,NaN,Infinity,missing assets,invalid threshold and report-writing failure. The private --basetemp path avoids an existing Windows shared-temp-root permission failure; no test is skipped. The CLI and independent evaluator are separately executed. The Judge reference paths resolve relative to its file and CLI paths relative to caller.

Canonical notebook contains actual executed outputs with six chapters:physical question,observations,physics,activeinverse,scientificcomparisons,limits. It freshly computes all3072directions,independently scores all3maps,projects every band back to observation space,computes the two-component ablation,nine spectral-index inverses,and five leave-one-band-out inverses. Six scientific figures are exported as private PNG/SVG/PDF plus numeric_sources.npz under figures/wmap_microwave_component_maps. Every plotted fresh result was computed. Full nbconvert exit0with180stimeout(actualunder6s including startup); allcodecelloutputs checked for errors. Temporary executed notebook is private; canonical outputs are retained. Actual kernel/environment printed and captured. Visual review is recorded in wmap_visual_review.json.

Plan/source semantic review covers GHz/Hz conversion,thermodynamic vs antenna units,reference-frequency normalization,dust spectral exponent1+beta,monochromatic/model limitation,equal weights,closed-form nature,no artificial iteration,no bounds/initialization,covariance meaning,and every actual callable signature/return shape. Source imports audited and archived in wmap_static_audit.json. Runtime contains no FGBuster,Astropy/HEALPix,upstream domain library,cross-task import,learned target weights or hidden-reference access. README+rawNPZ+requirements suffice for solving without maintainer assets; no standalone metadata JSON or public GitHubURL exists. LICENSE retained; source/capture/PDF/archive stay private.

## Completion gates

| Gate | Status | Evidence |
|---|---|---|
| Exact paper/revision/data hashes and equation/function mapping | passed | detailed provenance and private archive index |
| Original/local runs on exact bound observations | passed | original capture plus freshCLI and notebook |
| Independent stage parity and mathematical/physical checks | passed | source fixture and analytic tests |
| Constants,units,precision,coordinates preserved | passed | exact arrays and explicit mKconversion/native sampling |
| CLI and fresh Judge | passed | command records |
| Judge adversarial quality/input/evaluator contracts | passed |16-test suite |
| Self-contained public problem/data/environment | passed | completeREADME and four public arrays |
| No target/privateURL/metadata leakage | passed | raw observation-only data and static audit |
| Minimal cleaned runtime without domain libraries | passed | shared-runtime CLI/tests/NB |
| Local executable forward and inverse mathematics | passed | spectral laws/SVD/operator call-role audit |
| Tests | passed |16 passed |
| Notebook active reconstruction and visual comparisons | passed | executed canonical NB,6figures |
| Provenance/metrics/threshold consistency | passed | separate author/fresh labels;6independent metrics |
| README/plans/source consistency | passed | semantic review described above |
| Final inventory/license/temp cleanup | passed | evidence inventory and preservedLICENSE |

Parent visual review revision: spectral-law axis now uses exactly five plain-GHz tick labels and suppresses minor tick labels. Full active notebook re-executed successfully; private wmap_tick_revision.log/json records command and final source/notebook hashes. Figure2 re-exported in PNG/SVG/PDF. No scientific computation changed.
