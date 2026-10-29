# Provenance and completed validation

## Binding and distinction

Data class: **author_real_processed**. All 84 measured momentum-transfer points in each of the ++ and -- polarized channels are bound, without synthetic substitution. NIST identifies these as real NCNR measurements and supplies the processed reflectivity file. Its reduction header records source runs 171115B_30A45607 through45620, background runs, polarization calibration45621, footprint/background corrections, wavelength4.75 Angstrom and relative wavelength FWHM0.029999999329447746. The acquisition archive path contains ngd201801/24167. These are processed reflectivities with propagated standard errors, not detector-count Poisson observations.

This problem differs from the old unpolarized PLATYPUS silicon/oxide/film/heavy-water nuclear-SLD task: opposite neutron-spin potentials contain nuclear SLD plus/minus magnetic SLD. Their contrast identifies a magnetization depth profile and suppressed interfacial magnetic layer, while jointly fitting nuclear structure and beam nuisance parameters. Neither a different sample nor a new optimizer is the basis of novelty.

## Primary sources, versions and permissions

Wisser et al., *Ultrathin interfacial layer with suppressed room temperature magnetization in magnesium aluminum ferrite thin films*, Applied Physics Letters115,132404 (2019), publisher version, DOI https://doi.org/10.1063/1.5111326. NIST-hosted primary PDF: https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=928218 (archive/pnr_wisser2019.pdf). Collinear magnetic propagation is supported by Maranville et al., arXiv1510.07912v2, https://arxiv.org/abs/1510.07912v2.

Actual author case: https://www.nist.gov/ncnr/magnetically-dead-layers-spinel-films ; acquisition/model context: https://www.nist.gov/ncnr/fitting-polarized-neutron-reflectometry-data-beginners-guide . Exact downloads https://www.nist.gov/document/spinelfilmzip and https://www.nist.gov/document/spineldeadlayerszip are archived with their original MAFO_Saturated.refl, MAFO_Saturated.py and MAFO_DeadLayers.py. NIST explicitly permits teaching simplifications and warns some constraints normally use supplementary measurements. Accordingly the trial-and-error additional broadening0.026degree is an unknown, bounded[0,.08], shared by both channels, not privileged calibration. Known substrate SLD5.377 is composition/density calibration; all film properties remain fitted. The case represents the paper's material system and uniform/dead-layer scientific comparison, not a claim to reproduce its exact MCMC posterior or all applied-field datasets.

Original solver https://github.com/reflectometry/refl1d , executed v1.0.1, remotely verified commit3b325710060910785f2f71012d5613b5b8a0689c. Optimizer https://github.com/bumps/bumps , executed v1.0.4, remotely verified commit6790b82c02a8b544d63289975c2a2f88b7c1437f. Verification responses and wheel hashes: archive/pnr_source_versions.json. These are executed release versions, not current branch-head claims. Original installed source is archived under archive/pnr_sources, with SHA256 index archive/pnr_asset_hashes.json. Refl1D BSD3-clause and Bumps simplex original permission notices are retained in LICENSE and adapted source. NIST case attribution to Alexander Grutter/NCNR is retained. No upstream package is imported by shipped runtime, tests or notebook.

## Original execution and frozen criteria

Command from worker root: oracle-env/Scripts/python.exe archive/run_pnr.py. The capture loads the unmodified NIST model and native Refl1D/Bumps implementation; the sole model-configuration change fits additional broadening in both comparison models. NIST recommends Nelder-Mead or differential evolution, so native bounded simplex is a faithful tutorial option. It uses radius.15, range-normalized xtol1e-7, objective-spread ftol1e-9, maxiter15000 and original starting parameters; no random seed or stochastic steps occur. CPU single-thread OMP/MKL/OPENBLAS/NUMBA=1. Python/platform/package/backend records: archive/pnr_oracle_environment.txt and pnr_clean_environment.json. Source capture, numeric output, run report and independent preimplementation criteria are archive/run_pnr.py, pnr_oracle.npz, pnr_native_run.json and pnr_preimplementation_protocol.md. Exit0; uniform runtime0.5095807s, dead-layer0.5168645s. Independent repeat returned identical parameters. Both complete models were executed on all168 bound measurements.

Upstream Bumps Result status has an inverted status-to-string mapping: it prints Aborted when the convergence branch returns status0. The capture callback independently verifies final simplex size and objective spread: uniform8.9377e-8/8.3560e-12; dead9.9616e-8/6.2812e-12, each within the selected limits. Local reporting corrects only this status label, preserving numerical updates. This is an investigated upstream reporting error, not silently claimed unchanged status parity.

Independent tolerances were frozen before local extraction: forward rtol1e-8/atol1e-12, objective relative1e-8, full parameter maximum bound-range error1e-3 and objective absolute1e-4. Local collinear Parratt recursion was independently derived from decoupled spin potentials. Initial small discrepancy was traced to the original magnetic kernel's positive imaginary guard1e-10 in SLD units; preserving it gives forward relative disagreement about4.4e-14, absolute about1e-15. No tolerance was relaxed. Local full-fit range errors are4.0107e-8(uniform),2.2481e-8(dead); reduced chi-square1.6583749063928799 and1.4722735348804543. Native values are1.658374906392904 and1.4722735348803906. Runtime about1.02s and.97s respectively; exact measurements archived in pnr_clean_parity.json.

## Equation / implementation / independent case mapping

| Mathematical stage | Original source/function | Local implementation | Validation |
|---|---|---|---|
| Opposite collinear nuclear/magnetic potentials and interface reflection | refl1d/sample/reflectivity.py magnetic_amplitude; lib magnetic kernels | physics_model.layer_reflectivity | Both native spin curves, analytic Fresnel limit, zero-magnetization equality and reversal |
| Nuclear and suppressed-layer magnetic slab construction | MAFO_DeadLayers.py; Refl1D slab rendering | physics_model.layer_stack,unpack | Exact native slab SLD/width/roughness arrays |
| Shifted angles and Gaussian instrumental/additional resolution | refl1d/probe/probe.py; lib resolution convolution | physics_model.predictions,gaussian_resolution | Native complete168-point forward; normalized constant/linear integration |
| Gaussian-interface depth profiles | refl1d/profile.py build_profile | physics_model.depth_profiles | Independently sampled native profile; integrated magnetic-layer moment |
| Half summed standardized squared residual | bumps FitProblem.nllf | solvers.fit_model objective | Native residual and both full minima |
| Bounded simplex reflection/expansion/contraction/shrink | bumps/simplex.py simplex | solvers.bounded_simplex | Both full original inversions, independent convex optimum and nonconvergence |
| Spin asymmetry and propagated uncertainty | Algebra from opposite-spin observations | visualization.spin_asymmetry,asymmetry_uncertainty | Independently known hand-calculated values |
| Output physicality, profile agreement, forward chi-square | Independent evaluator mathematics | evaluation/judge.py | Known parity, float32, malformed/poor/hidden-reference and CLI contracts |

No paper equation number substitutes for executable physics. Detailed actual formula definitions and interfaces are in plan/approach.md and plan/design.md; they were checked against source signs, factors, Angstrom/degree units, resolution FWHM conversion, axis ordering, simplex decisions and full notebook caller flow.

## Reference identity, metric choices and consumers

No independent magnetization truth exists. evaluation/baseline_reference.npz contains native fitted parameters and native build_profile samples on the declared521-depth grid, produced by archive/freeze_pnr_references.py using upstream code, not the cleaned profile. notebooks/reference_outputs/upstream.npz contains the original uniform/dead results for figure comparisons. notebooks/reference_outputs/cleaned.npz contains newly computed notebook results. test/fixtures/upstream.npz contains native output-prefixed intermediate/final arrays consumed by comparing_test/test_oracle.py. Original raw redundant captures remain private, not additional shipped fixtures.

All metrics evaluate both required output arrays. Parameter error is maximum difference divided by declared physical ranges. Nuclear/magnetic NRMSE use their own reference dynamic ranges. Profile/parameter consistency is maximum per-channel dynamic-range-normalized discrepancy from the declared profile formula. Forward reduced chi-square uses168 measured uncertainties and159 degrees of freedom. No alignment, rescaling, crop or hidden calibration. Parameter tolerance5e-4 tightens the frozen1e-3 parity criterion; both profile NRMSE limits.002 and consistency1e-4 are nonzero numerical agreement tolerances, not assertions of measured physical truth. Forward limit1.5458872116244102 is1.05 times the native value. Float32 maximum parameter-range error3.7628e-7 and profile consistency3.3146e-7 demonstrate representation margins (full measurements archive/pnr_float32_report.json). The residual above unity is disclosed, not hidden by arbitrary threshold relaxation. Correlated reduction errors, simplified interfaces and broadening/dead-layer coupling limit interpretation.

## Actual commands, notebook and figure review

From task root, shared research-runtime Python: `python -m src.main`; `python evaluation/judge.py --prediction output/reconstruction.npz --report output/judge_report.json`; `python -m pytest test/ -v`. Final all exit0;24 tests passed. Full exact output is archive/pnr_final_commands.json. A final test attempt using Windows default TEMP failed14 fixture setups with WinError5 before Judge execution;10 non-temp tests passed. Redirecting TEMP/TMP to this worker's archive/pnr_temp resolved environment permissions without code/threshold changes; the entire24-test suite then passed.

`python -m jupyter nbconvert --to notebook --execute notebooks/spinel_polarized_magnetization_depth.ipynb --output spinel_polarized_magnetization_depth_executed.ipynb --output-dir output --ExecutePreprocessor.timeout=180` completed exit0. The canonical notebook retains all executed outputs; its temporary output copy was removed. Timeout180s comfortably covers both full fresh fits and162 conditional scans. All code cells executed with no error outputs. Windows kernel emitted the ordinary Proactor/TCP fallback warning, without failed cells. Final notebook rebuild changed only plot-title wrapping, then the full notebook was executed again.

Three actual scientific figure groups /12 panels were viewed: observations (both spin reflectivities, asymmetry, both residuals), profiles (nuclear/magnetic profiles, parameter agreement, uniform/dead goodness of fit), sensitivity (conditional dead-layer scan, conditional broadening scan, local linearized parameter correlations, actual optimization history). Private review images archive/review/pnr_observations.png, pnr_profiles.png, pnr_sensitivity.png. Titles/units/legends were visually checked; the sensitivity titles were wrapped and rechecked. Conditional scans hold other parameters fixed and are explicitly not marginal confidence intervals. Original uniform versus dead-layer comparison is preserved using two fresh fits and independent native overlays; no archived array is substituted for a fresh curve.

## Completion gates

PASSED: exact source/data/physical binding and traceability; original full runs; frozen independent parity plus physical checks; units/constants; CLI and fresh Judge; invalid/poor/missing/shape/dtype/nonfinite/GT-precedence/hidden-assets/report-write tests; visible-input sufficiency; no truth/target-derived calibration or public GitHub URLs; explicit local forward/inverse mathematics and minimal NumPy/SciPy/Matplotlib dependencies;24-test suite; active six-section complete notebook; original/fresh/reference labeling; detailed README/plans consistency; licenses and template inventory. No unresolved mandatory numerical gate. Parent acceptance remains independent.

Final portability follow-up: test/conftest.py now uses a unique external tempfile directory only when the caller did not specify basetemp. Standard `python -m pytest test/ -q` then passed all24 tests in5.01s with the default environment. Parent-supplied basetemp takes precedence. This changes no physics, fixtures or thresholds.

## Bound asset SHA256

- `archive\spinelfilmzip.zip`: `b448f6130c77b3662a3b600acf52122b9d025600c8432f05be5d473e0c889e9a`
- `archive\spineldeadlayerszip.zip`: `ee794e3994bf5989a672f2cef9a524d2aeec034a4d473c552b9583039b44b9dc`
- `archive\pnr_wisser2019.pdf`: `7a137baad6dfb99250d136fa4589ff38ca72ef04b26ff302d0d4f68a68ae7f8e`
- `archive\pnr_oracle.npz`: `20c7dffa01b02f734d9fc49ab392b258831803b9673f76c52980f67b1b6a8d3f`
- `archive\pnr_native_run.json`: `aa6d2649ff924b88253dd94b58385a32a2d84c883da2f218cb3b3b8476c9ec63`
- `archive\pnr_preimplementation_protocol.md`: `d543f836e4cfc0cada4ad0824dfcc78ddea4f526f421d4c22b7e22ba82eec6bb`
- `archive\run_pnr.py`: `d4808c7cd70d0f066e3e6ffda7fd6338ff6847a1cf91445032cb5dfb3c119adc`
- `archive\freeze_pnr_references.py`: `0d5fedca2204b56d336b6a0fe21384b6679a6e62602ea13026743c7d4788d0c4`
- `tasks\spinel_polarized_magnetization_depth\data\raw_data.npz`: `c1a55f1df4e204f41e16d2470e6670243bdfa70fe31434f1aa4a84f5eacaa4c3`
- `tasks\spinel_polarized_magnetization_depth\test\fixtures\upstream.npz`: `2d10f3ec8d9e94b792396c06f71280d04eb304dee5fb0228decd8509015da340`
- `tasks\spinel_polarized_magnetization_depth\evaluation\baseline_reference.npz`: `bdb6b3da5d609828aa63341806a8644337320d66f973e00f60cd43c753e1d3de`
- `archive\pnr_sources\bumps\bounds.py`: `c571ec1daf7aeb497aae9ed732cc256c11e053fa43f55d73d2d1df07552a7508`
- `archive\pnr_sources\bumps\bspline.py`: `6458c293ee0fee257b5e34c81e7fe205f1878bd94715d75c5b21dff96c000c01`
- `archive\pnr_sources\bumps\cheby.py`: `27e69a62edbedca2f2e0b05ca8dbc93b25cf063ccad6a4cb763a9ed02ace6989`
- `archive\pnr_sources\bumps\cli.py`: `d09acc5dd05f5397fb7017417f42c96d7478080cf1929c9b8cbe52a39fc00705`
- `archive\pnr_sources\bumps\curve.py`: `d75c6baa4bf9300f23691bb4cc821da497958cf4cbec1e417cb6c8cae5df7bfd`
- `archive\pnr_sources\bumps\data.py`: `f71dbae67fcfbaa1d726b8ea5c6efc198878b6504760fac2cf1454c2f5f9abae`
- `archive\pnr_sources\bumps\dump.py`: `92b1251e95a41a3a7996379632961084a0e2ac4bad67b25467657fe8591fbd7a`
- `archive\pnr_sources\bumps\errplot.py`: `860baf5d4296340f5abcc9199fcdece7c3fe50a42fb098df06d2138da196ff0d`
- `archive\pnr_sources\bumps\fitproblem.py`: `2224708bd12405197aa3b972deaa80f85a67e45ba89e35976a4481096cb16f50`
- `archive\pnr_sources\bumps\fitservice.py`: `4540989d62d9a3c7ba593dccadf481ca4d2254c78e5e24c23d6d13c25f0bb762`
- `archive\pnr_sources\bumps\fitters.py`: `efd21ab1d4a9568fb06d423d11adb022930beaebf10a00ea7c0258ba8a28a171`
- `archive\pnr_sources\bumps\history.py`: `76cf39a34feffdf0f2c9cf8bc12648d17dcd26a33165d5ae9aa606dfc1c18308`
- `archive\pnr_sources\bumps\initpop.py`: `5afb555e3bb7d68d527f9dd479367b04b079a1e0a7fc0b49d47f9bc9792b43b4`
- `archive\pnr_sources\bumps\lsqerror.py`: `a6c13fb47d0106872e599a23a056fe1986bdd27c2e7ec1071c1ca3642ba0146d`
- `archive\pnr_sources\bumps\mapper.py`: `cf0ec77d7fc7b703e2c84ee95ee9f52bae095fecb3ed76067bf97343d018db3d`
- `archive\pnr_sources\bumps\monitor.py`: `1c3d999657999b9f49de1c9b43a61f3dd82eff8042a8d1394919e21286f2a493`
- `archive\pnr_sources\bumps\mono.py`: `66b935e0b993fce4f4a58f4a5f3aea57a87ab3958facfe30e8d2d8cdef700cb4`
- `archive\pnr_sources\bumps\mpfit.py`: `bde3d257985809077fd7ae7a49046308bbb821abbc2a0bc3379248fcaf2e03c4`
- `archive\pnr_sources\bumps\names.py`: `e55941804718fa45ef94faf1390cd8587d9d63f56ff92b402aac144d07295917`
- `archive\pnr_sources\bumps\nested.py`: `dc7892b6d73e9d8ed5b96c23dba58dfdf6e8096de806add326e11e71f5067b5b`
- `archive\pnr_sources\bumps\options.py`: `bbef48f39df23f9f725cb2b8548cac2972f1e046fb43e5bf5ad86c158e21d1be`
- `archive\pnr_sources\bumps\parameter.py`: `8f7c139929e211ea1ed9c498312370a403020a6bb6c6cf0feebf61758d0a62a7`
- `archive\pnr_sources\bumps\partemp.py`: `cf76d35031718a78c2d406000f51e7fbf2f00dfa36aab2aecb453565e561e7f5`
- `archive\pnr_sources\bumps\pdfwrapper.py`: `addec831462cb050617a89b277e82960190ba9a2b9c3bea1cddf5ce8cab8f4e1`
- `archive\pnr_sources\bumps\plotutil.py`: `9ffa8948dbd91334f1a89bf643a8c51a306dd4b515bb46f777cc077709ca56ea`
- `archive\pnr_sources\bumps\plugin.py`: `b6fa41d461e9bc3588f3621fd384c7ab15a9f01a973caa6d2b42671e4ab7ee47`
- `archive\pnr_sources\bumps\pmath.py`: `a8dcabb05b7ad1cc4d341f3bdf5dcfd4823feb83a87e04f0f5f0e3e4225b2613`
- `archive\pnr_sources\bumps\pymcfit.py`: `61e91b7ab9ad488c6588d064a3f6d0ea5cdb09d24e994611f84b18c3d3764677`
- `archive\pnr_sources\bumps\pytwalk.py`: `6ae5f839826c6626b296fd395457a3cbd94a187ad70a602b7c7a1d9761806abc`
- `archive\pnr_sources\bumps\quasinewton.py`: `dc273014b7df4c3317a260cd2c7e88a552b221530d65cbdf590f49c0a1004e81`
- `archive\pnr_sources\bumps\random_lines.py`: `c16d2c06a82b002f2c0f85a1babef24020b924b88d2fc94b3ba535ab6ef5e904`
- `archive\pnr_sources\bumps\serialize.py`: `7e6a15fd062d3b03cfd8480793e65f1f0355d920d9d829adc104481c9addd452`
- `archive\pnr_sources\bumps\simplex.py`: `621efa8d9edd6ef00f947d7e708c5823f0615efbb92e8a98e6b24041fc256234`
- `archive\pnr_sources\bumps\util.py`: `b722279c464831b9b925ebb39e1cabc42d7194ef2d5cfd2e2ac6bf60cc48ef71`
- `archive\pnr_sources\bumps\wsolve.py`: `810b5fc7a81031ae24fb96e1caa9a4f80a9138781741e4b30ea7ff30b72e99af`
- `archive\pnr_sources\bumps\_version.py`: `69cb91fd7489ce9e0ead0e53f3e01ce6061ee3c994c0e6eec234668ac16667b9`
- `archive\pnr_sources\bumps\__init__.py`: `94130a1176d95730925ca72d41766085577bed226c34bcd19449ffb50385c53d`
- `archive\pnr_sources\bumps\__main__.py`: `5ba654e35921351a5ccb44a0ae8a4f61e3b53428c80ed675a5a71da4be2f4415`
- `archive\pnr_sources\bumps\dream\acr.py`: `9f3f2c0e31530e197c36e13aaea4f925faeb8032cbdd9f5d7afacfc5533787e0`
- `archive\pnr_sources\bumps\dream\bounds.py`: `caea208cee42e9f140f939a018c491e2c1e59df63c3a50c27c31035182ac3624`
- `archive\pnr_sources\bumps\dream\build_compiled.py`: `20eefc423c01e77f7452477bf3dd07b06eda760369353c0ab25f86b38fa6f377`
- `archive\pnr_sources\bumps\dream\compiled.py`: `541d63a299682866c650c22d48752328283d759c0d5be54d4cc1f07e3f3b330c`
- `archive\pnr_sources\bumps\dream\convergence.py`: `6659ecfa4eba5065b91dc4e079e6c59a45aeb26f00c60681fc5a6ba93837be92`
- `archive\pnr_sources\bumps\dream\core.py`: `891513cb01837c7e08f91113b87ff636dd7b7a5380a6f2baec77ff3edcbcd529`
- `archive\pnr_sources\bumps\dream\corrplot.py`: `31df03e9177061f0aa4c678c00827cc8d60fa29125994f86b5c0dfa5942aa23d`
- `archive\pnr_sources\bumps\dream\crossover.py`: `1a7d8f86e36d0dfd8f1ee620c1dcf5aa7f17fc2de176cd64a7a62b041b53bfde`
- `archive\pnr_sources\bumps\dream\diffev.py`: `3c2d6e513e8eb9cec5620a0cf88009febbe5635cc828bfed4e9f97b3f8d2a8bf`
- `archive\pnr_sources\bumps\dream\digits.py`: `f3b30693539122a71dd0d0059b46d1f5d1ce45a0594be2cb5bc9d000b8e05771`
- `archive\pnr_sources\bumps\dream\entropy.py`: `46913d887c63e379710e2910aa57f0c47417865b4f9751f395a07206c68d1ad7`
- `archive\pnr_sources\bumps\dream\exppow.py`: `ed09b1a77dce6f455ab2352045720149960faa94b2edd2a1b369125df6876a6e`
- `archive\pnr_sources\bumps\dream\gelman.py`: `d8b5ee43ad9a346d5551bd6c7c7a1a74165135756156f2316e3f3e8c10295d8d`
- `archive\pnr_sources\bumps\dream\geweke.py`: `1f4d1a7fd54b2e09a521170027bbf88da17d4913d121cd645be0390d43677c5f`
- `archive\pnr_sources\bumps\dream\initpop.py`: `442601fbb85ef3cd5cee7deacca4408013df783db18b880dba73e85c9af3cd6f`
- `archive\pnr_sources\bumps\dream\ksmirnov.py`: `20cf4253522d665ddc7a31ab14e7e2f0641525cc14888ff4133570ce6878f249`
- `archive\pnr_sources\bumps\dream\mahal.py`: `bb1ca029efa80d3e4e382beb8580f1d1023189c730432e3e3bd7d7a67f6792d8`
- `archive\pnr_sources\bumps\dream\matlab.py`: `da32b90d7e989d89e1222f12600094fdfaaf4e7fb3edc12b8d3feee7f916bef1`
- `archive\pnr_sources\bumps\dream\metropolis.py`: `39ebbb6d2ab41e264a1c8e91dbf1c314be950740a8f9f88a090f7ea8b03aa500`
- `archive\pnr_sources\bumps\dream\model.py`: `27bd858b98ca2e6109f3b18e27db321727733c380467f5582580dcc99fbc0494`
- `archive\pnr_sources\bumps\dream\outliers.py`: `fcce1f469e1754724c8ae1af7ba064a7ff58884cf81ce01f7eb936f6a51337ec`
- `archive\pnr_sources\bumps\dream\parcoord.py`: `c1551655672c2158799b62beaa8ea62df85ef84f8826bd8104b1284142928772`
- `archive\pnr_sources\bumps\dream\state.py`: `885cc4f8ffa8ad5ef2153f882acc2ccd88865aab46585e655192b219770f5b42`
- `archive\pnr_sources\bumps\dream\stats.py`: `fc14d1c97bcd7cbb352fdbcdee24a98b3dcde39ec3f85e8ea07d452578e3665e`
- `archive\pnr_sources\bumps\dream\tile.py`: `32c2b61df53f224c0d929c3214402893639fe92502d82cdf09d0f0d59055b85e`
- `archive\pnr_sources\bumps\dream\util.py`: `11fb36d5488fd71d31102556fad2a16e2f5f348c7884538efc951bb1f3bb1f92`
- `archive\pnr_sources\bumps\dream\varplot.py`: `8b2699e27fe372e54d64025656dbf5d16268c453355bc8a12c5e99dc8d4f9a89`
- `archive\pnr_sources\bumps\dream\views.py`: `02b8d13eb7b48178325ef73ce13285e291374f18b57fab0edbda49b2816d2031`
- `archive\pnr_sources\bumps\dream\walk.py`: `a55fe8c15d388dc1ba2b2756d321e7e2e67665a3f814454e4132ebe846e9bf24`
- `archive\pnr_sources\bumps\dream\__init__.py`: `05fd2b31aee48fd02ea0e337971849f748c5831ff20eb5bd79196b145014b95b`
- `archive\pnr_sources\bumps\gui\about.py`: `afc42260c92270f11b6a289d9cfef0e2b1e4d88a0b85ad228f5f7d1c8f816fbe`
- `archive\pnr_sources\bumps\gui\app_frame.py`: `b564f3943c348ddeeae10a9bfbf844e21cb309de9870a436b6dbe3ac9235e879`
- `archive\pnr_sources\bumps\gui\app_panel.py`: `4be77b4f31cfb7fa8e36afbf9f1066a1bab0f532eddcfcb849550aa55ad067a3`
- `archive\pnr_sources\bumps\gui\console.py`: `0072a2e44a7c42c30430102f826db3c722052f8fca5d6372d265caaa882ce95b`
- `archive\pnr_sources\bumps\gui\convergence_view.py`: `30171875a3e47c68a8f234825342e2a34b5a5b5531378a8e1899d2a6fb796988`
- `archive\pnr_sources\bumps\gui\data_view.py`: `82be1faffb3e1f2218a95f8499ee31478088091cdf266ee4f020182c2aa94f9b`
- `archive\pnr_sources\bumps\gui\fit_dialog.py`: `6bdc98c13bb93e30a6681e85b3197e987d1699591e5c297726c633ca613c30d1`
- `archive\pnr_sources\bumps\gui\fit_thread.py`: `32006eaea6f0069915a6126fb55dff818aa72f29f9865bffc79fe5436af6ecf8`
- `archive\pnr_sources\bumps\gui\fit_view.py`: `60dda083961c29e1b94e7ca3902bb6dcacbf24ba43d406836b7297428502b649`
- `archive\pnr_sources\bumps\gui\gui_app.py`: `342834164d9fa837045f0d584a1fc8bfca2421b36819fca73ec58104af9b4104`
- `archive\pnr_sources\bumps\gui\input_list.py`: `83d9e4976d58945f3e732411571c8651532646040341edbcdac2728864a14f3a`
- `archive\pnr_sources\bumps\gui\log_view.py`: `976bb8e766c8dbc4a07b876c6ca075510a2600e4bcd3f326c71a8d314a78a8dc`
- `archive\pnr_sources\bumps\gui\parameter_view.py`: `eff4f14edcfc8a6f13ba2eeae3e367b3d53d48539b187d0a3f8621f00033789a`
- `archive\pnr_sources\bumps\gui\plot_view.py`: `5fea758e72890aa64e0485e1d5a4c06e0b5b7db3b00d40cf1f3fc96a43245b4f`
- `archive\pnr_sources\bumps\gui\resfiles.py`: `f3ec453c141b8ca75d34d101c1a3502fac2cd1d59890c64f62688ff8d265ba39`
- `archive\pnr_sources\bumps\gui\signal.py`: `219a2f21c2f5c1b5fd3562973e7a985074934ade74dcc538750980a595c8383a`
- `archive\pnr_sources\bumps\gui\summary_view.py`: `39009762ce46a1a6c072f87ddc8240ccd9545365822a3fa92c8842cb90b5f09f`
- `archive\pnr_sources\bumps\gui\uncertainty_view.py`: `0094dea2482b8601c8734a6b201c18302dc35276fe124626cd258e0000130d98`
- `archive\pnr_sources\bumps\gui\util.py`: `0ad82afca288a6876640936f5ab8eb23b5c6694c439da5d8cdb96f4b751ce241`
- `archive\pnr_sources\bumps\gui\utilities.py`: `3573b3d0bbecbfdddad7be0abba084d4ee1f55a4a2fbb8b7a9cc595ab65221b8`
- `archive\pnr_sources\bumps\gui\__init__.py`: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- `archive\pnr_sources\bumps\gui\resources\__init__.py`: `14bfaea6cafb5f0790f8def7d85c76e7224fbbec21de6cae7f704038f9265a02`
- `archive\pnr_sources\bumps\mystic\condition.py`: `4ec0026beb2adcbb9ea22984f36d51c8d0e79bf6b8af6b6e1d835c55bac58a96`
- `archive\pnr_sources\bumps\mystic\solver.py`: `6d93634b31ccfef353ed78947fc907f3bdcfc91e92918bd4435c1a1530c4a9b4`
- `archive\pnr_sources\bumps\mystic\stop.py`: `f7659720d0b072026beb918c9669a2eefe6f2b9c5cbbb39b45436688b23b7769`
- `archive\pnr_sources\bumps\mystic\util.py`: `9e09678a2793501fbed23a7d25ec1ecf38b4d377594024b7e343dbffcee90276`
- `archive\pnr_sources\bumps\mystic\__init__.py`: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- `archive\pnr_sources\bumps\mystic\examples\circle.py`: `ed6b83818c90889d60527816bcdac75ad631f5611f75882f64c15146936d348e`
- `archive\pnr_sources\bumps\mystic\examples\corana.py`: `2d78fa43eacdae0392714fd215c5246fe8cb6ae65f82eaf28b5d538182e4b995`
- `archive\pnr_sources\bumps\mystic\examples\decay.py`: `9db20148a9ff71e7745aeeb016ba15a86d1130490ff4391efce1eb6d0709577e`
- `archive\pnr_sources\bumps\mystic\examples\model.py`: `63ec9f5deca145d74c0726778b86970506cd3ab345091f3affc7c61d3c0f7200`
- `archive\pnr_sources\bumps\mystic\examples\simple.py`: `5ec25c84c3aedf9ddb39e44c6bbc4bbf0b900e26f309251656f797b176fc0238`
- `archive\pnr_sources\bumps\mystic\examples\__init__.py`: `54283d462d1289a94668d57c4ee123dcb7c421bca77e8f47b0e0f0094bbe7998`
- `archive\pnr_sources\bumps\mystic\optimizer\de.py`: `b04a3d105ae2b122d5b044e339f804ca2ceffbd6ec56936f6dbc78dbd44f2346`
- `archive\pnr_sources\bumps\mystic\optimizer\diffev_compat.py`: `a0696af4a5f50c1e1284436d061950f1b39e350546e48cffe0ae7c5b8c6413f3`
- `archive\pnr_sources\bumps\mystic\optimizer\__init__.py`: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- `archive\pnr_sources\bumps\webview\build_client.py`: `d119ebe920214a0c2a6dd23f83d077f208d8f728b1e2859465dc8475627ca77a`
- `archive\pnr_sources\bumps\webview\__init__.py`: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- `archive\pnr_sources\bumps\webview\server\api.py`: `1db434ede4904d5e2f3f22b93a9c920b643935420f0e5347a1e3bfd4b080686f`
- `archive\pnr_sources\bumps\webview\server\cli.py`: `4ff21bf1cd1cdaed33584a37d9ea72fad0966171fc942ea1bff1e27d574136a7`
- `archive\pnr_sources\bumps\webview\server\colors.py`: `fbd89a78182e6d65b65a2274d98de715f8d172e10ced83d7e716fb8f00a12fdf`
- `archive\pnr_sources\bumps\webview\server\convergence_plot.py`: `80697a56ba97bb44df111c9845c69811b46f92761a419307b31f4e4266ca50a5`
- `archive\pnr_sources\bumps\webview\server\corrplot.py`: `1c3de57b62b6146717f3395560aabdf0484de0f2d43b16700e26c97fbe75d61b`
- `archive\pnr_sources\bumps\webview\server\custom_plot.py`: `2ad2e5560b66b00a74b07775ce6bc7e51d37cd359c25518236e70b6df8cd67fa`
- `archive\pnr_sources\bumps\webview\server\fit_options.py`: `f66d6288636e1c9b68c6e4aa9e96da9541207a991c9e9cf2d92316a48539675b`
- `archive\pnr_sources\bumps\webview\server\fit_thread.py`: `3cb47c28d5c919b347ac0841519bf70e37b33ead3af263a55053279c7af551cd`
- `archive\pnr_sources\bumps\webview\server\logger.py`: `69cbef867bf1b82bc8e21c28bb935e1518e3b785436a344cf191801e74618e2f`
- `archive\pnr_sources\bumps\webview\server\persistent_settings.py`: `b158412a9f295d14654bf27c1685aced799fc97fe5fffe7a70c1f2ab53fc356a`
- `archive\pnr_sources\bumps\webview\server\state_hdf5_backed.py`: `f02d5d63ee9104704325d7ed3b9efd49db3dfa6aeef9c8bf55291ecb609404cd`
- `archive\pnr_sources\bumps\webview\server\traceplot.py`: `5fef44b02303f248df856bcbea4be37e07b5a7c355c10a117be5ede0f82d9765`
- `archive\pnr_sources\bumps\webview\server\varplot.py`: `e40f5b9903d30e9d6faf16ca5417961348239245e3889942c29b7ecbf83a30b3`
- `archive\pnr_sources\bumps\webview\server\webserver.py`: `f8d81111c234cfa0357cb3f3d9e671e4e6581e385de289a4387c1784416fd55b`
- `archive\pnr_sources\bumps\webview\server\__init__.py`: `fe64e59777951c4e1629573b365d55ce977ac313da6003ead8ae435227bb48fe`
- `archive\pnr_sources\refl1d\backends.py`: `5e0b77eec787904c6fc3bd30b321eaadd17c90f9cd825d87dd75cf8351b61caf`
- `archive\pnr_sources\refl1d\dist.py`: `367882429e2cb83817a2d9e3aa139b81fdb4fb72957126bad285812096f9cfb3`
- `archive\pnr_sources\refl1d\experiment.py`: `3da7e997d0bd246004a82483174b5ced38937daf23edfe6feeb6fb7625be9ddc`
- `archive\pnr_sources\refl1d\main.py`: `a7c306e0780927825616ccc523832e627c39e88a4b91fa239b6791689e8da08d`
- `archive\pnr_sources\refl1d\names.py`: `08b98c44336732e9dae817fdd536f34ea01172155823427c511339a826325f7b`
- `archive\pnr_sources\refl1d\profile.py`: `d1ea094eaf0f1db0d55ae43e07d19b8e228b7d89c2b7f2e80ef9ff0bdad6fb0a`
- `archive\pnr_sources\refl1d\uncertainty.py`: `f9b24817a3c2c085b17cb9ddbdbcd750f771badcef5f4bc052cb49d2e906faea`
- `archive\pnr_sources\refl1d\_version.py`: `7784076264bfdf48f484f37c36634f9d0fd9d13a610d16eee0cb508fd713f5ef`
- `archive\pnr_sources\refl1d\__init__.py`: `8f9d3493220cd148ba9d8328cf35f226528fcc13e756607c2b57875ca2f267a6`
- `archive\pnr_sources\refl1d\__main__.py`: `bd1ce07b4b907bdf757121499893d159734dd29d0a5699f4d95c3a17794a24d9`
- `archive\pnr_sources\refl1d\bumps_interface\fitplugin.py`: `9490e516c0607031994019ea419c9b58b34e26a49e8ee94be6181e4a8da77fca`
- `archive\pnr_sources\refl1d\bumps_interface\migrations.py`: `9351deca16d2c9b416076be848af56ca8b049927a6a405c8c80be29188ac8de1`
- `archive\pnr_sources\refl1d\bumps_interface\__init__.py`: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- `archive\pnr_sources\refl1d\deprecated\freeform.py`: `e0b45a11f073fe967666fea7bc3c5214d430235d58ea480b3a8f62ac41a8703f`
- `archive\pnr_sources\refl1d\deprecated\magnetic.py`: `2ca24dd6069cfb48f5e5e5c7335c335028d79e9fad5be8863d02bd80cd0e51de`
- `archive\pnr_sources\refl1d\deprecated\__init__.py`: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- `archive\pnr_sources\refl1d\lib\__init__.py`: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- `archive\pnr_sources\refl1d\lib\numba\build_profile.py`: `7ded46019d993f23f4c18c39c6a6a8014931a8afaa65d7cfabf81d4a1362f4b0`
- `archive\pnr_sources\refl1d\lib\numba\clone_module.py`: `61bdbaf3ce36fd3853d35703aa076ff5aebd7ac100a339d12f9d4645f6d619e4`
- `archive\pnr_sources\refl1d\lib\numba\contract_profile.py`: `1a3bf4339b0d3fdaf0bac2a749632c3727d613d399deb15e4c89b9c94d3047c6`
- `archive\pnr_sources\refl1d\lib\numba\convolve.py`: `488405d4d0291b545ec72e16fba4ad1f45a9524f10159ee0304b6ea76184e028`
- `archive\pnr_sources\refl1d\lib\numba\convolve_sampled.py`: `729d91864ed393701d40c8e8de51109d4b9002a7f89e194d5286fdd45371e403`
- `archive\pnr_sources\refl1d\lib\numba\magnetic.py`: `84e706841ea1576da7da54dd11849c6c5ac69e3de66865049e39c4484d77d981`
- `archive\pnr_sources\refl1d\lib\numba\rebin.py`: `69c8cce0d9d572d7d5895bc8702067dc33c6b1ceee193257a6de508416dc7b3d`
- `archive\pnr_sources\refl1d\lib\numba\reflectivity.py`: `4eb86e4cfb3900b9ccdd8109d505906beb378bdc92ae7f4ce7c0b57d04118f1c`
- `archive\pnr_sources\refl1d\lib\numba\__init__.py`: `d39aa817f628426a967bf7425c1f98d4291868ce53938f3b6f41a986deb96505`
- `archive\pnr_sources\refl1d\lib\python\build_profile.py`: `726e7ae93546491451f179f4505eeed7882a0f0ac91f8dca9440a03892921793`
- `archive\pnr_sources\refl1d\lib\python\contract_profile.py`: `49fd1c218d81197a7a7c24ce846481ad3d2eab922af818cdebfe9ec55ba4afd9`
- `archive\pnr_sources\refl1d\lib\python\convolve.py`: `f4bae47b2b633125a13a96d84c5c350b910e634d183ee318b871b219a8931925`
- `archive\pnr_sources\refl1d\lib\python\convolve_sampled.py`: `e3c9752ce2f8d9cb09dbfdfb6dfc0ca61cd75b23d7aed9d980efa8760245976a`
- `archive\pnr_sources\refl1d\lib\python\magnetic.py`: `c91b327a94778e384115bfa0fb347c7a33a04b7c68c29009fe8faa8be6cedf7d`
- `archive\pnr_sources\refl1d\lib\python\rebin.py`: `3f88528315f6c75141cbc836cdf5dde3f822d61156daf3081d05b1df320cb516`
- `archive\pnr_sources\refl1d\lib\python\reflectivity.py`: `d7ca3a8eb4148b02e4c612287f04ccf3c880004530b1356f1e5ae5bef099c1f3`
- `archive\pnr_sources\refl1d\lib\python\__init__.py`: `d39aa817f628426a967bf7425c1f98d4291868ce53938f3b6f41a986deb96505`
- `archive\pnr_sources\refl1d\probe\abeles.py`: `4d0721b1821b3df765fb8daebc797ccf32363723c1d5d0aed3617ca7f3ad7c1d`
- `archive\pnr_sources\refl1d\probe\fresnel.py`: `5234ca163f34463a92af36ef971e98aa85c166dec7f7b58543b6ea1a9d8c3d32`
- `archive\pnr_sources\refl1d\probe\instrument.py`: `3e996276f2fef87807c572db1861d0ce139fca87f1722ab55e3be8df9433a3ea`
- `archive\pnr_sources\refl1d\probe\oversampling.py`: `dd3865fe02ff1ddf567537bd47bb17ad2f816d233dda290d88862ea52475d8d1`
- `archive\pnr_sources\refl1d\probe\probe.py`: `8947adadf23269cbeb71ce5206535b1834038a532e2444fbf937693128ac2bbd`
- `archive\pnr_sources\refl1d\probe\resolution.py`: `11b78cf46f5ad84a21e7873dea64a6fb937f20287e477cdde70afa3e84e4fc7f`
- `archive\pnr_sources\refl1d\probe\stitch.py`: `c7dd11beff4921e6ee6d6727a7c8b0a3808648eeb520280f6ea176c05c08e2ca`
- `archive\pnr_sources\refl1d\probe\__init__.py`: `91814460cd796715d1e540ce5f0f6bb4cff248fc5282fdb0bbf3233e74656ada`
- `archive\pnr_sources\refl1d\probe\data_loaders\anstodata.py`: `1bfba28e71b4fb653020c715ff83be6511c9dbb961492be0012fd2d1cf5cfaf8`
- `archive\pnr_sources\refl1d\probe\data_loaders\load4.py`: `d8a5d39d17fbd3662eaa16b523b40f500adf9864381b260828f34642612b1cf1`
- `archive\pnr_sources\refl1d\probe\data_loaders\ncnrdata.py`: `224facfc793966a23fd08087aed26f0b9ad1eecdca25834d4337d8ced97e0885`
- `archive\pnr_sources\refl1d\probe\data_loaders\rebin.py`: `00858582539b474e7550a7e8a514d7775152caacaef93b8de1bd90da3a45f7bb`
- `archive\pnr_sources\refl1d\probe\data_loaders\snsdata.py`: `caafe22c94ebf111210f283ef673959189bc60ede6e16d7f39071543dc85d570`
- `archive\pnr_sources\refl1d\probe\data_loaders\staj.py`: `64b1641a77239638501124732994988c212de483c157da64f169ec8be9bdc23f`
- `archive\pnr_sources\refl1d\probe\data_loaders\stajconvert.py`: `ad5c67fedad9e191b6b3226c09cdf782954affd7db53d3ebbf48ba1862672a42`
- `archive\pnr_sources\refl1d\probe\data_loaders\__init__.py`: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- `archive\pnr_sources\refl1d\sample\cheby.py`: `968119e9cc2bfdcf0ccff8190d513c8686eeb0f9bb57802ff379145b0c72853b`
- `archive\pnr_sources\refl1d\sample\flayer.py`: `bf37c563f0fb7ba0e56fa5d9f12b31891adbf16bf6228dfc4be75c06a06bf808`
- `archive\pnr_sources\refl1d\sample\layers.py`: `586e6f0e7132aad37294150a1eb3826e844f9f9b71b24935d4b1079b050decc2`
- `archive\pnr_sources\refl1d\sample\magnetism.py`: `33d2cfb12dae202cb622baa39f41d0711dac3790709c543eab23888825dbb9dc`
- `archive\pnr_sources\refl1d\sample\material.py`: `5ac91b89bc41c5213aa51354d2e7f8f0f746a8cb7e218e65f6b58f5e61644165`
- `archive\pnr_sources\refl1d\sample\materialdb.py`: `bf1d801e64ddb780f135e8ed10409885a65d4052c6c0807c0a6214796cdf907b`
- `archive\pnr_sources\refl1d\sample\mono.py`: `e230dfee308372e296e9f96b43fe23110aaeb95c8d3de1ea13e969c1af4dc2db`
- `archive\pnr_sources\refl1d\sample\polymer.py`: `540beb619f19a45be14d14594e61faa8669c91fc064dc7bb234119edc9fa27bf`
- `archive\pnr_sources\refl1d\sample\reflectivity.py`: `fce22db5e4116dd5b21a6ad2baabb3ee8c9f6ac195bf30b5324aad1eb1ccfbc3`
- `archive\pnr_sources\refl1d\sample\__init__.py`: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- `archive\pnr_sources\refl1d\utils\support.py`: `e3b2fb8da71e5acfe9c0b1518a3f8caf06684e11cdffe37921961afc15181be1`
- `archive\pnr_sources\refl1d\utils\util.py`: `765828ff0364bc8caea57280d04b33173d7dc3a8e0a26b9845632448d418e00f`
- `archive\pnr_sources\refl1d\utils\__init__.py`: `830f3061c1ca05103478388888dd59007b605b29b53dc438344f57e9305ca4a9`
- `archive\pnr_sources\refl1d\validation\gepore_runner.py`: `fadf58a373118628e89bf30a1ea3a4a982eaf3babd2ed7f97d52303281bc219b`
- `archive\pnr_sources\refl1d\webview\build_client.py`: `8f5549641aaabd270954e7749173fe42d073c21ea77c7b7161bf6326dd212429`
- `archive\pnr_sources\refl1d\webview\__init__.py`: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- `archive\pnr_sources\refl1d\webview\server\api.py`: `9cad2afb335793919249a2a3507c5f98be4520cec345d0b427e90ec4733e0e96`
- `archive\pnr_sources\refl1d\webview\server\cli.py`: `d039c4bd8604a78074bffa4359a4ae5c1b08e90662ff89117494c7cfe2cf1b9a`
- `archive\pnr_sources\refl1d\webview\server\colors.py`: `77b47057a3b7719ce510562968498282c38c24de90e182037ee3cd3aa44233a5`
- `archive\pnr_sources\refl1d\webview\server\names.py`: `40efc1a1c6fcadce864c0ed50d3c2e901d138337bc69584ea5be7dd2c6e50b00`
- `archive\pnr_sources\refl1d\webview\server\profile_plot.py`: `c4e668633dac740de92050fd440ccc90776380091686841dc42d787ddc952b9b`
- `archive\pnr_sources\refl1d\webview\server\profile_uncertainty.py`: `9c8a79f300adf510a1fcf09f5b595880ed1c7b23519402e35355219946238715`
- `archive\pnr_sources\refl1d\webview\server\scriptify.py`: `b5c8fd61268d67008a34dd9a0845e7acb7d6bde0580b56dea27ec7e87a94bdbb`
- `archive\pnr_sources\refl1d\webview\server\__init__.py`: `0696f6c1e07729e8dd93e95483bf4156ff234230a0ba9dd7a736f864c8fb8309`
- `archive\pnr_sources\refl1d\webview\server\__main__.py`: `3129adff95e0f38b87ab34cddfc27082c7892bcaec267ff5d40f160fdf55b44a`
- `archive\pnr_sources\refl1d\wx_gui\binder.py`: `13ff94e1a891a9c33e1a651dc77f46460e69032334a096886e52a4a83e4e331b`
- `archive\pnr_sources\refl1d\wx_gui\config.py`: `017ebf5c341c232f809114432b44027fae762436ffe1bd40a0d418e49c96ace8`
- `archive\pnr_sources\refl1d\wx_gui\data_view.py`: `2b9b06a8e9e44c2c07ae83dde9e6d67dbef1620fe27e1706e1c53cc2733b87b2`
- `archive\pnr_sources\refl1d\wx_gui\demo.py`: `576c240179d6f49d3b2445373df393a4666b7f14c8f26b879168f7b5b81dfa87`
- `archive\pnr_sources\refl1d\wx_gui\interactor.py`: `f66d689ed7c224683b4e05519192085f89bf40d12bb3e543f124ee41bb2bd10a`
- `archive\pnr_sources\refl1d\wx_gui\interface.py`: `91a5f618d4c369550ba100db64e5806bbea61b1c7633173020925e2972d25819`
- `archive\pnr_sources\refl1d\wx_gui\layer.py`: `a45f868fe8e30da5fa77c16a104dbce1a3ec26c4d875e465fb870758900d8d34`
- `archive\pnr_sources\refl1d\wx_gui\layer_dialog.py`: `9daee7b9f93282bac0bd617af3737c5709296dd42a5e24be8a9bd04957bf7169`
- `archive\pnr_sources\refl1d\wx_gui\model_view.py`: `b3ec4d81432357617682a91b03616c6d0426e9947ae3a9f0055ee48e560bbb78`
- `archive\pnr_sources\refl1d\wx_gui\monoi.py`: `d6206b29921276cc45e19108028cf39a7ccb790d6c7b4a014a08f2fbcedf41d1`
- `archive\pnr_sources\refl1d\wx_gui\polymeri.py`: `9dc5cc8fb74b2b73cfd5f87d0addaaaf84e51515786b2e270874a9c1395997c9`
- `archive\pnr_sources\refl1d\wx_gui\profilei.py`: `31febb117219c53aa1a1c42459bcd2b5ae5ee3c7dc6aada92ca770596c415586`
- `archive\pnr_sources\refl1d\wx_gui\registry.py`: `4c61e45619aa5b48b2e9f49ea419352fdc7be21e9b9183f265e4f88017942712`
- `archive\pnr_sources\refl1d\wx_gui\thickness.py`: `547efbd6dc0d8c938743759fc0b2e3e18ced5599b4a449f36b7a891d9598766a`
- `archive\pnr_sources\refl1d\wx_gui\util.py`: `b2f39b54708f836d8c49ca8228deece49a9a9508829a95c01e6fbcd1b3c73adf`
- `archive\pnr_sources\refl1d\wx_gui\__init__.py`: `3ee67b746b59095936d6248f5a4b085c6a452cf9cc7aa8337cdc071a5068b6b6`
- `archive\spinelfilmzip\MAFO_Saturated.py`: `69dab9e19c0261ce6268ea53f1449c87c8e93064f3d262b72c21c016f5a2c38b`
- `archive\spinelfilmzip\MAFO_Saturated.refl`: `4c599663fdba4d3facc29e114f2291181bb602c375a2fed48e9a053edfd8decc`
- `archive\spineldeadlayerszip\MAFO_DeadLayers.py`: `9446eaa5414f9ee8607d187617640ab4ffd0d82f58f74488f5ac8bd8f16e9e6d`
- `archive\spineldeadlayerszip\MAFO_Saturated.refl`: `4c599663fdba4d3facc29e114f2291181bb602c375a2fed48e9a053edfd8decc`


## Independent parent acceptance

CLI, standalone Judge, pytest, active nbconvert and dual-engine math rendering were independently rerun: all exit0. Parent notebook: 6 executed code cells, 3 computed PNG outputs, 7 formulas. Private evidence: research_batch10_20261027/acceptance/spinel_polarized_magnetization_depth/execution.json and semantic_review.json. All panels actually inspected by parent; source/physics/GT roles and limitations semantically checked. Canonical notebook is this parent-executed artifact. No new source or notebook execution is inferred from cached upstream outputs.


## Portable notebook kernel metadata

Final packaging replaces the maintainer-only kernelspec name batch10-worker with the standard python3 name. All code cells, Markdown cells, execution counts, outputs, scientific arrays and non-kernel metadata are unchanged; parent execution still used the explicitly recorded validated shared interpreter. This is a metadata-only portability amendment, not a new numerical execution. The shared environment must be selected when running the notebook. Private before/after hashes and cell equality evidence: acceptance/spinel_polarized_magnetization_depth/portable_kernel_amendment.json.
