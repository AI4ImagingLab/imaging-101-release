# Provenance and validation

## Binding, scope and deduplication

Domain: chemistry & material science. Recover a full 90-value effective finite-grid microfacet slope distribution and two finite-quadrature roughness summaries from measured angular reflectance of gold metallic paint. This differs from existing ellipsometric film thickness/index, EXAFS pair-distance inversion, SAXS particle-size inversion, and spectral chemical mixing: its acquisition is a calibrated bidirectional optical scattering table and its unknown is a nonparametric surface orientation distribution with self-masking. It is not a new sample of any old task.

The raw observation is classified **author_real_processed**, not raw camera frames and not synthetic. Matusik et al., *A Data-Driven Reflectance Model*, ACM TOG/SIGGRAPH 2003, DOI 10.1145/882262.882343, describes the actual camera/light/sphere measurements and HDR processing. Primary paper: https://merl.com/publications/docs/TR2003-83.pdf. Its apparatus uses a Retiga1300 camera, Kaidan turntable and xenon source; calibrated sphere geometry maps camera pixels to angular reflectance. Per-bin robust averaging removes the top/bottom quarter before averaging. The complete encoded 90x90x180 RGB table is preserved. Specific measurement date, exact coating formulation and sample radius are not supplied and are not invented.

Original file: https://people.csail.mit.edu/wojciech/BRDFDatabase/brdfs/gold-metallic-paint.binary, 34,992,012 bytes, SHA256 69c493d7f36fa5267d15948a616056f9bc84e352ec6db49b64afe6ff232e27d6. Dataset primary record https://zenodo.org/records/8101681 (DOI 10.5281/zenodo.8101681) specifies CC-BY-SA4.0; archived API metadata confirms this license. `NOTICE` preserves attribution and the separate embedded MERL code notice. The source header otherwise declares public domain. No fitted target, reference distribution, or privileged data appears in raw_data.npz.

## Paper and exact source

Dupuy et al. 2015, *Extracting Microfacet-based BRDF Parameters from Arbitrary Materials with Power Iterations*, Computer Graphics Forum 34(4),21–30, DOI 10.1111/cgf.12675. Primary paper archived from https://diglib.eg.org/server/api/core/bitstreams/2c8f2cd2-6cc9-4ab1-a999-0081ccbecc31/content. Author repo https://github.com/jdupuy/dj_brdf branch `egsr_2015`, remotely verified commit e0f2ba9af3a792338bc473905d78c5449937b723. The README's esgr branch spelling is a typo. Selected example `examples_merl_params.cpp` extracts tabular slope density and converts it to Gaussian/GGX moments; the cleaned task does not claim the unused full Fresnel model.

| Paper stage | Original function | Local function | Independent case |
|---|---|---|---|
| Backscatter reduction, sec4.1 | merl::eval; dir constructors | preprocessing.backscatter | 89 original RGB triplets, exact agreement |
| Isotropic operator, eq22–24, Algorithm3 | tabular::compute_p22_smith | physics_model.build_kernel | Original 89x89 kernel |
| Four powers, Algorithm1 | matrix::transform/eigenvector | solvers.reconstruct | Four original unnormalized states |
| Radial density normalization | tabular::normalize_p22, p22_radial | quadrature, mass | Original 90 samples; independent scalar weight derivation |
| Appendix C moment conversion | tabular::to_gaussian/to_ggx | roughness | Original two roughness values; independent moments |
| Positive-eigenvector physics | Integral eigenproblem | eigen_residual | Direct dominant eigensolution and intensity-scale invariance |

## Original execution and discrepancies

Private archive root is the enclosing `material_mechanics` directory, outside the released task. `brdf_egsr` contains unmodified selected original files. `brdf_fixed` changes only two `vector.reserve(size)` calls to `resize(size)` in matrix::eigenvector. The original release executable ran, but `-D_GLIBCXX_ASSERTIONS` aborted because reserve does not construct vector elements. The allocation-corrected checked run passed and its full90 distribution and both roughness values were bit-identical to the unchecked original result. This is a documented memory-safety correction, not unchanged portable execution.

A separate unused half_diff_coords_to_std branch assigns `in=dir(in)` instead of the computed vector. It corrupts Fresnel extraction. Neither the selected original PDF/moment results nor the cleaned runtime depend on that branch. No claim of original full BRDF/Fresnel correctness is made.

Commands in WSL Ubuntu: `g++ -O3 -I. examples_merl_params.cpp -o merl_params -lm`; `./merl_params gold-metallic-paint.binary`. Capture: `g++ -O3 -D_GLIBCXX_ASSERTIONS -I. capture.cpp -o capture -lm`; `/usr/bin/time -f '%e' ./capture ./bound.binary > native.log 2>runtime.txt`. The bound binary was reconstructed from the final NPZ and byte-compared with the downloaded original before this run; it is identical. Corrected exact-bound run exited0, 0.21s, g++13.3/libstdc++, Linux CPU double arithmetic, no stochastic seed, no warnings in native log. Original input/output/capture helpers and failing checked-source experiment are private archive evidence, never imported by the clean runtime.

Native roughness: Beckmann alpha0.42808655741306284, GGX alpha0.1822130943915658. Clean maximum errors: RGB0, kernel2.7284841053187847e-12, four states3.637978807091713e-12, PDF4.440892098500626e-15, roughness3.885780586188048e-16. Frozen tolerances precede local extraction in `brdf_frozen_contract.md`: RGB1e-12 absolute, kernel1e-10 relative/1e-9 absolute, states1e-10 relative, PDF1e-8 absolute, roughness1e-9 absolute. No tolerance was loosened.

## Explicit failed continuous-tail check and finite scope

The initial phrase normalized PDF was ambiguous. Independent continuous analysis found that linear interpolation in sqrt(theta) to a zero horizon endpoint gives P(s) proportional to1/s for nonzero penultimate sample. Hence continuous mass and moments diverge at infinite slopes. **Continuous infinite-domain normalization: FAILED for the original representation.** It is not repaired silently or claimed as passed. Root approved a finite-grid inverse scope: output the same90 samples, normalized by the same128-point author quadrature, plus the same finite-quadrature moments. No fixtures, outputs, thresholds, model updates, or dataset changed. The public contract and notebook explain this limitation and avoid claiming a continuous normalized microscopic PDF.

The largest quadrature slope is40.89529491397538. Independent scalar interpolation and weights verify nonnegative finite mass1 and both moments. The notebook plots continuous truncated mass and moments over maximum slopes1..1e8; mass reaches8.590516764468454 at1e8, visibly exposing the divergent extrapolation. Finite-grid moment estimates therefore must not be interpreted as convergent asymptotic material roughness. This is a limitation of the original finite representation, not an unreported failed gate within the clarified finite task.

## References, metrics and runtime

There is no independent microscopic truth; evaluation/baseline_reference.npz is an original-method reference. test/fixtures/upstream.npz contains original retro, kernel, powers, PDF and roughness, consumed by comparing tests. notebooks/reference_outputs/upstream.npz contains the original final arrays for notebook comparison. They are not clean-generated parity fixtures. Raw NPZ SHA256094c410f42675030643cfca813384f48d39c1ee12415c1ebb4acd95f181c6676; arrays are lossless full measurements.

Clean runtime: Python3.11.11, NumPy2.4.6, SciPy1.17.1, Matplotlib3.11.1 on Windows CPU, single-thread settings. Main reconstruction approximately0.0021s, peak data payload35MB. No upstream domain package is installed/imported by cleaned source, Judge, tests or notebook. Requirements only list used runtime packages. Eigenproblem density NRMSE uses RMS/reference dynamic range, NCC uses uncentered cosine. Roughness errors are absolute dimensionless values; all physical boundaries1e-5, NCC minimum.999999. These are finite nonzero numerical tolerances, not multiplication by a self-comparison zero.

Fresh report: NCC0.9999999999999998, NRMSE1.50037364355385e-16, roughness error3.885780586188048e-16, mass error2.220446049250313e-16, moment error1.1102230246251565e-16, relative eigen residual4.664496566781733e-8. The Judge independently recomputes the selected observation operator. No alignment, crop, target rescaling or hidden calibration is applied. Strict float32/64 shapes, finite overflow, missing files/keys, invalid shapes/dtypes, NaN/Inf, negative density, invalid roughness, poor values, missing references and report-write failures are covered.

## Notebook and final review

Active six-section notebook reconstructs the entire bound inverse, writes the fresh submission, invokes the standalone Judge and computes all comparisons. It shows full measured RGB angular slices; actual backscatter and kernel; finite-grid forward prediction/residual; original/fresh full distribution and roughness; analytic Beckmann/GGX compression; fresh1/2/4/8 powers and direct eigenvector; divergent continuous-tail diagnostics; and synthetic1% angular-modulation sensitivity versus uniform intensity scaling. No synthetic observation replaces the benchmark. Every plotted alternative is computed; original results are labeled separately. No paper-rendering comparison is falsely claimed, since the task excludes the defective original Fresnel branch.

Equations, units, direction conventions, factor4/8, mass measure, finite integration domain, signatures and file flow were reviewed against source. In particular G1=1/(integral P(u)uA du), giving G1(0)=1 for a normalized ideal continuous density. Tests explicitly verify scalar finite weights and analytic comparison-distribution mass. Library imports do not change Matplotlib backend. Public README/requirements/raw observations suffice, and public material contains no GitHub URL. Figures were inspected for labels, scales, legend clarity and absence of clipping. Temporary notebook is removed after transferring executed outputs into canonical notebook.

## Gate status

Passed: verified measured provenance; exact remote source/paper; exact-bound original allocation-corrected execution with original comparison; frozen independent parity; finite-domain physics; fresh CLI/Judge; strict Judge contracts; clean runtime isolation; explicit local mathematics; required README/plans; active nbconvert; all scientific comparisons; inventory/license and public leak audit. Continuous infinite-domain PDF interpretation failed and is explicitly excluded, not counted as a success. Final rerun: 26 tests passed in 4.72s. Full nbconvert exited0, eight figure groups rendered and visually inspected; final additional selected-slice prediction plot has maximum relative reflectance residual0.00016726222007408458. Canonical notebook preserves executed outputs, temporary duplicate removed. Standard Windows Jupyter selector-thread warning is environmental, not a cell error.

## Private archive index

| Relative private asset | SHA256 |
|---|---|
| brdf_egsr/dj_brdf.h | 48fb0185295a6f499419c5bf728420ecf8057946723707c1d9ff79a1c457ae06 |
| brdf_egsr/examples_merl_params.cpp | 96b8fc83b9d97a01a853b16a03cc7e402b88a89ad94d32da066abd8573d20bc9 |
| brdf_egsr/capture.cpp | f831eaf94f776bc49ed297e832a5f8da1b0d9d82792ef3c451d4f1fa928181c3 |
| brdf_fixed/dj_brdf.h | fbc59c760602e506b0b30645bbe9693f37a5cb3869675f63afd77a09fdb462c8 |
| brdf_fixed/native.log | c48516d9cc1989b89ad0f3ced09d01345ee5329ebdb5d2a1b6ba333f23cb84bc |
| brdf_fixed/runtime.txt | be46f7d50b21c1e9b40dc81079a4947e257511e80023845216066a9b0bac77da |
| brdf_paper.pdf | 9d1d4ea34a88fdfb56a0649d3999e8dadcd3bdd9ac129740c80e90f3409362dd |
| brdf_measurement_paper.pdf | 063e1e2d8f83a222afc34de696df8c982a820a6b8002dc0d503fc34d0721224b |
| brdf_zenodo.json | bab90b3e10ce633d89ed66d36c78b45a4fd03284437897d420e72f29aca5b0fb |
| brdf_oracle.npz | c7c582b109b9741e061d97d4e2d097fc571610c3bac89b4998d34ab5375e7472 |
| brdf_raw_binding.npz | 094c410f42675030643cfca813384f48d39c1ee12415c1ebb4acd95f181c6676 |
| brdf_binding.json | 6b962ff4b78a555f8ed938b596fffca4bf7cc86d2755955503721750e18e788f |
| brdf_frozen_contract.md | 93218a7eff2eb331dc37bea2af61c73c24966f79f69d7cc5fc5300c75877a05c |

Parent review correction: strict horizon sample equality now explicitly validated; nonzero1e-15 horizon is rejected. 27 tests passed after this correction, and full notebook re-executed with finite-grid figure title. Roughness docstring explicitly states finite quadrature.


## Independent parent acceptance

CLI, standalone Judge, pytest, active nbconvert and dual-engine math rendering were independently rerun: all exit0. Parent notebook: 8 executed code cells, 8 computed PNG outputs, 48 formulas. Private evidence: research_batch10_20261027/acceptance/metallic_paint_microfacet_distribution/execution.json and semantic_review.json. All panels actually inspected by parent; source/physics/GT roles and limitations semantically checked. Canonical notebook is this parent-executed artifact. No new source or notebook execution is inferred from cached upstream outputs.
