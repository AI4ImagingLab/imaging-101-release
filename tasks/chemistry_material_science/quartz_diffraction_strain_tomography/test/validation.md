# Validation and provenance: quartz diffraction strain tomography

## Fixed problem and source binding

This new chemistry and material-science task reconstructs six spatial strain fields from directional diffraction line averages. The unknown is an image-domain tensor field in a known grain support, not a scalar calibration or spectrum fit. All existing 500 by192 author-supplied observations and all six spatial components are retained; no target, ray selection, observation downsampling or new synthetic generation is introduced. No old task was modified.

Paper: Axel Henningsson and Stephen A. Hall, An efficient system matrix factorization method for scanning diffraction based strain tensor tomography, Acta Crystallographica A79(6),542-549, version of record online29September2023, DOI https://doi.org/10.1107/S2053273323008136 . Full open primary text is https://pmc.ncbi.nlm.nih.gov/articles/PMC10626655/ ; complete publisher-version XML is archived as quartz_paper.xml from https://www.ebi.ac.uk/europepmc/webservices/rest/PMC10626655/fullTextXML under the article CC-BY terms. Publisher PDF download returned403, so no successful PDF download is claimed. The full text including mathematical content and AppendixB was reviewed. This task binds the strain reconstruction demonstration, not the separate AppendixC coupled orientation extension or the paper large-volume memory benchmark.

Original author code and existing simulation data: https://github.com/AxelHenningsson/flyxdm at remotely verified exact commit b9f473c814204513ab3e8210b96c6dcf6014501d, archived flyxdm_commit.json and flyxdm_tree.json. The original demo/data files are repository author attachments covered by its MIT license, Copyright2023 Axel Henningsson, with no separate conflicting file license. Data URL https://github.com/AxelHenningsson/flyxdm/tree/b9f473c814204513ab3e8210b96c6dcf6014501d/demo/data . The author paper explicitly identifies this supplementary simulation code/data. Data type is author_existing_simulated, never real experiment. The supplied strain.npy is actual author simulation input truth, not a reconstructed reference.

Modern verification backend ASTRA2.5.0 has source commit934920a46781142c2b8f0177007c395eae601783, remotely verified in astra_commit.json. Relevant archived primary source is cuda/3d/par3d_fp.cu, with GeometryUtil3D.cpp, geometry headers and COPYING. Exact source URL https://github.com/astra-toolbox/astra-toolbox/blob/934920a46781142c2b8f0177007c395eae601783/cuda/3d/par3d_fp.cu . ASTRA is GPL-3.0-or-later; the extracted visible kernel preserves original copyright and license notices. Task LICENSE contains full GPL3 terms; NOTICE additionally preserves the original flyxdm MIT license. No ASTRA binary/package or domain operator is redistributed or loaded by the cleaned runtime. This is a local explicit mathematical extraction using general CUDA primitives, with a declared gradient correction, not an unchanged original-paper environment reproduction.

## Archive, unchanged data and hashes

Maintainer archive: D:/yan1/agent/AutoSkill/imaging-101-maintainer/research_batch4_20261021/material_physics. flyxdm/ holds the exact author source and data. All capture scripts, downloaded original source, compiler investigation, full impulse matrices, logs and temporary executed notebooks remain there rather than in the task. The private figure archive is figures/quartz_diffraction_strain_tomography, containing four PNG/SVG/PDF groups, numeric_sources.npz, measurements.json, and figure_manifest.json with per-file hashes. group_manifest.json/report_catalog.json index final group metadata.

| File relative to archive or task | SHA256 |
|---|---|
| flyxdm\demo\data\strainsino.npy | b90f357ca051a7bcc08de7dc9e185a51a8ec69a0505952d45622313816338327 |
| flyxdm\demo\data\gsino.npy | b7b640550676129fb3918c7bc8c60bd881714db14bbb50920aebda6f680a8d2a |
| flyxdm\demo\data\strain_noise_std.npy | 0d9afc64df31188a117d80181258934afee565e8b072f73bb61e94368f4f8823 |
| flyxdm\demo\data\omega.npy | 1277acc4eda8627ea9e6da5b350962ab4843f07bcd99fd6e28690ccd03ae2e45 |
| flyxdm\demo\data\grain.npy | 23740657a32035b489650df1cf6a39b405ffaa6fede1b4690a46261d1d131262 |
| flyxdm\demo\data\strain.npy | 8ee79a8d67d5f193378e5f2663a5fe017715e6ac7078da0dbf6c81c18afcc065 |
| flyxdm\demo\data\ub_field.npy | fdb4bb707684d217bc50938fc1aecd9decb9a3d555941a6b1af46ff06ec9e979 |
| flyxdm\demo\data\UB0.npy | ea85dba6e2c0451b1eb20ffaffc589fb107c03dabf005e3188b345c39e60be21 |
| data\raw_data.npz | 5c3ce517a145b3eb3583a09dcd8095d219fd191c10b9ddeaa22772b247d2a7ef |
| evaluation\ground_truth.npz | 682da89387dd34492d0b7981e43a0c25c16dcc52398afbd74c50b3570f0c2c7c |
| test\fixtures\upstream_stages.npz | aef7d81305083e85fcc5150207bac98ad77317db0b413c70839cde375b706b5c |
| test\fixtures\upstream_projected_basis.npz | 550555c4033b30c561ba0e5aef5ed73c5ebe75c4d451a138d7977912702bab67 |
| test\fixtures\upstream_impulses.npz | c11f35297560cf6af2f1aa8d8e63dad0c8f9370963bfe0b0c7d1444e968be0b3 |
| quartz_oracle_gpu.npz | 43f603ee7bf087c86d471a61b4fec4bdffe990e22e2a9a23fbfba282c36ef5ec |
| quartz_oracle_corrected.npz | 4c0b84247fe46233a326905f7fdfb7772571845c991f16a1151abdc557a25756 |
| quartz_paper.xml | 9f521ce10b39aa1464f6e3af97bb5063b1f29f5ef1baf2d0423ae5f60de527b7 |
| src\joseph_kernel.cu | d2ee1b0f68194caae936f3724a4fa5d82e28e8b839cf94b1e988716801eb88d4 |

Public NPZ arrays are exact batch-axis views of existing strainsino.npy, gsino.npy, strain_noise_std.npy, omega.npy, grain.npy. All float64 and Boolean dtypes are preserved; no recomputation/downcast replaces observations. strain.npy is hidden truth, batch-first(1,6,106,96), float64. Its outside-support values are exactly zero. An independent original utils.to_euler_and_strain recomputation from existing ub_field and the unchanged quartz reference cell agrees with strain.npy to about6.93e-16 (private quartz_ub_truth_audit.npz). This excludes a mislabeled or rotated truth explanation for the early CPU failure. UB fields, unstrained reference arrays, simulation random selections and object construction are not exposed publicly. Known support is the method's explicitly supplied morphology prior. Propagated noise standard deviations are an explicitly known simulation noise prior; they do not encode hidden strain values. Empty-ray std=1 is an author sentinel, not actual camera uncertainty.

No new benchmark observations or generator are shipped. The archive simulate_diffraction.py records the existing author acquisition: quartz reference cell[4.926,4.926,5.4189,90,90,120], wavelength0.2845704100778472angstrom,500events, measured reciprocal-vector noise mixture. The reconstruction does not rerun that generator. Absolute image-length calibration is unnecessary for support-normalized averages; image/detector spacing is one voxel and tensor strain is dimensionless. Physical reciprocal vector scale cancels in its supplied normalized directional contraction. The README defines the exact image row/column and tensor-coordinate conventions without assuming that array orientation can be silently changed.

## Oracle first and retained failed investigations

All commands below were run from the private archive with OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1 and PYTHONDONTWRITEBYTECODE=1. Isolated oracle environment: Windows/Python3.11.11, NumPy2.4.6, SciPy1.17.1, ASTRA2.5.0 with CUDA, xfab0.0.6 for independent truth audit. Matplotlib3.11.2 is an oracle plotting dependency only. Exact original author functions ran on the same observations before cleaned extraction; source numerical arrays and stages were frozen privately. Source source-manager adapters are clearly separated below.

1. `./oracle-env/Scripts/python.exe capture_quartz_gpu.py`: unmodified default GPU MeanProjector, renderer, StrainSampler, DiagonalPrecision and utils.lsq, runtime5.3815655s, exit0; quartz_oracle_gpu.json/npz/log and quartz_stages_gpu.npz. Original repeat capture_quartz_gpu_rerun.py runtime5.63586579999901s, exit0; every coefficient, reconstructed field and forward output is bitwise equal to the first run. This establishes the actual unmodified default source baseline.
2. Official source CPU branch was separately captured before selecting the final backend: capture_quartz.py runtime251.682s, repeat242.785s, repeat bitwise. The CPU branch erroneously calls data3d.delete for CPU data2d resources; a resource-release-only adapter fixes deletion, not its operator. Its strip geometry differs from default GPU Joseph geometry and row orientation. Its support-only yy NRMSE=.453 was unacceptable. A diagnostic tightened optimizer run of1017s did not fix the result; no such changed stopping is used in the final task. quartz_oracle*.npz/log, quartz_stages*.npz and capture_quartz_tight.py preserve this failed branch. A row-reversed truth comparison was diagnostic only; truth and evaluation were never rotated or rescaled.
3. capture_quartz_gpu_geometry.py froze a complete8by10 pixel,5angle,20detector impulse matrix using unmodified original GPU calls. Original CPU line/strip/linear alternatives were compared and do not define the accepted method. capture_quartz_gpu_adjoint.py showed the original GPU BP differs from FP transpose by relative.1510168, max.3775543. This is an actual unmatched adjoint, not a numerical tolerance assertion.
4. audit_quartz_bound_gradient.py tested the actual weighted RBF objective by central finite differences at a nonzero probe and original returned solution, for three directions and eps1e-3,1e-4,1e-5. Original claimed gradients differ by5.52%,4.05%,.094% at the probe; at the returned solution errors are89.2%,92.9%,112%, including wrong direction/sign. Full values in quartz_bound_gradient_audit.json/log. This establishes the necessity of a named mathematical correction independently of source parity.
5. Corrected oracle capture_quartz_corrected.py replaces only A.backward by the transpose of the bound forward RBF Jacobian. Original forward A.forward, data, support, basis, directions/precision, zero initialization, sum-squared objective, L-BFGS-B, effective tol1e-8, maxiter390 and stopping remain unchanged. J is constructed by original GPU projection of all65 original basis images, then original W*S weights. Original FP versus J action has relative1.1039758e-7 rounding difference, consistent with float32 forward evaluation. Corrected source runtime7.9943699s, exit0; repeat7.284358600001724s, exit0, all coefficients/fields/forward arrays bitwise equal. quartz_oracle_corrected.npz/json/log and quartz_rerun_oracle_corrected.* preserve both runs. This is a qualified corrected oracle, not unchanged reproduction. Both stop at108 iterations/114function evaluations with objective.000497801428256455, gradient infinity.02225757, relative-function-reduction message. The source caller passes tol1e-4 but original utils.lsq hardcodes1e-8; final code preserves the actual1e-8.
6. Independent CPU Joseph extraction and simplified CUDA kernels were deliberately rejected as standard backends. CPU matrix norm/projector differences were about3e-5 relative but inverse differences about1.1%L2 due conditioning and default ftol sensitivity. Simplified GPU branches still crossed754 native texture-bin boundaries among9.1million weights, and changed the inverse despite close forward values. prototype_quartz_joseph*, audit_quartz_arithmetic_variants.py, prototype_quartz_cuda*, quartz_cpu_local_prototype.*, quartz_cuda_local_prototype.* preserve these failures. No optimizer tolerance or parity threshold was relaxed.
7. Exact local extraction preserves original CUDA direction templates, native 3D texture sampling and32-slice accumulation expressions. A full unmodified GPU impulse capture, capture_quartz_full_impulse.py, created the96000by10176 sparse forward matrix with9,114,541nonzeros in23.1728692s. The local exact kernel full impulse matrix has the identical9,114,541nonzeros; the difference has nnz0 and maxabs0 (quartz_cuda_exact_full_impulse_audit.json). The first audit aggregation raised max(empty) only after successful capture because every difference was zero; its zero-difference postprocessing branch was corrected and the independently loaded matrices were verified, without changing the numerical capture. The earlier failed simplified kernel matrix and all its nonzero differences remain archived. Original full measured-data stages and projected65basis Jacobian likewise match the final visible kernel bitwise. The final local full inverse coefficients/fields/objective/108iteration stopping are bitwise identical to the corrected original oracle.

## Paper, source, local mathematics and independent tests

| Model/algorithm stage | Paper scope | Original code | Local code | Frozen/independent check |
|---|---|---|---|---|
| Directional strain line average and six contractions | Sec2, Eq4-6 | StrainSampler | weights_and_directions; TensorAcquisition.forward | analytic tensor contraction/engineering-shear factor check; original stage |
| Block directional projection factorization | Sec3, Eq26 | MeanProjector; operator join | mean_project and forward | original full6channel projection and complete impulse fixture |
| Per-ray known precision | Sec4, Eq27 | DiagonalPrecision | weights_and_directions | independent [1,.5,0] scale case; original weight fixture |
| Supported basis rendering | Sec4, Eq28-29; AppendixB | RadialBasisRenderer | radial_basis; render | exact original probe field; support and bias tests |
| Joseph discretization/texture conventions | Reference numerical realization | ASTRA par3d_fp.cu | visible joseph_kernel.cu via general runtime | all80smallbasis impulses, full9.1M private impulse audit, full projected basis fixture |
| Weighted least squares | AppendixB reference implementation | utils.lsq | solvers.reconstruct; model.objective | complete corrected source coefficients/field/iterations; finite-difference bound objective |
| Named matched gradient correction | Independent correction, not a paper claim | corrected oracle A.backward=J transpose | TensorAcquisition.adjoint | independently known inner products, finite differences, original BP defect retained |
| Truth and metrics | Author simulation input | existing strain.npy | independent standalone Judge | known numeric metric case, all6fullfields, malformed/poor/overflow/evaluator tests |
| CPU portable forward diagnostic | Numerical audit only | independently extracted Joseph equations | cpu_diagnostic.joseph_matrix | CPU axis-sum and transpose identity, active notebook probe vs earlier capture |

Fixture consumers: test/fixtures/upstream_impulses.npz is the original complete small geometry capture; test_original_complete_impulse_geometry consumes all GPU impulse outputs and parameter arrays, while the other original CPU impulse variants are diagnostic source measurements documenting rejected discretizations. upstream_stages.npz contains original probe, renderer, projection, directional sampling, weighted forward, source unmatched backward and named corrected backward, precision weights, support normalization/mask, and Jacobian action. test_original_all_stages_and_full_pipeline consumes these arrays, including an explicit nonzero original/corrected-backward discrepancy. upstream_projected_basis.npz is the original GPU65basis projection consumed by that same test. No test imports an oracle package or generates its own expected parity arrays.

notebooks/reference_outputs/corrected_oracle.npz stores the qualified source result; uncorrected_gpu_oracle.npz stores the unmodified default source; original_cpu_diagnostic.npz stores the rejected official CPU result; cpu_rounding_diagnostic.npz stores the earlier explicit CPU-Joseph inverse numerical sensitivity run, with unchanged default optimizer. cpu_forward_diagnostic.npz contains its observed source-probe projection/norm, against which the notebook actively recomputes the CPU probe. cleaned_measurements.npz identifies only fresh local default/short convergence fields and actual histories. Every reference asset has a notebook or test consumer; none is presented as actual truth or supplied to the solver.

Independent physical tests use axis-aligned rays with known exact pixel sums and constant support-normalized averages, explicit tensor matrix contraction and factor-two shear terms, exact bound J transpose inner products, and actual weighted objective finite differences in three independent directions. CPU ray sums and adjoint checks run without a GPU kernel; the full standard reconstruction explicitly requires the validated GPU environment. Jacobian adjoint tolerance1e-12, FD relative1e-4 at eps1e-3, and source parity exact equality were fixed before declaring completion. Large fullcase field differences were investigated rather than hidden within those thresholds.

## Clean environment, explicit GPU exception and resource lifecycle

Cleaned dependencies are standard library, NumPy2.4.6, SciPy1.17.1, Matplotlib3.11.1. Python3.11.11, Windows10.0.26200, NVIDIA GeForce RTX4080SUPER16376MiB, driver610.88, NVCC12.2.91, MSVC compiler19.29.30158/toolset14.29.30133, WindowsSDK10.0.19041.0. quartz_clean_environment.json confirms `find_spec('astra') is None` in the actual shared runtime. All standard CLI/tests/notebook executions used that runtime and not the isolated oracle environment. ASTRA/xfab and oracle directories are absent from the solver import/call path and task requirements.

This is a genuine GPU numerical environment exception: CPU and simplified GPU versions failed strict parity because they changed interpolation-bin or accumulation results. The explicit local kernel is visible source and calls only general CUDA texture/memory/kernel/runtime facilities. cuda_runtime.py invokes a generally installed nvcc/MSVC compiler to build that source, loads the self-compiled DLL using ctypes, and uses the CUDA toolkit's general cudart runtime. It contains no private absolute paths, ASTRA linking, vendored framework or binary solver. Compilation cache is a per-process system temporary directory; _ctypes.FreeLibrary unloads it before recursive removal of the verified prefix directory. Unit test explicitly verifies handle/path provenance, removal, and clean subsequent recompilation with equal output. No DLL/build artifacts are bundled in the task. General compiler/SDK requirements are public, not oracle-only hidden dependencies. Current host compatibility is actually tested; no untested cross-platform compatibility is claimed.

Single-thread CPU environment was maintained; active two-model notebook and CPU diagnostic remain below the coordinated1.5GiB allocation limit. The bound Jacobian alone is299,520,000bytes, original projected basis49,920,000bytes, a CPU-Joseph CSR about73MB. Large-scale factorization memory claims are intentionally not made because the bounded matched-J correction explicitly assembles this compact basis Jacobian. Private full impulse archives are not used as hidden reconstruction operators or target-derived calibration.

## Executed commands, metrics and notebook

From task directory, shared Python with the single-thread variables above:

```
python -m src.main
python evaluation/judge.py --prediction output/reconstruction.npz --report output/judge_report.json
python -m pytest test/ -v -p no:cacheprovider --basetemp <private>/pytest-quartz-final21
python -m nbconvert --to notebook --execute notebooks/quartz_diffraction_strain_tomography.ipynb --output quartz_executed_final.ipynb --output-dir <private> --ExecutePreprocessor.timeout=600
```

Actual executable is D:/yan1/agent/AutoSkill/imaging-101-maintainer/research-runtime/Scripts/python.exe. JUPYTER_PATH=<private>/jupyter selects imaging101-research. IMAGING101_FIGURE_DIR points only to the private final figure archive; it is not a numerical input. CLI first-compilation reconstruction7.916057599999476s, exit0, quartz_cli.log, exactly108iterations and objective.000497801428256455. Standalone and foreign-current-directory Judges both exit0; quartz_judge.log and quartz_foreign_judge.json. Final complete suite21passed in13.07s, exit0, quartz_tests_final.log. The sole SciPy warning is upstream's deprecated disp option; it is recorded, with no numerical failure. Parent independently repeated21tests in13.01s. No required failed check is omitted.

Final active notebook runtime and metrics:

{
  "iterations": 108,
  "objective": 0.000497801428256455,
  "weighted_residual_norm": 0.022311464054527094,
  "standardized_residual_rms_inside_rays": 1.0033853617403523,
  "truth_components": [
    [
      0.9942767320349342,
      0.028540341078467844
    ],
    [
      0.9995032889644002,
      0.020471019164044413
    ],
    [
      0.9988109658254863,
      0.011578087609279947
    ],
    [
      0.9995421944206316,
      0.014986728126204545
    ],
    [
      0.9995133239857824,
      0.013357775303162896
    ],
    [
      0.999619489141162,
      0.010293266718433615
    ]
  ],
  "cpu_projection_relative_error": 2.9054190235910937e-05,
  "fresh_projection_max_error": 0.0,
  "fresh_seconds": 8.279311199999938,
  "short_fresh_seconds": 1.4251499000019976,
  "short_iterations": 20
}

All full-array truth component NRMSEs are approximately[.02854,.02047,.01158,.01499,.01336,.01029], NCC[.99428,.99950,.99881,.99954,.99951,.99962]. These are RMS over the complete declared image including known zero outside-support values. Earlier source audit support-only NRMSE values[.03487,.02501,.01414,.01831,.01632,.01258] are explicitly different-domain diagnostic values, not silently relabeled full-array scores. No mask/alignment/rescaling enters Judge scoring. Each component independently must meet NCC>=.9*corrected_source_NCC and NRMSE<=1.1*corrected_source_NRMSE, as fixed in metrics.json from actual truth. The baseline is strong; zero, huge finite, malformed and missing/evaluator-error outputs are rejected with strict status/JSON/exit contracts. Source-method parity uses exact arrays, not these scientific-quality thresholds.

The notebook actively reconstructs all observations and six fields through local modules, actively runs maxiter20 convergence comparison, recomputes the independent CPU probe projection, saves the fresh declared output and invokes the standalone independent Judge. It asserts exact corrected-source field agreement. Original GPU, corrected source, failed CPU strip and CPU rounding results are separately identified rather than treated as fresh or truth. It shows four actual multi-panel groups: all6truth/reconstruction/error fields in microstrain; original/corrected/CPU metric and actual convergence comparisons; measured/predicted directional sinograms, standardized residuals, precision and directional sensitivity; CPU numerical projection differences, exact CUDA projection parity and inverse error amplification. The selected paper strain truth/REC/RES scientific comparison is preserved for every component. Its separate orientation inverse and large-resolution memory experiments are outside this explicitly bound task scope, not fabricated or silently substituted.

The standardized residual RMS over material-intersecting rays is1.0033853617403523, consistent with the supplied simulation noise scale and retained small approximation mismatch. Empty-ray noise_std=1 sentinel is shown as unavailable in the noise-only plot, explicitly labeled material rays only; the full raw data and evaluation arrays are unchanged. Every actual final PNG was opened and visually reviewed, including the revised physically meaningful noise panel; SVG/PDF and exact numeric sources are exported with hash manifest. No cached array substitutes for active benchmark reconstruction. Full executed outputs are preserved in the canonical notebook; temporary executed notebooks remain private outside the task.

## Semantic, interface and completion review

README defines the complete measurement model, quantities, component units/order, detector geometry, interpolation conventions, explicit GPU numerical environment, data table and batch-first output contract without hidden arrays or public GitHub URLs. The strain hint remains conceptual. approach.md matches exact RBF factors/cutoff, weight normalization, signs/shear factors, objective normalization, matched-gradient correction, init/tol/iterations and evaluation; design.md matches actual callable signatures and data paths. All symbols, factors and execution paths were reviewed semantically, not only by signature matching. Raw LaTeX and ASCII mathematical text avoid dropped backslashes/control-character escapes; final scans cover README/plans/notebook.

Core forward/inverse mathematics is local executable code: explicit kernel integration, support normalization, measured directional contraction, RBF basis, objective, matched adjoint and original optimizer pipeline. General SciPy optimization and CUDA runtime are computation primitives, not imaging-library shortcuts. Source, Judge, CPU diagnostic, tests and notebook contain no domain reconstruction imports, cross-task imports, hidden operator loading or matplotlib backend mutation. Requirements contain only cleaned general dependencies; no metadata JSON exists. Extra src files are genuine mathematical/runtime modules, not duplicate entry-point wrappers. The one CUDA source is required executable mathematics. Task references have documented consumers and original provenance; unused temporary arrays, source checkouts, PDFs/XML, logs and binaries remain private.

Gate status: passed paper/revision/licenses/data binding; original exact-data oracle and deterministic repeat; named corrected oracle and repeat; independent full/native impulse and bound-FD/adjoint/physics checks; explicit local math; constants/precision/coordinates preserved; clean actual runtime and declared GPU/compiler exception; standard CLI; strict standalone and foreign-CWD Judge; all21unit/parity/physics/pipeline/contract tests; full active executed notebook; rich actual comparisons and all4groups visually reviewed; README/plans semantic consistency; public leakage and dependency/call-role audit; final task inventory and private hash/manifest indexing. No failed or unverified mandatory local gate remains. Ready for independent final parent audit; parent acceptance is separately recorded.

Final audit file: quartz_final_audit.json, errors=[]; all task files indexed by SHA256. Audit performed after final21test suite and executed notebook. No packaged compiled DLL, binary object, exported image or temporary notebook exists.


## Parent acceptance review

Parent independently ran final21tests (21 passed in13.01s) in the shared Python environment, including actual first-use CUDA compilation, numerical parity, independent ray sums/adjoint, CPU diagnostic and temporary-DLL lifecycle. Read explicit Joseph CUDA texture integration and wrapper, RBF renderer, tensor contraction, matched coefficient Jacobian and L-BFGS-B settings; inspected all4 actual scientific groups. The author-existing500-event dataset recovers all6 spatial strain components against genuine supplied simulation truth. Qualified corrected oracle replaces only the independently demonstrated mismatched backward operator with the matched65-basis Jacobian transpose, preserving acquisition, initialization and optimizer settings; original GPU and failed CPU branch remain separately labeled. Full96000x10176 impulse matrix has9114541 nonzeros and exact equality; local108-step corrected inverse is bit-identical. Visible extracted kernel contains explicit mathematics and preserved GPL/MIT notices, no ASTRA package or binary dependency. NVIDIA Ada/CUDA12.2/MSVC14.29/SDK10.0.19041 requirement is a documented validated environment exception; CPU arithmetic diagnostics are not substituted for the standard inverse. One SciPy disp deprecation warning is retained, not represented as an error-free log.
