# Provenance and validation: cell-membrane FRAP transport

Validation date:2026-10-27. Status: standard selected diffusion path validated locally; awaiting batch integrator review. No independent physical truth is asserted. Original239 release tasks are unchanged by this private staging work.

## Binding, novelty and measured-data evidence

Unknowns: lateral diffusion coefficient and the resolved41×89 optical recovery field. Observation: all42 frames of the author's `HiFRAP_v0/HiFRAP/Examples/Sample_1D.tif`, native519×408 uint16. This is an active bleaching/transport inverse problem. Closest old binding is `staphylococcus_membrane_diffusion_maps`, which infers transport from passive fluctuation/correlation measurements; the acquisition perturbation, Green-function inverse initial-field treatment and observation operator differ. It is not merely a different sample or regularizer for that task.

Classification: **author_real_processed**. Primary source: the author installation guide explicitly calls the included1D and2D TIFFs “FRAP experiments”; its worked ROI instructions and the native ImageJ metadata/camera intensities support that classification. The package's `params_1D.csv` and historical `Examples/Output/Results.csv` identify this exact file, all42 frames, the89-pixel line,0.0431µm/pixel and30s interval. TIFF ImageDescription reports ImageJ1.54f,42 slices, native image display range95–144. There is no data-generation step or synthetic augmentation in the bound observations.

Important limitation: the individual file does **not** independently name its organism/protein or microscope model. The paper's experimental section describes fission-yeast Mtl2p imaging on a Nikon spinning-disk instrument, but that section reports7–10s intervals, whereas this file's explicit parameters say30s. We do not claim those identities or overwrite the file-specific calibration. The public problem is therefore called cell-membrane FRAP, not Mtl2p/yeast FRAP. The distributed TIFF is a processed experimental file, not claimed to be untouched camera firmware output. This limitation is explicit in README and notebook.

## Primary sources and exact revisions

* Paper: Lorenzetti et al., *Biophysical Journal*124,2941–2960, published16September2025, DOI10.1016/j.bpj.2025.07.036, accepted30July2025. Publisher DOI: https://doi.org/10.1016/j.bpj.2025.07.036 . Author-hosted publication PDF: https://www.minclab.fr/wp-content/uploads/2025/10/73.pdf . The author PDF endpoint had a local TLS certificate-chain problem; a public unauthenticated retrieval with certificate verification disabled was used solely for that PDF and its SHA256 is recorded. No authentication material was transmitted.
* Author code/data: https://github.com/lorenzetti1996/HiFRAP_project , remote tree/commit **239b992e33a550a2227681966ed51ff494ba1fe3**, verified through the GitHub API on validation date, not inferred from a stale local branch. Selective `HiFRAP_v0.zip` retrieval at that exact commit supplies unmodified Python functions, Fiji macro, example TIFFs and historical outputs.
* Author guide: `InstructionHiFRAP_FijiMacro_v0.pdf` at the same commit. Per-file parameters and historicalCSV remain in the private source archive; historical numeric results are never used as evidence of fresh execution.
* ImageJ1.54f oracle: https://repo.maven.apache.org/maven2/net/imagej/ij/1.54f/ij-1.54f.jar . Corresponding sampling formula source is https://github.com/imagej/ImageJ/tree/v1.54f . ImageJ is public domain. The final runtime uses explicit NumPy sampling, not Java/ImageJ.
* License: author repository's exact statement is Creative Commons Attribution-NonCommercial-ShareAlike4.0. Code adaptations and redistributed observations retain this license and attribution in task `NOTICE`; downstream use must respect its noncommercial/share-alike terms. The publication PDF is archived privately and not redistributed with the task.

## External maintainer archive index

Archive identifier: `research_batch10_20261027/bio_med/`, outside the final task. Relative entries:

| Entry | Role |
|---|---|
| `HiFRAP_v0.zip`, `hifrap_archive/` | Exact original package and extracted source/data, including historicalCSV |
| `hifrap_guide.pdf`, `hifrap_guide.txt` | Author instructions and extracted reading text |
| `hifrap_paper.pdf`, `hifrap_paper.txt` | Published paper and extracted reading text |
| `ij.jar`, `Straightener.java`, `PolygonRoi.java`, `ImageProcessor.java`, `FloatProcessor.java`, `ShortProcessor.java`, `DownsizeTable.java` | Independent ImageJ runtime/source for sampling verification |
| `straighten.ijm`, `line_FRAP.tif`, `line_control.tif`, `strip_frap.tif`, `strip_control.tif`, coordinateCSV files | Original ImageJ sampling capture, all frames |
| `capture_frap.py`, `original_capture.log`, `frap_oracle.npz`, `oracle_records.json` | Independent original run/stage fixtures under documented ARPACK initialization |
| `original_capture_default_rng.log`, `frap_oracle_default_rng.npz`, `oracle_records_default_rng.json` | Original stochastic-initialization numerical ensemble |
| `investigate_frap.py`, `fixed_rng_investigation.json`, `probe_sampling.py` | Controlled conditioning/RNG and exact sampling investigations |
| `capture_comparisons.py`, `comparisons_oracle.npz` | Original diffusion/reaction/reaction-diffusion comparisons |
| `pre_extraction_contract.json` | Frozen acceptance criteria before local extraction |
| `oracle-env/` | Isolated upstream-only pandas/tifffile/opt_einsum dependencies |
| `runtime_environment.json`, `runtime_backend.txt`, `source_audit.json`, `hashes.json` | Environment, backend, signatures/import roles and file hashes |
| `figure_review/figure1.png` … `figure6.png` | Extracted canonical notebook outputs for visual inspection |
| `jupyter/` | Private kernel configuration for the validated Python interpreter |

### SHA256 identities

| Asset | SHA256 |
|---|---|
| Original archive | `488e300057d6594cce51d2c8ec2820f877844e413c975e75f897a7b88e72a319` |
| Author guide | `ba11153449dd13732790ac8dc7a435a2d37f7641fa08062e23a764e86801cebc` |
| Paper PDF | `e56e49abb0fa1275a11e9eb0cb84a66821c35f66813c6d613e21ae6b61c844ff` |
| ImageJ JAR | `e5ee9b846e3c2f819d945e6d3fca5641ca690b1ef28591415ae0dc7c6d82fcd9` |
| Native bound TIFF | `bd289d55fc7322534457ce9a4a1657a9ea14c334ce20fd9e5a4e844dd2e95996` |
| Original extracted FRAP line | `2c0e80397882ca1fd34e1a73456ee8874d018856b51a9dfd43d582bd258dcf8f` |
| Original extracted control line | `b8b5b1ae00f2be30ace1d9feb4a4e84c6d722f09518ea4670471d1be77ab7e54` |
| Original deterministic capture | `c9f0dcda8586b94e468c0db92416c9a3bb10cf9dd0cf4a6e49cd62a8e5a03a37` |
| Original default-RNG ensemble | `6b7aab863e7d85be89c07a633a572fff04b11e90d14c22e46f9d6d24bb72f5df` |
| Original comparison models | `d8b18045120c581aea3737f30b084bb3b3b6e1c7a30f3d3a38d5e7c4d5c0132e` |
| Frozen pre-extraction criteria | `c822d44580d168a72890bd546045a618f1f602b853e6f8ebd3433b0ca5841b65` |
| Public raw NPZ | `bd855962ec6c458237e22cb76e6563b1d0c3e30e384f1235d484390e62a6b258` |

NPZ conversion adds a batch dimension without downcasting or modifying native observations. Sampling coordinates are the original ImageJ float32 centerline geometry. They are derived from the supplied observation ROI, not from hidden diffusivity or reconstructed fields. The original TIFF contents and public movie are exactly equal. Data file has no inferred answer, privileged second acquisition or simulation seed.

## Original executions before extraction

Run from the private archive with one-thread OMP/MKL/OpenBLAS and Agg backend:

```text
java -jar ij.jar -batch straighten.ijm
oracle-env/Scripts/python.exe hifrap_archive/HiFRAP_v0/HiFRAP/PyMacros/main.py 1 1 1 1 1 1 2 42 original_diff 63,50,204,452 0,0,0,0 0,0,0,0 247,246,246,246,253,262 258,243,230,216,198,190 257,265,272,282,290,303,312,316,316 383,398,404,406,408,405,396,390,381 9 9 30 0.0431 hifrap_archive/HiFRAP_v0/HiFRAP/Examples/Sample_1D.tif ./ 0 0 1 0 0 0
oracle-env/Scripts/python.exe capture_frap.py
oracle-env/Scripts/python.exe capture_comparisons.py
```

Original main exited0, total process elapsed3.54s, diffusion5.955147313521103e−5µm²/s, adjustedR²0.8396954398671057. This first default initialization result is a reproduced original measurement, distinct from cached authorCSV diffusion5.9314912594303506e−5. Independent later default-RNG runs ranged5.917241290641326e−5 to6.145885086171916e−5. The ridge is of order machine precision; small largest-singular-value differences affect optimization stopping along a shallow profile objective. No ridge relaxation or physical-model replacement was made.

Controlled investigation: supplying `rng=np.random.default_rng(1729)` at the SciPy svds API boundary gave three identical original runs, D5.993295324677043e−5, cost0.2032763322763038, residual trace336.63982186686394, adjustedR²0.8397065163161715. Recorded original numerical execution times0.1604256,0.1599715,0.1602237s, excluding process startup. The original HiFRAP functions remain unmodified; this initialization is a documented execution convention, not a claim about the unmodified default RNG behavior. Capture includes preprocessing, compression, Green Gram kernel, fixed-parameter residual and final fitted field.

The local sampling initially exposed a wrong assumption: a plain average across the9-pixel strip does **not** match ImageJ's antialiasing. The independent ImageJ DownsizeTable implementation establishes triangular weighting with replicated boundaries; implementing that formula gives exact equality for all raw strip pixels and both complete extracted profiles. The error was resolved before accepting local sampling parity.

## Environment and resource evidence

Cleaned environment: Python3.11.11, NumPy2.4.6, SciPy1.17.1, Matplotlib3.11.1, Windows10 build26200, CPU. Backend details are recorded by np.show_config in the private archive. Numerical thread variables are all1. The oracle environment adds pandas3.1.0, tifffile2026.3.3, opt_einsum3.4.0; these are not imported or required by the cleaned task. Existing shared packages were not upgraded. ImageJ1.54f runs under installed Java8 only in the private oracle capture.

No GPU is required. Before work the machine reported about13.9GiB free physical RAM and28.3GiB free virtual memory, above the required margin. The bound movie is about17MB uncompressed; runtime matrices are369×369. CPU execution is directly verified, not inferred from package availability.

## Equation/function/test traceability

| Required stage | Paper/upstream implementation | Local implementation | Independent validation |
|---|---|---|---|
| Line sampling and antialiasing | Author Fiji macro Straighten/Size; ImageJ Straightener and DownsizeTable | preprocessing.sample_line | Whole42-frame original profile fixtures; uniform-field analytic test |
| Camera normalization/background/control loss | Paper acquisition preprocessing; preprocessing.compute_preprocessing | preprocessing.prepare | Exact profile/normalized arrays and captured epsilon/background |
| Orthonormal real Fourier compression | Paper compression operator; functions.compress_matrix/fourier_compression | physics_model.compression_matrix | Original matrix/data fixtures; row orthogonality, DC and adjoint checks |
| Diffusion Green operator/initial-field Gram | Paper diffusion equation and kernel/shift table; compute_GGT_0_q_1D_henkel_tensor | physics_model.heat_gram | Original fixed-parameter kernel; independent Gaussian quadrature convolution; PSD/symmetry and loss-time factors |
| Ridge marginalization and effective trace | Paper profile objective; compute_pseudo_inverse_choleski/project_residuals/residuals_reduced_K_full_lapack | physics_model.profile_residual | Original fixed-parameter residual fixture; kernel/linear-algebra mathematical review |
| Trust-region diffusion inference | optimisation.runAnalysis selected diffusion/control/background branch | solvers.reconstruct | Original final field and numerical ensemble envelope; complete CLI pipeline |
| Reaction alternatives | Same upstream model2/model3 branches, known prebleach state | reconstruct optional model | Original and fresh model costs/fields visibly compared; weak parameter instability disclosed |
| Optical field reconstruction | plot_results.plot_data_vs_fit | reconstruct field return, visualization.recovery_figure | Original fitted field; no initial-field answer supplied |
| Evaluation/display metrics | Independent mathematical definitions | evaluation.judge.metrics; visualization.agreement | Analytic known numbers, zero estimate, degenerate reference, strict CLI contracts |

The optional reaction parameters are weakly determined: for example, the fresh reaction-only loss differs from the oracle despite similar cost and field. These optional comparison parameters are not claimed to have strict parity and are not benchmark outputs. The notebook reports the actual differences. The mandatory selected diffusion path meets all pre-frozen numerical and physical checks.

## Hidden arrays and consumers

* `evaluation/baseline_reference.npz`: original-execution `reconstruction`(1,41,89) float64 camera a.u. and `diffusivity`(1,1) float64µm²/s; consumed only by standalone Judge. Actual truth is unavailable, so no ground_truth.npz is invented.
* `test/fixtures/upstream.npz`: original `output_profile`, `output_control`, `output_normalized`, `output_compression_matrix`, `output_compressed`, `output_kernel`, `output_fixed_residual`; each is consumed by test_independent_upstream_stages. No locally generated expected arrays.
* `notebooks/reference_outputs/upstream.npz`: labeled original field/D comparison consumed by active review notebook.
* `notebooks/reference_outputs/comparisons.npz`: independently executed original diffusion/reaction/reaction-diffusion fields and diagnostics consumed by the model comparison.
* `notebooks/reference_outputs/cleaned_measurements.npz`: fresh canonical notebook field/D/cost/runtime and three comparison fields; retained as identified fresh measurement evidence, not used as a reconstruction cache.

## Threshold rationale and measured results

Before extraction, `pre_extraction_contract.json` froze full-field NCC≥0.9999, dynamic-range NRMSE≤0.003 and relativeD error≤0.05. The nonzero D tolerance encloses the independently observed default-RNG numerical envelope and is smaller than the approximate original physical standard error (~11%). The field bound restricts weak modes to the declared optical bandwidth. Stage tolerances include compression absolute1e−5, kernel relative1e−12, residual absolute0.0003; sampling in fact agrees exactly. This is not self-comparison with zero-error thresholds and not a statement of independent biological accuracy.

Fresh local notebook run: D5.972747858553252e−5µm²/s; NCC0.9999999992942029; NRMSE0.0001297644865760305; relativeD error0.0034284087485541015. All pass. Fresh cost0.20330644810086113 and adjustedR²0.839682783507744. Complete standard reconstruction took0.1940326s inside the active notebook. Ignoring imaging loss raised inferredD to9.638023468325415e−5 and worsened residual cost, demonstrating that the independent control correction materially affects the inverse result.

## Executed checks and notebook review

```text
python -m src.main
python evaluation/judge.py --prediction output/reconstruction.npz --report output/judge_report.json
python -m pytest test/ -q --basetemp=<private-maintainer-temp>
python -m jupyter nbconvert --to notebook --execute notebooks/cell_membrane_frap_transport.ipynb --inplace --ExecutePreprocessor.timeout=180
```

All ran with the validated research interpreter and one-thread environment. Pytest: **19 passed in3.14s**. The system default pytest temp folder was inaccessible(WinError5); the same tests succeeded with an explicit private writable basetemp. This is a host temporary-directory exception, not a scientific test failure. StandaloneJudge report is pass with empty errors. Contract tests cover good/poor submissions, missing file/key, shape/dtype/complex, NaN/Inf, missing evaluator assets, report-parent creation, report-write failure and caller-working-directory independence.

Native nbconvert exited0 in7.44s, all code cells executed, no error outputs. Timeout180s accommodates process startup and comparisons. Canonical notebook retains executed outputs in place; no duplicate executed notebook remains in output. Harmless host warnings concerned frozen-module debugger support, a Windows zmq selector thread and local kernel transport; no reconstruction warning or failed cell occurred.

All6 figures were visually inspected from canonical outputs: acquisition ROI/control/recovery; Fourier basis/orthogonality; full diffusion objective; five-panel measured/compressed/original/fresh/residual kymographs; fresh three-model comparison; no-bleaching ablation/residual histogram/time traces. Labels/units, shared intensity scale, negative-loss interpretation, colorbars and figure layout are legible. The notebook preserves the selected experimental scientific comparisons. Paper-wide synthetic validation ensembles are outside this single measured-data task and are explicitly not claimed as reproduced.

## Consistency and completion gates

Signatures were extracted with ast and compared to design.md. Mathematics was semantically reviewed for pixel/frame conversion, Gaussian variance, sum-of-times Gram kernel, background sign, optical loss, Fourier row ordering, ridge grouping, trace normalization, source/loss convention, optimizer defaults and field restoration. Imports/call roles were audited across source, Judge, tests and notebook: only allowed general computation tools; explicit acquisition/Gram/objective mathematics; no domain-library solver, cross-task import, oracle runtime import or hidden-reference access by src. No matplotlib.use call exists in any library or main import path. Public README/requirements/data metadata contain no GitHub URLs or hidden targets. Runtime requires only documented NumPy/SciPy/Matplotlib; development tooling stays in the shared environment.

| Gate | Result |
|---|---|
| Paper/revision/data hashes/equation mapping | Passed |
| Original and cleaned same bound observations | Passed |
| Independent stage parity and applicable physics | Passed |
| Constants/units/precision/coordinates preserved | Passed |
| Standard main and fresh Judge | Passed |
| Judge negative/error contracts | Passed |
| Public inputs sufficient and no hidden-answer leakage | Passed |
| README acquisition/output contract, no metadataJSON | Passed |
| Minimal CPU runtime without domain packages | Passed |
| Local explicit forward/inverse mathematics and import-role audit | Passed |
| Unit/metric/Judge/parity/physics/pipeline tests | Passed,19 tests |
| Active six-section notebook and all figures inspected | Passed |
| Historical/original/fresh distinction; justified metrics | Passed |
| Plans/README/source semantics and signatures | Passed |
| Template inventory, private archive and license | Passed after generated-cache cleanup |

Disclosed non-gates: per-file organism/instrument identity is unavailable; independently measured molecular truth is unavailable; optional reaction-model scalar parameters remain poorly identified. These are not silently claimed as verified. Root acceptance and final release integration are separate from this local validation record.


## Independent parent acceptance

CLI, standalone Judge, pytest, active nbconvert and dual-engine math rendering were independently rerun: all exit0. Parent notebook: 7 executed code cells, 6 computed PNG outputs, 50 formulas. Private evidence: research_batch10_20261027/acceptance/cell_membrane_frap_transport/execution.json and semantic_review.json. All panels actually inspected by parent; source/physics/GT roles and limitations semantically checked. Canonical notebook is this parent-executed artifact. No new source or notebook execution is inferred from cached upstream outputs.
