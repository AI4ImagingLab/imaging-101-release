# Source, mathematics and validation evidence

Status: ready for independent parent audit, 2026-10-21. This is a new spatial observation-to-target task reserved by the parent as `raw_burst_radiance_reconstruction` in mathematics. It reconstructs a1024² latent Bayer image, with motion as a nuisance, rather than delivering registration alone. Existing single-image Bayer demosaicking, generic video decomposition and calibration tasks have different observations/unknowns. The first four biology/medicine tasks are independently accepted and frozen; this fifth package is separate.

## Paper, exact source and real observations

Primary selected method: Antoine Monod, Julie Delon, Thomas Veit, *An Analysis and Implementation of the HDR+ Burst Denoising Method*, IPOL11,142–169(2021), DOI10.5201/ipol.2021.336. The current publisher paper includes the2022-01-01 editor-information update;28 pages, low-resolution PDF private `paper.pdf`, downloaded from https://www.ipol.im/pub/art/2021/336/article_lr.pdf. Paper version history is archived in paper_record.html. Sections3–4/Algorithms1–2 and equations3–13 supply the selected alignment/noise/temporal/spatial/window mathematics. The original2016 Hasinoff et al. paper DOI10.1145/2980179.2980254 is historical context, not a claim that proprietary Google2016 source was executed.

Author source: https://github.com/amonod/hdrplus-python exact remotely verified commit `98ebf1724196bc070c1248e0f8efa2df6c18b8ef`. Parent initial local/remote pin evidence is at sibling root/source_inspection.json; a fresh REST commits/HEAD response in `remote_revision.json` independently matches that full hash. Source checksum rows below identify original and minimally corrected files separately. Original AGPL3-or-later copyright2021Antoine Monod and patent notice are preserved in adapted source headers, LICENSE and NOTICE. No shared file or previously accepted task is modified.

Real measured dataset: author-deposited `test_data/33TJ_20150606_224837_294`, all ten `payload_N000.dng` throughN009. The HDR+ dataset page https://hdrplusdata.org/dataset.html explicitly links Creative Commons Attribution-ShareAlike4.0 https://creativecommons.org/licenses/by-sa/4.0/; the exact link/version is archived in dataset_license.html. Source paper also publishes this ten-DNG supplementary test burst. NOTICE and public README preserve dataset authors, source, license and modification statement. Adaptation is fixed central1024² crop and lossless NPZ repackaging, under CC BY-SA4.0 for data. No synthetic benchmark observations were generated; analytic unit-test fields are mathematical checks rather than released observations.

Each original DNG array is3120×4208 uint16ADU, BGGR, black64 and white1023. Before original-oracle execution or local extraction, the same even-aligned native ROI was fixed: rows1048:2072,columns1592:2616. All ten exposures are retained, no resize, prealignment, hidden-answer-derived calibration or artificial noise. The public file contains only frames uint16(1,10,1024,1024). Original DNG decompression occurs privately with rawpy; cleaned runtime reads NPZ only.

`reference_frame.txt` contains0. The last frame is not selected as truth or as a privileged exposure reference by the supplied author pipeline. Native metadata check of every DNG, archived `all_exposure_metadata.json`, shows frames0–8: ISO1143, exposure297140608/4294967295s≈0.0691834390s; frame9: ISO1551, exposure429466976/4294967295s≈0.0999930725s. This contradicts a same-exposure simplification, so README/notebook/plans explicitly model photometric mismatch. The selected original algorithm iterates over every DNG, does not normalize this last exposure and uses reference noise parameters. That approximation is preserved rather than silently discarding/rebalancing a frame. Independent first9/all10/DC-weight/brightness-contribution experiments below quantify this limitation. No absolute radiometric accuracy or true clean scene is claimed.

## Original-first run and two investigated corrections

Private archive root `D:/yan1/agent/AutoSkill/imaging-101-maintainer/research_batch4_20261021/bio_med/hdrplus` contains capture.py, original and corrected complete source snapshots, paper, independent numeric fixtures, logs and figures outside the released task. The original mathematical implementation executed on the exact selected ROI before local code construction. First unmodified trial used utility temporal8, with alignment0.4427417s/merging1.5689029s; reading the selected2021 CLI/paper then fixed temporal75 before freezing the selected source75 fixture. The CLI default75 is distinct from the utility function default8; the paper section5 explicitly discusses that choice. The final pipeline and all final fixtures use75, spatial0.1, four original scales and all ten frames. Fresh corrected8 is separately labeled only as a comparison.

Original source75 was executed unchanged mathematically (only deprecated NumPy aliases restored privately and generic shared PyTorch/Matplotlib made available via sys.path). Latest original core times: alignment0.4195039000005636s, merging1.3879678000012063s. The first shell-redirection wrapper reported a PowerShell NativeCommandError on a harmless Python SyntaxWarning about `is` string comparisons; the original program nevertheless completed and saved arrays. Subsequent captured Python subprocesses establish actual exit0, with full stdout/stderr separately recorded. No successful notebook import or cached result is substituted for an original run.

Independent paper/code checks revealed two specific correctness defects. Source `subPixelMinimum_` interpreted nine contiguous flattened values from a9×9 search-cost surface as its3×3 neighborhood; the independent quadratic proof uses a true two-dimensional surface with known optimum. Source `patchesRMS` squared uint16 values before promotion, wrapping values above255ADU. Diagnostic results:
```json
{
  "quadratic_original_offset": [
    -0.2111111432313919,
    0.833333432674408
  ],
  "quadratic_true_offset": [
    -0.3,
    0.2
  ],
  "quadratic_true3x3_original_filter_offset": [
    -0.30000001192092896,
    0.20000000298023224
  ],
  "uint16_rms_original": 163.38910612400082,
  "uint16_rms_true": 353.5533905932738,
  "bound_rms_relative_error": 0.0023305993206786607,
  "bound_patches_with_wrong_rms": 2,
  "bound_total_patches": 4225
}
```
The exact minimal patch is private `minimal_correction.diff`: actual2D neighborhood offsets and full row/column validity margin; float64 multiplication for RMS. No parameter, regularizer, source-selection method or alignment/merge family is replaced. Original unmodified source75 and corrected source75 fixtures remain separately identified and immutable in the task. This is explicitly a corrected implementation, not unchanged reproduction. The corrected original Python/Numba code is an isolated independent oracle; cleaned code replaces NumPy/Numba kernels with local explicit mathematics and imports no upstream package.

Latest corrected oracle core times: alignment0.4161414000009245s, merge1.430009700001392s. Original-source75 and corrected-source75 independent repeats are each exactly identical to their separately frozen outputs (relative L2=0). Local corrected full image, all36 scale motion arrays, native motion, aligned sample and reference pyramids match the corrected oracle exactly.

Two extraction differences were investigated without relaxing stage tests: original L2 arithmetic subtracts/squares in float32 before float64 accumulation; prematurely squaring in float64 changes discrete minima in flat regions. Original Wiener ufunc dispatch promotes reference/alternate coefficients to complex128 before subtraction because variance is float64; subtracting in complex64 changes temporal stage values by up to1.68e−5 despite identical final ADC quantization. Preserving both conventions resolves the frozen stage tests at their original1e−12 tolerances.

## Equation/function/test traceability

| Paper equation or algorithm stage | Original source/function | Explicit local implementation | Validation consumer |
|---|---|---|---|
| Section3.2, Algorithm1 Gaussian/Bayer pyramid and overlapping blocks | alignment.py:hdrplusPyramid/alignHdrplus; imageUtils.py:downsample/getTiles | preprocessing.downsample/getTiles; solvers.hdrplusPyramid/alignHdrplus | Four frozen reference pyramids, Bayer average, all36 frozen scale arrays |
| Section3.2 propagated search candidates and L1/L2 | alignment.py:upsampleAlignments/alignOnALevel; imageUtils.py:computeDistance/computeTilesDistanceL1_ | solvers.upsampleAlignments/alignOnALevel; physics_model explicit distances | Native motion/aligned sample parity; analytic known shift and clipped pullback adjoint |
| Equations3–6 actual2D quadratic displacement | imageUtils.py:subPixelMinimum_ with documented indexing correction | physics_model.subPixelMinimum | Independent9×9 known quadratic; all36 corrected-original arrays |
| Equations7–9 ISO noise and tile RMS | merging.py:getNoiseParams/patchesRMS with documented safe-square correction | fixed independent NOISE tuple; patchesRMS | Known[300,400]RMS, original frozen RMS, no-overflow Bayer mean |
| Equations10–11 pairwise temporal Fourier shrinkage | merging.py:mergeWienerDFTPatches/temporalDenoisePairPatches/mergeChannelHdrplus | Same named local formulas, general PyTorch FFT | Independent frozen reference/pairFFT sample; temporal0/identical-tile limits |
| Equation12 radial spatial Wiener filter | merging.py:spatialDenoisePatches | Local radial shaping/closed attenuation | Independent spatialFFT sample; DC preservation/attenuation test |
| Section4.4, equation13 cosine overlap reassembly | centeredCosineWindow/cosineWindow2Dpatches/depatchifyOverlap | Same named explicit local functions | Analytic partition of unity and constant interior; full reconstruction parity |
| Photometric mismatch/last-frame contribution review | Original author all-frame loop and declared model approximation | visualization.alternate_dc_weights/run_experiments | Fresh first9/all10, shared-scaleDC weight maps, ADU contribution map/per-site bars |

Plans were semantically reviewed against the actual source: signatures/defaults/paths; row-column indexing, Bayer parity, finite boundaries, direct formulas, quantization, Fourier normalization, source precision and correct reference units. README required sections/tables and visible key/shape/dtype/scalar metadata are covered by public-contract tests. Reconstruction stops after four finite scales and direct filters; no fictitious iterative convergence or optimizer is documented.

## References, fresh experiments and numerical metrics

No independent clean truth is available. evaluation/baseline_reference.npz is the independently executed corrected standard method. notebooks/reference_outputs/corrected_upstream.npz is intentionally a second named copy for notebook comparison; unmodified_upstream.npz preserves original source75; cleaned_metrics.npz stores separately identified freshly measured scalars consumed/checked against active notebook recomputation. test/fixtures/oracle_stages.npz contains only independently frozen corrected source stages; every stored key is consumed in test_original_parity.py. The private aligned_private.npz files are original-oracle archives only and never solver input/imports. Fresh output/reconstruction.npz is written by CLI, not copied from a reference.

Full-precision fresh actual measurements:
```json
{
  "corrected_upstream_relative_l2": 0.0,
  "unmodified_source75_relative_change": 0.027214062377598963,
  "single_reference_relative_error": 0.12680487363286258,
  "no_alignment_relative_change": 0.06010894401256488,
  "spatial_filter_relative_change": 0.007816531631912201,
  "last_frame_relative_contribution": 0.01485130082610768,
  "last_frame_mean_brightness_change_ADU": 0.08065128326416016,
  "frame1_mean_alternate_DC_weight": 0.9429084405849525,
  "frame9_mean_alternate_DC_weight": 0.443799544465267
}
```
The differing last exposure’s average alternate DC weight is0.443799544465267 versus0.9429084405849525 for the equal-exposure frame1, before the1/10 temporal mean. First9 versus all10 image difference is0.01485130082610768 relative L2; mean brightness contribution is+0.08065128326416016ADU, with per-site bars and a spatial difference map. This is measured contribution, not absolute true brightness error. The unmodified/corrected difference0.027214062377598963 demonstrates the correction is not mislabeled numerical noise. No independent-truth PSNR or simulated biological/radiometric accuracy is invented.

Five computed multi-panel groups preserve measured frames, common-scale same-region method comparisons, temporal8/75 and spatial on/off comparisons, motion, original/corrected difference, frame-count1/2/5/9/10, spectra and explicit last-frame influence. Preview uses2×2 Bayer mean minus64 only for labeled display. The native1024² submitted array remains unchanged. Figures1–5 were opened and inspected for clipping, signs, units and complete comparison labels; title wrapping and quiver row direction were corrected before final execution. The native-code approximation does not guarantee accurate motion everywhere in noisy/flat or different-exposure regions; that is not presented as geometric truth.

Private gallery/ exports PNG/SVG/PDF, numeric_sources.npz with every plotted spatial solution/weight map/count/motion and metrics.json. Manifest hashes bind all plots to these computed numerical sources. There are no standalone PNG/PDF files inside the task.

Metric: full-array relative Euclidean error sqrt(sum(abs(estimate-reference)²))/sqrt(sum(abs(reference)²)), cast to complex128 before unsigned subtraction. This is explicitly relative L2, not dynamic-range NRMSE or mean-subtracted NCC. Boundary1e−5 was fixed before final scoring: original repeats have zero spread and exact local parity, so this nonzero floor allows only tiny numerical/ADC effects. It is far below observed noisy reference0.1268, omitted-alignment0.0601, unmodified-source0.0272 and omitted-spatial0.00782 differences. All pixels/native dtype are required; there is no hidden rescaling, crop, alignment, magnitude conversion or answer-derived calibration. Zero or unsigned-max fields cannot pass. Undefined zero/nonfinite reference norms and overflow fail explicitly. This score assesses the selected standard method, not physical correctness on its own.

## Actual environment, commands and notebook display verification

Cleaned runtime: Windows10.0.26200 x86_64, Python3.11.11, NumPy2.4.6/OpenBLAS0.3.31.188.0, SciPy1.17.1, PyTorch2.9.1+cu128 used on CPU, Matplotlib3.11.1. requirements.txt contains only these used pinned numerical/plotting packages. Shared runtime was not modified/upgraded. NumPy/SciPy supply general filtering/interpolation/strides; PyTorch supplies general FFT, no domain inverse. pytest9.1.1 and nbconvert7.17.1 are shared development tools.

Original oracle runtime is sibling root/oracle-env/Scripts/python.exe with NumPy2.4.6, SciPy1.17.1, rawpy0.27.1, numba0.68.0; generic shared PyTorch/Matplotlib are appended only privately. ExifRead3.5.1 was installed solely under this private hdrplus/oracle_extra folder, not into the shared runtime or parent environment. Full private freeze is oracle_freeze.log, exact live numerical backend in clean_environment.log. PDF extraction uses separate private pypdf. Cleaned solver/test/Judge/notebook imports none of rawpy/numba/cv2/exifread/original package, and no cross-task code.

All commands use OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=NUMBA_NUM_THREADS=1 and PYTHONDONTWRITEBYTECODE=1; CLI/notebook explicitly set torch CPU threads1. Memory snapshot: `{"FreePhysicalMemory": 17048692, "FreeVirtualMemory": 37270696}` KiB, both free physical and commit exceed6GiB. Largest full aligned-tile array is approximately86.5MB; no allocation/download exceeds coordinated limits. Dataset was already provided by parent, no large download.

| Stage | Exact argument list | Exit | Runtime(s) | Private log |
|---|---|---|---|---|
| `unmodified_source75_repeat` | `["D:/yan1/agent/AutoSkill/imaging-101-maintainer/research_batch4_20261021/root/oracle-env/Scripts/python.exe", "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research_batch4_20261021\\bio_med\\hdrplus\\capture.py"]` | 0 | 21.915596 | `unmodified_source75_repeat.log` |
| `corrected_source75_repeat` | `["D:/yan1/agent/AutoSkill/imaging-101-maintainer/research_batch4_20261021/root/oracle-env/Scripts/python.exe", "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research_batch4_20261021\\bio_med\\hdrplus\\capture.py", "--corrected"]` | 0 | 22.107710 | `corrected_source75_repeat.log` |
| `cli` | `["D:/yan1/agent/AutoSkill/imaging-101-maintainer/research-runtime/Scripts/python.exe", "-m", "src.main"]` | 0 | 5.913558 | `cli.log` |
| `judge` | `["D:/yan1/agent/AutoSkill/imaging-101-maintainer/research-runtime/Scripts/python.exe", "evaluation/judge.py", "--prediction", "output/reconstruction.npz", "--report", "output/judge_report.json"]` | 0 | 0.161949 | `judge.log` |
| `tests` | `["D:/yan1/agent/AutoSkill/imaging-101-maintainer/research-runtime/Scripts/python.exe", "-m", "pytest", "test/", "-v", "-p", "no:cacheprovider", "--basetemp", "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research_batch4_20261021\\bio_med\\hdrplus\\pytest-final"]` | 0 | 8.512861 | `tests.log` |
| `notebook` | `["D:/yan1/agent/AutoSkill/imaging-101-maintainer/research-runtime/Scripts/python.exe", "-m", "nbconvert", "--to", "notebook", "--execute", "notebooks/raw_burst_radiance_reconstruction.ipynb", "--ExecutePreprocessor.kernel_name=imaging-101-research", "--ExecutePreprocessor.timeout=600", "--output", "hdrplus_verified.ipynb", "--output-dir", "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research_batch4_20261021\\bio_med\\hdrplus"]` | 0 | 32.758662 | `notebook.log` |
| `gallery` | `["D:/yan1/agent/AutoSkill/imaging-101-maintainer/research-runtime/Scripts/python.exe", "-c", "from pathlib import Path;import numpy as np;from src.preprocessing import load_data;from src.visualization import run_experiments,make_figures,export;references={};\nfor key,file in [('corrected','corrected_upstream.npz'),('unmodified','unmodified_upstream.npz')]:\n with np.load('notebooks/reference_outputs/'+file,allow_pickle=False) as f:references[key]=f['reconstruction'][0].copy()\ne=run_experiments(load_data('data/raw_data.npz'),references);export(make_figures(load_data('data/raw_data.npz'),e),e,Path(r'D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research_batch4_20261021\\bio_med\\hdrplus\\gallery'));print(e['metrics'])"]` | 0 | 40.983201 | `gallery.log` |
| `clean_environment` | `["D:/yan1/agent/AutoSkill/imaging-101-maintainer/research-runtime/Scripts/python.exe", "-c", "import platform,numpy,scipy,torch,matplotlib;print(platform.platform(),platform.python_version());print(numpy.__version__,scipy.__version__,torch.__version__,matplotlib.__version__);numpy.show_config()"]` | 0 | 1.617227 | `clean_environment.log` |
| `oracle_freeze` | `["D:/yan1/agent/AutoSkill/imaging-101-maintainer/research_batch4_20261021/root/oracle-env/Scripts/python.exe", "-m", "pip", "freeze"]` | 0 | 0.448180 | `oracle_freeze.log` |
| `source_revision` | `["git", "rev-parse", "HEAD"]` | 0 | 0.018896 | `source_revision.log` |
| `final_notebook` | `["D:/yan1/agent/AutoSkill/imaging-101-maintainer/research-runtime/Scripts/python.exe", "-m", "nbconvert", "--to", "notebook", "--execute", "notebooks/raw_burst_radiance_reconstruction.ipynb", "--ExecutePreprocessor.kernel_name=imaging-101-research", "--ExecutePreprocessor.timeout=600", "--output", "final_hdrplus_executed.ipynb", "--output-dir", "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research_batch4_20261021\\bio_med\\hdrplus"]` | 0 | 37.433031 | `final_notebook.log` |

Final pytest: **23 passed in 7.68s**, no failures. Independent physics includes a same-gain known translation and identical-tile Wiener limit; contract cases include missing file/key, shape/dtype, NaN/Inf/object/overflow, known metric values, reference/evaluator errors, report-write failures, zero/poor outputs and Judge invocation from a foreign cwd. Judge resolves hidden assets relative to itself and returns0/1/2 with UTF-8 finite JSON, without importing src. It evaluates fresh CLI output successfully.

The final canonical active notebook uses explicit `%matplotlib inline` and kernel imaging-101-research, timeout600s, via shared `python -m nbconvert`. It reconstructs all selected comparisons from raw input, then checks separately archived fresh scalar measurements. Its final runtime is37.433031s. All code cells executed and five nonempty PNGs were independently base64-decoded and checked for PNG signatures; exact byte counts/output hashes are in final_notebook.json. This avoids a text/plain-only Agg display. Library modules do not set any plotting backend. Temporary executed notebooks and all export images are private, not task output extras. Jupyter Windows selector-thread/local TCP warnings and harmless original string-comparison SyntaxWarnings are archived with actual exit0 rather than suppressed/claimed absent.

## Hashes and inventory

| Original/private or bound asset | Bytes | SHA256 |
|---|---|---|
| `D:\yan1\agent\AutoSkill\imaging-101-maintainer\research_batch4_20261021\root\hdrplus\upstream\test_data\33TJ_20150606_224837_294\payload_N000.dng` | 26284016 | `83f4bdee37093266ea34f1fbbd80f69f0d2a87d99561b108d0c514c0acd6417c` |
| `D:\yan1\agent\AutoSkill\imaging-101-maintainer\research_batch4_20261021\root\hdrplus\upstream\test_data\33TJ_20150606_224837_294\payload_N001.dng` | 26284016 | `2d2cfda94a0ca3d1003c506b7f436576d4966b29cb14008fd851d1188f3fd4a1` |
| `D:\yan1\agent\AutoSkill\imaging-101-maintainer\research_batch4_20261021\root\hdrplus\upstream\test_data\33TJ_20150606_224837_294\payload_N002.dng` | 26284016 | `4d1a34138fe2462e5d83fca7b300987f40203d2311ab55abba7dd923b371cbac` |
| `D:\yan1\agent\AutoSkill\imaging-101-maintainer\research_batch4_20261021\root\hdrplus\upstream\test_data\33TJ_20150606_224837_294\payload_N003.dng` | 26284016 | `a196d709f75c34371809198d5fdc287b4e557fd9dcb9f02b871a1b5e808fded2` |
| `D:\yan1\agent\AutoSkill\imaging-101-maintainer\research_batch4_20261021\root\hdrplus\upstream\test_data\33TJ_20150606_224837_294\payload_N004.dng` | 26284016 | `3d4411c31377d870d7acf432108fa6a6df518818acaf0bd58df6901dbcc74b3c` |
| `D:\yan1\agent\AutoSkill\imaging-101-maintainer\research_batch4_20261021\root\hdrplus\upstream\test_data\33TJ_20150606_224837_294\payload_N005.dng` | 26284016 | `0870d7879893cd3b478aeab226ba23478dc0a1612dffa182d4d6c1f2a8eabc69` |
| `D:\yan1\agent\AutoSkill\imaging-101-maintainer\research_batch4_20261021\root\hdrplus\upstream\test_data\33TJ_20150606_224837_294\payload_N006.dng` | 26284016 | `2d6c4b2841c01329518edaa70282380114580cf79c18dc23aeb81af326bbdc9c` |
| `D:\yan1\agent\AutoSkill\imaging-101-maintainer\research_batch4_20261021\root\hdrplus\upstream\test_data\33TJ_20150606_224837_294\payload_N007.dng` | 26284016 | `5aa6690dbfcf798c74684c154530aca0f009fc64641227654a34209d25695880` |
| `D:\yan1\agent\AutoSkill\imaging-101-maintainer\research_batch4_20261021\root\hdrplus\upstream\test_data\33TJ_20150606_224837_294\payload_N008.dng` | 26284016 | `12436f9008dc2006d7f1964af8f214c5496a77f10a0d9fddefe47641f71353f7` |
| `D:\yan1\agent\AutoSkill\imaging-101-maintainer\research_batch4_20261021\root\hdrplus\upstream\test_data\33TJ_20150606_224837_294\payload_N009.dng` | 26284016 | `8f4d22d17dc28ee51d2d2532b286d32ca1a75134db509a21802ebe587ec188b5` |
| `D:\yan1\agent\AutoSkill\imaging-101-maintainer\research_batch4_20261021\root\hdrplus\upstream\test_data\33TJ_20150606_224837_294\reference_frame.txt` | 3 | `13bf7b3039c63bf5a50491fa3cfd8eb4e699d1ba1436315aef9cbe5711530354` |
| `D:\yan1\agent\AutoSkill\imaging-101-maintainer\research_batch4_20261021\root\hdrplus\upstream\package\algorithm\alignment.py` | 21271 | `8be86c39afc1dbac65d1435aeaa645eedc2a0c3454b40f85371fd871e0e37be3` |
| `D:\yan1\agent\AutoSkill\imaging-101-maintainer\research_batch4_20261021\root\hdrplus\upstream\package\algorithm\imageUtils.py` | 15075 | `3714eb94c90f54ef73365aad075467153f28ddef741999d5c911db7829a7edfe` |
| `D:\yan1\agent\AutoSkill\imaging-101-maintainer\research_batch4_20261021\root\hdrplus\upstream\package\algorithm\merging.py` | 22085 | `1165be33d19c083ee22c30716926bfd00b0eddfdb14c93d4871052c60a89fe09` |
| `D:\yan1\agent\AutoSkill\imaging-101-maintainer\research_batch4_20261021\root\hdrplus\upstream\package\algorithm\params.py` | 7924 | `635bc8f551dbf7487d653eb5c54da714670d67c90ff421d97b4929475451ce95` |
| `D:\yan1\agent\AutoSkill\imaging-101-maintainer\research_batch4_20261021\root\hdrplus\upstream\package\algorithm\hdrplus.py` | 5980 | `dce10310db1e398062ff70cfa7b257d5e78e954c316000de27c93a21bac95d6e` |
| `D:\yan1\agent\AutoSkill\imaging-101-maintainer\research_batch4_20261021\bio_med\hdrplus\corrected_upstream\package\algorithm\alignment.py` | 21300 | `480e8d7532dae3462af45d0864a4e5531904dd0fab531391fd52912fdd9dc8ba` |
| `D:\yan1\agent\AutoSkill\imaging-101-maintainer\research_batch4_20261021\bio_med\hdrplus\corrected_upstream\package\algorithm\imageUtils.py` | 15144 | `7fe5f87e67dfdc0ab21a39054009da97457573d0321ff56314143b37aca8af6b` |
| `D:\yan1\agent\AutoSkill\imaging-101-maintainer\research_batch4_20261021\bio_med\hdrplus\corrected_upstream\package\algorithm\merging.py` | 22123 | `6a0aa46b15c9c27204b924a45cb2507a29b4ba9fbacf27e00f5a67910716b152` |
| `D:\yan1\agent\AutoSkill\imaging-101-maintainer\research_batch4_20261021\bio_med\hdrplus\paper.pdf` | 1198622 | `fde43f240c15b40ceee18f26ff4d619e695dc9aa7d94b3314e0be1ed5cedf79e` |
| `D:\yan1\agent\AutoSkill\imaging-101-maintainer\research_batch4_20261021\bio_med\hdrplus\paper_record.html` | 20265 | `7f793f770087e01875119f4a0a17d548e720e9552a75ecc52e11ce3b185cdae1` |
| `D:\yan1\agent\AutoSkill\imaging-101-maintainer\research_batch4_20261021\bio_med\hdrplus\dataset_license.html` | 9925 | `6ff115b49305e6305f99033961b6ec1b3a5e12723cdc48f20cb3a3172aeed211` |
| `D:\yan1\agent\AutoSkill\imaging-101-maintainer\research_batch4_20261021\bio_med\hdrplus\minimal_correction.diff` | 1567 | `96a32b68bb0a054f4917577a40fbdd351993789e8861b4cfae56fe8d2b4fbce9` |
| `D:\yan1\agent\AutoSkill\imaging-101-maintainer\research_batch4_20261021\bio_med\hdrplus\oracle.npz` | 1701805 | `5f18f208d01989efea7299e591758a906595a737e7272b9ec86b7e41a9c11abe` |
| `D:\yan1\agent\AutoSkill\imaging-101-maintainer\research_batch4_20261021\bio_med\hdrplus\corrected\oracle.npz` | 1701538 | `5f4d46ab68cbbb01efcb8f1f40b92c14c626f09748beeea4f6c6f50ddeff2a08` |
| `D:\yan1\agent\AutoSkill\imaging-101-release\tasks\raw_burst_radiance_reconstruction\data\raw_data.npz` | 7840965 | `0f3ab37626941bea9814105351d3c0105df7eaeed4e45c66733af621273a924e` |

Full task file/NPZ/reference/fixture hashes, primary paper URLs/hashes, source correction diff hash, every gallery/numeric source and exact command records are in private manifest.json. No standalone metadata JSON, raw DNG/PDF, complete source checkout, temporary notebook or image export is packaged. LICENSE/NOTICE preserve source/data terms. Imported symbols/call roles were checked by AST, and source equations are executable local code rather than external-solver wrappers. The public allowlist (README, raw_data.npz, minimal requirements/environment) suffices without hidden weights or references; README contains no GitHub URL. The benchmark does not claim legacy evaluation_harness integration.

## Completion gate ledger

| Gate | Status | Evidence |
|---|---|---|
| Paper/exact remote revision/license/data hashes | passed | Sections above, verification.json, final_notebook.json, frozen tests and manifest |
| Original exact observations run before extraction; separately corrected oracle and immutable original | passed | Sections above, verification.json, final_notebook.json, frozen tests and manifest |
| Independent source stage/full fixtures and independent physical checks | passed | Sections above, verification.json, final_notebook.json, frozen tests and manifest |
| Preserved constants/units/boundaries/precision and documented minimal corrections | passed | Sections above, verification.json, final_notebook.json, frozen tests and manifest |
| CLI/strict Judge pass on fresh output | passed | Sections above, verification.json, final_notebook.json, frozen tests and manifest |
| Judge malformed/poor/evaluator/foreign-cwd/report tests | passed | Sections above, verification.json, final_notebook.json, frozen tests and manifest |
| Public observations/metadata/output sufficiency and no answer/GitHub leaks | passed | Sections above, verification.json, final_notebook.json, frozen tests and manifest |
| Minimal shared environment and no forbidden symbol/call roles | passed | Sections above, verification.json, final_notebook.json, frozen tests and manifest |
| Full23-test suite | passed | Sections above, verification.json, final_notebook.json, frozen tests and manifest |
| Active complete notebook and five actual embedded figures | passed | Sections above, verification.json, final_notebook.json, frozen tests and manifest |
| Historical/original/corrected/fresh metrics clearly separated | passed | Sections above, verification.json, final_notebook.json, frozen tests and manifest |
| Explicit first9/all10 contribution, weight and brightness analysis | passed | Sections above, verification.json, final_notebook.json, frozen tests and manifest |
| Plans/signatures/math/output conventions semantically checked | passed | Sections above, verification.json, final_notebook.json, frozen tests and manifest |
| Final template inventory, archive manifest and required source/data notices | passed | Sections above, verification.json, final_notebook.json, frozen tests and manifest |


## Parent acceptance review

Parent independently ran23tests (23 passed in8.91s), foreign-cwd Judge0 and final source/inventory audit errors[]. Read explicit alignment, Fourier Wiener fusion, overlap weights and the minimal original-source correction diff; inspected all5 actual inline scientific figure groups. Corrected-oracle reproduction explicitly preserves unmodified original results separately: independently demonstrated subpixel-neighborhood indexing and uint16 RMS overflow corrections; no unchanged-reproduction claim. Real author10-frame raw Bayer burst, heterogeneous last-frame exposure and gain retained and quantitatively analyzed; output is reference-exposure camera ADU, not absolute radiance or true photographic RGB.
