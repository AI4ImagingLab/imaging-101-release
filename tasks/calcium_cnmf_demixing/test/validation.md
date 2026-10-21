# Provenance and validation evidence

## Task identity and real observation binding

Task: `calcium_cnmf_demixing`, biology, full native two-photon source demixing. Physical unknowns, method, input and output contract were fixed in private `method_selection.md` before the original run. All 3000 frames and all 170 by 170 pixels are retained. This is a real movie, not a synthetic phantom or a tiny demonstration crop. No `generate_data.py` is needed or bundled. No actual neuron/spike truth exists; `evaluation/baseline_reference.npz` is an independently executed method result, never relabeled as truth.

Original source: https://github.com/flatironinstitute/CaImAn at **remotely verified** commit `240e1f2edbaefd8e6e0311004daedf38695f49ca`, release 1.13.2. The API response is archived in `remote_commit.json`; this is verified remote revision evidence, not merely a local tracking ref. Complete archive `upstream.zip`, extracted original tree, all source hashes, download metadata and unmodified source stay outside the task. Original selected driver calls the official `demo_pipeline.py` acquisition and `params_demo_pipeline.json`, with supported `use_cnn=False` fixed before runs and one CPU process. This intentional non-CNN quality branch differs from the demo default; no classifier result is fabricated and no selected inverse stage is dropped.

Provider: https://users.flatironinstitute.org/~neuro/caiman_downloadables/Sue_2x_3000_40_-46.tif. The official demo credits Sue Ann Koay and David Tank, Princeton University. Native TIFF is 347214136 bytes, shape `(3000,170,170)`, float32. SHA256 `cbbcb243f49c03d86d5db2679fae6262fdd5ab1ed97c66f3b9224cddea7de486`. The private conversion adds only a batch axis and stores NPZ, without clipping/resampling. `conversion-verification.json` records exact array equality between native TIFF, private conversion and delivered public NPZ. Delivered `data/raw_data.npz` SHA256 `70033f2d1d7706082c60e9e78a5b5c1af4752d5bad56d71d03c337fd55a89fc8`. Public key is only `movie` `(1,3000,170,170)` float32. Negative values down to -296 are original measurements. Nominal frame rate 30 Hz, pitch 2 micrometres/pixel and decay time 0.4 seconds come from the official configuration; absolute photon calibration and independently calibrated pitch are unavailable. Public download availability is recorded separately from software licensing; no separate data-specific license was supplied by that download. Provider attribution is preserved; no claim that software GPL independently establishes ownership of the recording is made.

Public solver inputs are only README, raw NPZ and shared environment/task requirements. No hidden factors, selected indices, upstream code URL, registered movie, privileged calibration, source template, truth or source count is exposed there. README references use paper DOI and the provider's movie URL, with no GitHub domain. Maintainer material is hidden from end-to-end solvers.

## Papers and equation/stage traceability

The selected method is the published two-photon batch CNMF pipeline, with its release numerical conventions explicitly identified rather than claiming every later quality heuristic is a paper equation. Published version-of-record DOI metadata is privately archived for eLife 2019 (`10.7554/eLife.38173`), CNMF Neuron 2016 (`10.1016/j.neuron.2015.11.037`), NoRMCorre 2017 (`10.1016/j.jneumeth.2017.07.031`) and OASIS 2017 (`10.1371/journal.pcbi.1005423`). The actual downloaded and read eLife v1 PDF is SHA256 `6f577dc59331b19f07d8205b24c6f55bb3f6cae19f3e10ca662eff96225985c1`, official `elife-38173-v1.pdf`; OASIS published PDF is `aac7bb0c813612cc122429f1c4dd3befe804e4dbb6514ff53932f5a31ecd4987`. Extracted paper text and originals remain private. An initial eLife article-PDF URL returned non-PDF HTML; the Crossref-provided official CDN version was then fetched and verified. This is recorded in private capture scripts, not represented as a successful first URL.

| Paper equation/stage | Exact upstream module/function | Local executable mathematics | Independent validation |
| --- | --- | --- | --- |
| eLife Eq.1, two-photon `Y=AC+bf+E` | cnmf/cnmf.py fit; spatial.py/temporal.py | physics_model.fluorescence_forward/adjoint; solvers.fit | adjoint identity, scale-gauge conservation, fresh full pipeline |
| NoRMCorre subpixel registration and eLife motion-correction stage | motion_correction.py register_translation, _upsampled_dft, tile_and_correct; base/movies.py template methods | preprocessing localized DFT, constrained maxima, explicit template and warp construction | 3 original full frames, exact subpixel shifts, constant/direction boundary checks, full 3000-frame fields |
| eLife initialization strategies GREEDYROI and batch Algorithm 1 patch processing | initialization.py initialize_components, greedyROI, finetune, hals | solvers.initialize_components, greedyROI, finetune, hals | frozen full 31x31x3000 initialization; selected nb1 patch and rank-one refinement |
| eLife Eq.3 merging rank-one fit | merging.py merge_components, merge_iteration | solvers.merge; ten rank-one updates, AR deconvolution | full original initial/refit arrays; exact observed 37+1 merge groups and 149 sources |
| eLife Eq.4 low-rank background principle; release two-photon compression branch | map_reduce.py run_CNMF_patches | explicit NNDSVDar, coordinate descent, overlap embedding, 100 ALS cycles | independent full global NMF input/W/H fixture and patch NMF saved RNG state |
| CNMF spatial nonnegative sparse inverse and batch Algorithm 2 refit | spatial.py update_spatial_components, regression_ipyparallel, threshold_components_parallel | solvers.positive_lars, update_spatial, energy/morphology threshold, background HALS | 10 real positive LARS designs, full spatial update fixture, orthogonal known case |
| OASIS Algorithm 2 and Eq.15 noise-constrained AR1 inverse | oasis.pyx constrained_oasisAR1; deconvolution.py constrained_foopsi | solvers._Pool, constrained_oasisAR1, deconvolve_trace | 16 original trace/AR/noise cases, p0/p1 full temporal updates, recurrence/nonnegative-event/feasible-noise physics checks |
| eLife Eq.5 projected residual | estimates.py compute_residuals; temporal.py update_temporal_components | solvers.initialize_patches and update_temporal; explicit normalized Gram residual | full temporal fixtures and selected stitched source convention |
| eLife Eqs.6-9 SNR family, with release baseline/noise convention disclosed | components_evaluation.py compute_event_exceptionality/evaluate_components/select_components_from_metrics | solvers event log-CDF windows, released mode heuristic, explicit source selection | original SNR exact on original factors, selected indices exact, captured mode triplets and corrected symmetric comparison |
| eLife spatial quality section, release discrepancy described below | components_evaluation.py classify_components_ep/find_activity_intervals | explicit plateau peaks, neighbor activity exclusion, raw active-frame Pearson | original spatial quality errors below 1.9e-7; identical 102 selected sources |
| Relative fluorescence normalization | cnmf/utilities.py detrend_df_f; utils/stats.py df_percentile/kde/fixed_point | solvers.dff, explicit histogram/DCT/fixed-point KDE and percentile filtering | original accepted factors yield DF/F max error 3.6e-15; full fresh output scored |

## Independent original execution and environments

Oracle environment is separate from cleaned runtime: WSL Linux x86_64, Python 3.12.3 GCC 13.3.0, CaImAn 1.13.2 from the verified editable original source, NumPy 2.5.4, SciPy 1.18.1, OpenCV 5.0.0.93, torch 2.14.1+cpu and original upstream dependencies. Full distributions and numerical environment are archived in `oracle-package-environment.json`. CPU process count and OpenBLAS/OMP/MKL threads are 1; random seed 0. No GPU is used or claimed. The cleaned environment has no CaImAn/scikit-learn/compiled-OASIS/peakutils reconstruction dependency.

Original first command: `/home/admin/imaging101-runtime/oracle-caiman/bin/python /mnt/d/yan1/agent/AutoSkill/imaging-101-maintainer/research10/task07/run_oracle.py`, launched by the archived persistent Windows Python driver. Full native movie motion, save_memmap, CNMF.fit, refit, quality and DF/F all completed with exit 0. Original run time **130.36504530906677 seconds**, peak RSS **1639944 KiB**, 149 fitted and 102 accepted components. `oracle_first/` preserves status, logs, initial/refit/final HDF5 and independent numeric arrays. Original corrected memmaps remain in private WSL `/home/admin/caiman_data/temp/`; no shipped code reads those paths.

A second independent original execution uses `profile_oracle.py`, recording actual called functions via profiling while running the same complete selected pipeline. Exit 0, **257.4271206855774 seconds**, RSS **1640124 KiB**, again 149/102. `oracle_run/`, profile logs and `upstream_called_functions.json` are private. All stored original motion, initial, refit and final arrays are bitwise equal across the two runs, verified by `test_independent_original_repeat_is_bitwise_reproducible`. The additional profiler costs time; this does not imply an algorithm change.

Original warnings include negative sample-template mean/first-percentile adjustment, NMF maximum-iteration convergence notices, original deprecation notices and source-quality warnings for sources active only together with neighbors. Logs preserve the exact text. They were investigated: template adjustment and finite NMF iteration count are selected release conventions; quality warnings remain visible in fresh notebook. They are not suppressed or described as a zero-warning original run.

## Cleaning chronology and discrepancy resolution

Independent stage cases were captured from original code before extracting the corresponding local mathematics. All capture scripts import original code only inside the oracle environment; shipped tests import frozen arrays, never those scripts or packages. Standalone early checks established subpixel motion, greedy/HALS, NNDSVDar, positive LARS, spatial and temporal parity before a full local run.

Full native local motion takes about 52 seconds in a dedicated single-thread run: all 3000 rigid and all 3000x9 piecewise displacement values exactly match original; rigid/elastic templates differ at most 6.1035e-5 counts. Three complete cubic-warp frames differ at most about 9.2e-5 counts across OS backends. These are interpolator precision differences, not a relaxed registration criterion.

Two development mismatches were diagnosed rather than hidden by threshold changes. The pixel-noise helper defaults to log-mean while selected preprocessing uses arithmetic mean; callers now explicitly select mean, giving full pixel-noise relative difference 5.2e-8. Global NMF integer `random_state=1` reinitializes separate SVD and zero-fill RNGs; a single continuously advanced local stream initially changed background factors and quality. Independent global NMF fixture isolated that cause. After preserving both seed resets, independent W/H errors are approximately 5e-13/1.2e-12 and full global A/C/b/f relative differences are 3.5e-6/3.4e-6/3.4e-6/1.4e-6; innovation relative error is 7.6e-6. The earlier incorrect 103-source development result is private diagnostic history, not the released reference or a reported final result.

Positive LARS has a finite-path convention: interpolation uses maximum remaining inactive correlation, including when it turns negative. Replacing this with a generic exact QP changes coefficients substantially, so the explicit path matches the selected released method. Ten real design cases agree to about 1.3e-12. No claim that every finite positive path obeys exact small-alpha Lasso KKT conditions is made.

The original three-point mode heuristic has a verified asymmetry: a duplicated left-narrower condition fails to take a right-pair average. Of 298 original active triplets, 135 take the affected configuration. The released even-length recursion also omits its final admissible half-interval start: the independent case `[0,10,20,20.1]` yields released mode 5 versus symmetric mode 20.05. The local default is explicitly called `release_compatibility_mode`, with independent fixtures; a separate symmetric corrected definition is shown in notebook. On fresh complete factors, release and corrected versions both accept **102**, changed selection indices `[]`; maximum absolute SNR change **0.5131946799547389**, median **0.0008588507827882452**. These sensitivity values include both the three-point and even-boundary corrections and must not be attributed solely to the former. This identifies and bounds this data's quality consequence, not a universal claim about all movies. The inverse factors are identical across this diagnostic.

Paper/code discrepancy: the paper's spatial quality description subtracts background/neighbor contributions before footprint correlation. This selected release's executed `classify_components_ep` excludes overlapping neighbor activity intervals but correlates the footprint against raw active-frame mean pixels; it does not perform that explicit subtraction. Local code reproduces the actual verified release path and labels the difference. Likewise, released event noise uses below-mode RMS and an explicit factor 3 in z scores rather than presenting the paper's half-normal scale discussion as exact code. These quality conventions are documented, not silently represented as unchanged mathematical formulas.

The upstream fixed-AR merge branch appears to pass an index where an AR coefficient would be expected. Full actual profiling shows it is not used: refit p0 merging finds no groups. Local code explicitly rejects that unsupported branch if reached; selected patch merging starts without fixed AR parameters and is completely validated. Other upstream modalities and unselected OASIS L0/decimation/AR-optimization branches are not advertised as implemented.

## Fresh cleaned runs and numerical acceptance

Cleaned Windows x86_64 runtime: Python 3.11.11, NumPy 2.4.6, SciPy 1.17.1, OpenCV 5.0.0, Matplotlib 3.11.1. Exact installed subset, OpenBLAS configuration and architecture are in `cleaned-runtime-environment.json`. Shared environment `environment-research.yml` pins the task's subset; no per-task package upgrade is performed. OpenCV is a justified general primitive exception: DFT/inverse DFT, area/linear/cubic interpolation, affine warp, remap, Gaussian filtering, normalized template correlation and extrema. All registration, acquisition, inverse objective, pool updates, regression paths, background ALS, quality and KDE mathematics remain explicit local code. `symbol-call-role-audit.json` records every imported symbol and call role; library names alone are not treated as proof of compliance.

`python -m src.main` completed a fresh complete native reconstruction and wrote all declared arrays, **180.108526 seconds** on the first formal CLI run with the host's unconstrained BLAS thread defaults. `final-cli.log` records stages and warnings. Its result passes independent `evaluation/judge.py`. The notebook then actively ran the entire selected pipeline under single-thread numerical settings: measured reconstruction **159.5671056000283 seconds**, before-summary notebook **175.30823120009154 seconds**. The variation reflects concurrent host load/backend threading; no seconds-only requirement or cached preview is imposed.

NCC/F1/spatial-assignment/motion criteria were fixed before full cleaned assessment. Dynamic-range NRMSE boundaries were added after development diagnostics and before final CLI/notebook judging; this is not presented as preregistration. They allow alternate numerically valid source solutions while F1 and source-specific NCC prevent trivial omissions. Original independent repeat is exactly equal; nonzero tolerances recognize cross-platform finite precision and solver variation. Tighter frozen-stage parity checks remain mandatory and independently establish extraction fidelity. 0.11-pixel maximum motion difference allows one 0.1-pixel original registration quantum; actual observed full-run error is zero. Background product comparison preserves factor-basis ambiguity instead of rewarding arbitrary factor labels. No metric boundary was widened to accept a numerical failure.

The original innovation array itself reaches approximately -5.0275e-5 native counts due to float32 AR subtraction. The public nonnegativity contract explicitly permits 1e-4 count absolute recurrence roundoff; arrays are not clipped to conceal that. This is an input/physical-consistency tolerance, not a changed reconstruction accuracy threshold. Spatial unit-gauge tolerance is 1e-4, with float32 epsilon normalization documented. Zero valid temporal estimates score NCC=0 and fail; undefined hidden reference configurations are evaluator errors. No NaN/Infinity is emitted.

Fresh notebook metrics, full precision:

```json
{
  "detection_f1": 1.0,
  "spatial_ncc": 0.9999999999963264,
  "spatial_nrmse": 2.516004425711197e-08,
  "fluorescence_ncc": 0.9999999999961919,
  "fluorescence_nrmse": 5.747386320935075e-08,
  "innovations_ncc": 0.9999999999799316,
  "innovations_nrmse": 5.8726806790966425e-08,
  "dff_ncc": 0.9999999999962285,
  "dff_nrmse": 7.041005775811564e-08,
  "background_ncc": 0.9999999999996654,
  "background_nrmse": 1.8534525623020457e-07,
  "motion_max_error": 0.0
}
```

All eight declared output quantities are scored. Required output count K is variable 1..196; no hidden requirement to predict exactly 102 is imposed. Hungarian assignment, cosine>=0.7 matching, unmatched F1 penalty, no temporal/spatial realignment or intensity rescale, background full-product scoring and every boundary are in README and metrics.json. Reference SHA256 `4fb699ab107b72a613d8d0e15260afe52ebaa3cd1fe9fc771de634d572f7f2d5`. Fresh output SHA256 `7ca2ae89350f0bb80ded05227889f3823428386f5550cd8dfb3e48201c5544f6`. Judge resolves hidden assets relative to itself, validates shapes/dtypes/finites, is independent of solver imports, writes UTF-8 strict JSON and uses pass/fail/error exit codes 0/1/2.

## Tests, notebook execution and final review

Commands are run in the task workspace using the shared research interpreter. BLAS/OMP/MKL threads are set to 1 for parity. The default host pytest temporary base `C:/Users/admin/AppData/Local/Temp/pytest-of-admin` is inaccessible; its first attempt produced setup permission errors rather than scientific failures. No global ACL is changed. Use a private maintainer writable `--basetemp` directory, as archived in the final command. Initial test-authoring key typos were corrected; all scientific checks then passed. This environment exception is not hidden.

```text
python -m src.main
python evaluation/judge.py --prediction output/reconstruction.npz --report output/judge_report.json
python -m pytest test/ -v --basetemp D:/yan1/agent/AutoSkill/imaging-101-maintainer/research10/task07/pytest-release-final
python -m jupyter nbconvert --to notebook --execute notebooks/calcium_cnmf_demixing.ipynb --ExecutePreprocessor.timeout=1800 --output calcium_cnmf_demixing_executed.ipynb --output-dir output
```

Earlier complete suite: **66 passed in 186.35 seconds**, including a second active native CLI reconstruction and Judge execution from a foreign cwd. Final complete suite: **79 passed in 212.89 seconds**, including another fresh complete native reconstruction, zero-estimate Judge policy, exact metadata/grid/patch conventions, scope enforcement, original independent-repeat arrays, identified notebook global-stage arrays and malformed metric configuration. Final targeted Judge contracts: **24 passed in 27.62 seconds**; reference-storage comparisons: **3 passed in 5.42 seconds**. After adding the independent even-mode boundary case, that test passed separately without altering reconstructed factors or selected runtime. Judge tests include missing files/assets/keys, extra keys, wrong shape/count/gauge, integer and complex dtypes, NaN/Infinity, negative innovations, poor finite output, zero temporal output, report-write failure and permutation/background-basis invariance. Each metric boundary is validated before scoring: finite numeric nonboolean, NCC/F1 in [0,1], NRMSE/motion nonnegative; malformed settings return error2 with empty metrics and serializable JSON. Metric presentation and Judge agree on a known three-element numerical case. Physics tests independently check the acquisition adjoint, scale-gauge conservation, white-noise RMS units, constant/direction boundary warps, AR recurrence and a feasible noise budget.

Notebook nbconvert completed with **6/6 code cells executed, 6 PNG figure outputs, no error output**. Timeout is 1800 seconds, justified by measured complete runtime and comparisons. All six scientific H2 sections are present: Problem, Data, Physics, Method, Results, Conclusion. The required notebook contains the executed cells and figures; the temporary executed notebook and original logs are archived outside the task. The six extracted figure outputs and contact sheet were actually viewed: native frames, rigid/local motion, registered mean/accepted/rejected footprints, original/fresh/difference map, full-time fluorescence/innovations/relative activity and release/corrected quality comparisons are legible, scientifically labeled and free from missing zero-filled method panels. No paper result or archived source replaces a computed fresh figure.

Plans were reviewed against source signatures and actual caller flow, and semantically checked for LARS normalization/signs, pixel index order, motion inverse sign, two-seed NMF behavior, noise convention, p0/p1 sequence, intensity/scale transfer, OASIS baseline handling and DF/F automatic quantile override. The generated signature appendix is only a supporting check. Unsupported source branches are explicit. No runtime plotting backend is forced on library import.

## Frozen upstream fixtures

All outputs below come from independent original oracle calls on exact bundled observations or their identified stage inputs. `full_quality` stores original residuals plus original quality and DF/F arrays; its filename is not a claim of a new local oracle. Saved RNG state and parameter keys are part of fixture provenance. These are hidden maintainer assets, never allowed solver initialization. Full upstream numerical capture files and original hashes remain private; packaged fixtures retain their original numeric values. The unused full stitched fixture was removed from the final package and retained privately. A later exhaustive safe-loading audit identified original numeric object containers in `all_components` (g, baseline, initial calcium and source noise); private trusted original capture conversion stacks g into `(149,1)` float64 and casts the other numeric containers to float64 without changing values. Six unused None coordinate/grid placeholders were omitted from three OpenCV nonrigid fixtures. The two storage-normalization ledgers record exact original/packaged hashes and transforms. Final audit loads every key of all **55 NPZ files** with `allow_pickle=False`; no object arrays remain. The initial expanded 72-case suite had one packaging-only safe-loading failure before this correction; the final 79-case suite passes and never enables pickle in shipped code or tests.

| Packaged fixture | SHA256 | Consumer and independent capture |
| --- | --- | --- |
| background_nmf.npz | b969322207970cea1083f258b3c36c2bfdab857a1829fd5247ef7996b4909bf5 | test_patch_background_nmf_state; capture_initialization.py |
| full_patch_initialization.npz | 8ed943553b904ab94512a9a0ef28882e93ddd96f1ab6155f53cba7aa5cb6a263 | test_full_greedy_hals_initializer, test_selected_one_background_patch; capture_initialization.py |
| full_quality.npz | bb38c29ad6636c5431799d7a5cb3df0efb6e5081d0fb0b9e769d7209ffc6bd40 | test_quality_and_relative_fluorescence; original refit residual and original quality/DF-F arrays captured through check_quality.py |
| full_spatial_update.npz | 80475fbe1d909221b83ce3e8cf23c5663e081b7977eb13d38c9a38eb6b14e232 | test_complete_spatial_update and test_complete_temporal_update; capture_spatial.py |
| full_temporal_p0.npz | 6c30380193dd8da5aa923382d3fabba68783f90c6312378dbd3e17b341df050e | test_complete_temporal_update; capture_temporal.py |
| full_temporal_p1.npz | c2dae828520b3404afd09ca6f631c965de395903fdf9605bada182ab3bc5d03f | test_complete_temporal_update; capture_temporal.py |
| fullpatch_global_nmf.npz | f282b5a09fd8d92845e5d5ab80f45f2b4080870996cc73ee4a26f6920ae2b2e1 | test_global_background_nmf; capture_fullpatch.py |
| greedy_refinement.npz | e535c53ebe68b8b06fa93fc9a82e6a52b9553e72128e073a70464517f982632b | test_greedy_rank_one_refinement; capture_cases.py |
| half_sample_modes.npz | 802dca24fc3326a3ba54f6e24e575e884859e2493aea9c5f04adf932fbfe1b87 | test_release_mode_known_discrepancy_is_explicit; capture_modes.py |
| nonrigid_0.npz | 9275c24df590fb40c868e203305f8bac6a928a680308a10884dbe7cf8ce41915 | test_piecewise_registration_and_cubic_warp; capture_cases.py |
| nonrigid_1000.npz | 8fd6b26c5e2e8433a9a45b8d80a72f4736683010886d2b798bbd335d7b7032f2 | test_piecewise_registration_and_cubic_warp; capture_cases.py |
| nonrigid_2999.npz | f9e544d37a429d78d89a5b339ee0cffc0432cfb4e46776d65f4ff5dcdb8bced3 | test_piecewise_registration_and_cubic_warp; capture_cases.py |
| oasis_0_0.npz | 352bb7895a2808704630580196f93066e0c24d9dae8b2f4bf05b1048bbad86af | test_oasis; capture_cases.py |
| oasis_0_1.npz | 1327e36f3e9a5e00db6eb532054f905d34b1fd133879c6c3e963a4d4a8c2b4ad | test_oasis; capture_cases.py |
| oasis_100_0.npz | b7bc965317d50eb424739f64627ac34aff480cb83af7c3b79257cab7e935854e | test_oasis; capture_cases.py |
| oasis_100_1.npz | 8e82edae6478d258dd1e24744ee664ebf61af9bf88b723fc95e78677df048291 | test_oasis; capture_cases.py |
| oasis_10_0.npz | 6767169d1728a34f3b154f161abf0b2f07f793d54fbc587172190831146f98b1 | test_oasis; capture_cases.py |
| oasis_10_1.npz | 7c798207c56622aa3b9e420b5b49906172ec8d3a27bd35e133fba0becf780eb7 | test_oasis; capture_cases.py |
| oasis_40_0.npz | 92ba007bb8e4f3285eef422492d5d777346719d3788aa453c14f91a63d81cabb | test_oasis; capture_cases.py |
| oasis_40_1.npz | 4c3bc81094feda7cd3129ab8ed82b3e4219454b95d2e2198c052719113c5aea3 | test_oasis; capture_cases.py |
| oasis_actual_0_0.npz | e9923edfe96a8ad72eb8b29f0a8ace0f688da49e45fdf571333b6e12efb07f9c | test_oasis; capture_oasis_actual.py |
| oasis_actual_0_1.npz | efaf564d949d32e1679d17030ced997e12a7075576a5ff6e18c2680632942c14 | test_oasis; capture_oasis_actual.py |
| oasis_actual_100_0.npz | d3cc4ac5ae59dd0ab94fd7ec2c7ecace825dd0468a3abbfefc58e241d5d0e257 | test_oasis; capture_oasis_actual.py |
| oasis_actual_100_1.npz | bb1364b6120988a574ce1fdd00646dfce6f9d46902c012a08c873878a08656a2 | test_oasis; capture_oasis_actual.py |
| oasis_actual_10_0.npz | 03c6de9d5ce33b44195b7c0fae5a470f21f3a05714ce9ed314db41db9af4b100 | test_oasis; capture_oasis_actual.py |
| oasis_actual_10_1.npz | 05b313bbfdcdb77fc7eef507c07a93c74defd9fd071e6a670390127fed469339 | test_oasis; capture_oasis_actual.py |
| oasis_actual_40_0.npz | d0e99ae11767fc000d2da7986325ce9bcf566f4fac222fa73d5d5433c5803d59 | test_oasis; capture_oasis_actual.py |
| oasis_actual_40_1.npz | 16ce9229eb754691a2c847dbf2c50a51b0297c689de31784fd385dbc85af7165 | test_oasis; capture_oasis_actual.py |
| pixel_noise.npz | b9757247c97eb5574dbfbc728178fbfb1dd8d79ee157ca2d6083da3b81702db4 | test_spectral_noise_and_psd; capture_cases.py |
| positive_lars_0.npz | b9d29cccbd77f47e66a9b1261c4a6d3e86939a7af2a14bb9e4d6bf06d33041d7 | test_positive_lars; capture_lars.py |
| positive_lars_105.npz | 3defdfa9171e2879465210f9f012d82ffc36b59c6d4a8bad11bdd4af0c92540a | test_positive_lars; capture_lars.py |
| positive_lars_120.npz | edc6179987c5df4a2a93a49cff0a4bea6d3099271647ccdc48dfbbcef26231a3 | test_positive_lars; capture_lars.py |
| positive_lars_135.npz | c96881b945955ca159197db4ffbe6748d93620b439f4ce97cb07ecf32c35a4dc | test_positive_lars; capture_lars.py |
| positive_lars_15.npz | 665089d8e165ce52946c8f886e7d63f1a1303aa15f74eeb53b2194642a766811 | test_positive_lars; capture_lars.py |
| positive_lars_30.npz | 26123c3a26f018e3c88faebb30b2be95073baa93291a8f4ecc06dd8d53fbe985 | test_positive_lars; capture_lars.py |
| positive_lars_45.npz | a2dbcae3b43080aadf4b8a63495e505ec8e5727e3b23f0b902d30568261a263a | test_positive_lars; capture_lars.py |
| positive_lars_60.npz | ee9ebbd5f277b444dcd6b6657bade84dd406d53af96c812155eb122a794c3b82 | test_positive_lars; capture_lars.py |
| positive_lars_75.npz | 5d44f1f5fb35a3c68a8190351e083bb6befa573dc410e62fcd60f9c1f632a0ee | test_positive_lars; capture_lars.py |
| positive_lars_90.npz | 812479f85ca62fa254a9020cabf9bbc62a1070784c88a98fda6d5bf79033e66a | test_positive_lars; capture_lars.py |
| registration_0.npz | 48f69bf1d55641eff6b0e6434478dfa59f334ebbf54a2f379ba92b2c5d5eeb0c | test_registration; capture_cases.py |
| registration_1000.npz | 7f38dd2d3dfdd2022b51beec8a9a6786a1f79abcae9b3502a6de801c51658317 | test_registration; capture_cases.py |
| registration_2999.npz | 748eb9318bc4c63633927f1493bf09cb90c5716a7025b3a1aab1611755d3efe1 | test_registration; capture_cases.py |
| selected_patch0.npz | 1bc8747be95509c43216f9dcc1b8eb777d7f95c98b9916da89d6f0f9bc753e3b | test_selected_one_background_patch; capture_fullpatch.py |

## Identified reference measurements

Reference files are distinct original independent runs and identified fresh notebook measurements, not arbitrary duplicate copies. The first original final arrays supply the baseline after documented batch/axis formatting; no fitted array becomes truth. The fresh review file stores diagnostics/all-source factors and corrected-mode sensitivity, without duplicating the complete registered movie or a second reconstruction submission.

| Measurement file | SHA256 | Consumer/provenance |
| --- | --- | --- |
| fresh_notebook_review.npz | 1fdcb9e9b06f18e6a12d80197183d8ead4834f05bc88626d94e464ee372f34db | Active notebook comparison and test_identified_fresh_notebook_global_stage; fresh local arrays |
| upstream_first_all_components.npz | 03ddb6f7aa16cfd16fdca0ec040119e43ac865073eb6f2be717e28af2a47454f | test_independent_original_repeat_is_bitwise_reproducible; notebook uses first/refit and first/repeat final comparisons; original independent run |
| upstream_first_final_arrays.npz | 2e887d6d77e4a182efe18255f62d51cb59d3c042243db6ea10efe9399cf6d3de | test_independent_original_repeat_is_bitwise_reproducible; notebook uses first/refit and first/repeat final comparisons; original independent run |
| upstream_first_initial_arrays.npz | 76f81922aef443da217d5ebbaadf8a9c6d78bbd87392e0f421115facc122016b | test_independent_original_repeat_is_bitwise_reproducible; notebook uses first/refit and first/repeat final comparisons; original independent run |
| upstream_first_motion_outputs.npz | 426e37cceebb038a9db5bd9e05755e744e22066dfb8722b47da884fec69b322c | test_independent_original_repeat_is_bitwise_reproducible; notebook uses first/refit and first/repeat final comparisons; original independent run |
| upstream_repeat_all_components.npz | 03ddb6f7aa16cfd16fdca0ec040119e43ac865073eb6f2be717e28af2a47454f | test_independent_original_repeat_is_bitwise_reproducible; notebook uses first/refit and first/repeat final comparisons; original independent run |
| upstream_repeat_final_arrays.npz | 2e887d6d77e4a182efe18255f62d51cb59d3c042243db6ea10efe9399cf6d3de | test_independent_original_repeat_is_bitwise_reproducible; notebook uses first/refit and first/repeat final comparisons; original independent run |
| upstream_repeat_initial_arrays.npz | 76f81922aef443da217d5ebbaadf8a9c6d78bbd87392e0f421115facc122016b | test_independent_original_repeat_is_bitwise_reproducible; notebook uses first/refit and first/repeat final comparisons; original independent run |
| upstream_repeat_motion_outputs.npz | 426e37cceebb038a9db5bd9e05755e744e22066dfb8722b47da884fec69b322c | test_independent_original_repeat_is_bitwise_reproducible; notebook uses first/refit and first/repeat final comparisons; original independent run |

## Maintainer archive index

Private archive root is `D:/yan1/agent/AutoSkill/imaging-101-maintainer/research10/task07`. The complete source revision, downloaded originals, DOI metadata, data native file, isolated oracle install/driver/capture scripts, full logs, original HDF5, private memmaps, rejected development diagnostics, paper PDFs/text, numerical captures, legal originals and executed notebook are maintained there rather than in the task. `final-archive-index.json` SHA256 `346810da60605d3701d4e85c62f8d3505cc03c15164f28795d201879306f545c` lists file sizes and hashes. `source_hashes.json` enumerates the exact upstream tree. Do not infer an absent remote freshness claim from these local paths.

| Archive item | Bytes | SHA256 |
| --- | --- | --- |
| remote_commit.json | 78160 | 17f632abd51face4211b48086470b593e95d8c5ffdab438f40aa5362d9dc059e |
| source_hashes.json | 13501 | 68c45a7c6a0367f64b4f7604c67ebb166d3d11411ea2dd2c60f1f767f17b896d |
| selected_params.json | 5753 | 1e0f9edc5b1ec3bb78cef7b9ef800b6d41470135354f98340647270003c54d85 |
| screening.json | 2000 | 355043679e51d78e1f3b801c9bf928728add687fe22db4a6444b218551080745 |
| method_selection.md | 2138 | 822b454c031dac0d8cf5fa5c029390a13dbf20025b86e9313caa6fa18846c268 |
| oracle-package-environment.json | 3633 | 2433ee067aa7aad813b095bf514922c54fd84ba8d2c43cd94e85667fe523475d |
| cleaned-runtime-environment.json | 4763 | 8822eb1453fe90ce51b6c0fb6cacb5f636cc6843a9c56f381459f2cc7aa956df |
| conversion-verification.json | 154 | 1bea4632b1e711e6766b6028809767ab7ba0fef28f20569ac72a83661a66588f |
| paper-download-ledger.json | 526 | 8bdf92b2758f764d6f38a6e151ebddb8764c482a8aa2a04ab2a44581718a2bc2 |
| oracle-run.log | 1162 | e56726427a82ce33150a634a852ed6432a809d4241f70cf172770e63488c201b |
| profile-oracle-run.log | 1086 | e2532ed6ec32a9e3c36544a3025c0352f8937500913d1f74325fc2d61548075b |
| oracle_status.json | 342 | 179426170e7380048bd263a18958e8e88d5930c1637b07712c5df56b02336879 |
| oracle_first/oracle_status.json | 343 | 9a2d36411728d16784473de8ffb73a3992f1cedc2e87ee4c49c44a3cd3c1c249 |
| oracle_first/oracle-run.log | 1162 | e56726427a82ce33150a634a852ed6432a809d4241f70cf172770e63488c201b |
| oracle-exit.json | 65 | ea506e57ef13b7695e871767131b6e5295aa0cf15373670f468fbe59c61c453f |
| profile-oracle-exit.json | 64 | 289f811ffbd8690afbd872dc50af0de3a06802eef4660193ebd2958203d3c3f2 |
| run_oracle.py | 3546 | b8f2fb6ad95778e60ca9d48665e8ce94e712089f2ddbfdf4cff389164b2c7010 |
| oracle_driver.py | 747 | 04f38d04e245de1109721a6d0aa4c1b84bf9a0a742b7f4889ff17093b86628ac |
| profile_oracle.py | 565 | 123db34ce9ac76f7c6b1c36a4fd7405235a32ee08893c949cff88d6a5d17ee75 |
| profile_driver.py | 775 | b2068740c811d7381f8791787d1a95c59723c7b71b52f00fb1c20af6aceb5f21 |
| capture_cases.py | 2456 | ebbd41487c5bb8622b5eff11329548999d925f43845b91e115623b0f75c833a1 |
| capture_initialization.py | 1305 | fa9a77122398e0c0be8148933df8b77e2ecf7fb83713a6028a8f2d209f412f08 |
| capture_lars.py | 1444 | 7142dd7eb6f3fa33851eafc63d15e10d9807bc43259a4dc110c6f4c0be7c0371 |
| capture_spatial.py | 1104 | c632189ff3a6a6deef7209b8a7ff4248b085de435f17f58ff9e4a9bb82e6fd29 |
| capture_temporal.py | 1160 | f9f26aa7b49c82a68f633e98d907a903e96ce0b6e079a65b1260327f84c5c828 |
| capture_oasis_actual.py | 1242 | afcf5069f2a22b09419e6a82469a65af852a7708a48880dd9db5f36231e61a84 |
| capture_fullpatch.py | 1426 | da21d511f68086328912dc338ce322b43d7af3b1595c3fae77d5abd258712dbe |
| capture_modes.py | 1180 | ebdf433446683ea2196cecb9e7e5c29e33ba36a43d88eb3f2701f6416ad3776a |
| archive_legal_environment.py | 878 | e65deaadd7a33815e55514afd0723f8a02c9dde2b48def357f7473a007e854b8 |
| tests-final2.log | 21804 | 86de90523efdca08cdc66ed5ad80a2fef7e5dd94a79721e04c02854d86fd8699 |
| notebook-execute.log | 2318 | 432cef50ccd2339ffef2f5d6832f40ef09e6a833435375e7701d97003b940ce0 |
| notebook-review-evidence.json | 89 | cb2b8ed353ea40e626f0521ce1b6b9d17173a550bcab57d3bc9d5656f425c549 |
| notebook-final-review.json | 763 | fe8bed950a134ad5b02db0de8866aba7be545f6814a4ac12bf0928fa3e5d4841 |
| executed-final-notebook.ipynb | 1750956 | f9efcf7751af259f9672272fad6b08aa3c4d2342efc978bd7df726f85d0881ec |
| notebook-contact-sheet.jpg | 344519 | a33eeb32252fc8c34b9007a97ec88cc8e49b3f685fd0253da03af44512914556 |
| notebook-output.txt | 5186 | 5c78d45760fbd085e11d6ba8c77721467397de138f121c9ff3cf1e1b26cc0a37 |
| final-cli.log | 3772 | d403f7babfa7591d46f8e58f626b70c58145d5a748973b97d1e4c023eba234cf |
| mode-policy-report.json | 179 | 2333d278dda1726eda4298e89724fc7137d425970c5919d86604f4f8487335c3 |
| paper_caiman.pdf | 15327971 | 6f577dc59331b19f07d8205b24c6f55bb3f6cae19f3e10ca662eff96225985c1 |
| paper_oasis.pdf | 4515374 | aac7bb0c813612cc122429f1c4dd3befe804e4dbb6514ff53932f5a31ecd4987 |
| upstream.zip | 107017099 | 42193794ee6c93c7a9e5a90c2e3419afb1411d90f73544ab1e3215814f650e68 |
| Sue_2x_3000_40_-46.tif | 347214136 | cbbcb243f49c03d86d5db2679fae6262fdd5ab1ed97c66f3b9224cddea7de486 |
| tests-release-final.log | 15746 | ab3956d301c9d7d0554acbfcb294d26f855804fd7c0748de4da4d079eb1d79ad |
| tests-judge-final.log | 5520 | 79398252743c3af79891e59cacfe35ca662cf6747a45d7fc105850e14d6866e5 |
| tests-reference-final.log | 1612 | d9d47c867d4a625b6e33727bb2587b99209aaa3e53a4dcf51abbfa13a1550cce |
| tests-mode-final.log | 1186 | 22053fadc6d9ada97573c3e4c96a4037c12f238b0729b2a957c8c2e3668bcb95 |
| reference-storage-normalization.json | 1852 | 289473b2298afe67cfcc376e61e12554c2178ff2cf4e15539de00d358adccd08 |
| fixture-storage-normalization.json | 957 | 0e954ee075ae669d055d95525f8dd713813c39c46e877fa3714470572e56f4a7 |
| normalize_reference_storage.py | 1169 | c70f93599b886094a2acf8916409326a4c6e935d0f7c2b7c87a70607280e6f0b |
| remove_none_placeholders.py | 731 | d0890d57663de687abd64ccfd07fe1e0c0c0ebd05062eed329c9de999a3a24c9 |

## Completion gate ledger

| Mandatory gate | Status | Evidence |
| --- | --- | --- |
| Paper/revision/data/equation trace | Passed | Remotely verified commit response, published paper versions, native/NPZ hashes and stage table |
| Original and cleaned complete bound observations | Passed | Two full oracle exits0, formal CLI, full pipeline pytest and active notebook |
| Independent upstream parity and applicable physics | Passed | Frozen original stage fixtures, full-source fresh comparison and independent unit checks |
| Constants/units/coordinates/precision | Passed | Explicit selected noise/bias/patch/axis/gauge/seed conventions and metadata/runtime tests |
| Standard CLI and standalone Judge | Passed | All required keys from fresh full native run, report pass; foreign-cwd Judge pipeline test |
| Judge malformed/poor/evaluator failures | Passed | Contract unit tests, zero finite prediction fail and report-write error handling |
| Public inputs sufficient and no hidden leakage | Passed | Raw key movie only, self-contained geometry/model/output, no public GitHub domain or target calibration |
| Minimal cleaned dependencies and standalone tasks | Passed | Shared pinned NumPy/SciPy/OpenCV/Matplotlib; no original domain import; CPU complete runs |
| Explicit local acquisition and inverse mathematics | Passed | Fourier/operator/NNDSVDar/HALS/LARS/OASIS/KDE local functions and symbol/call-role audit |
| Complete test suite | Passed | Final 79 passed in 212.89s, complete native pipeline included; independent Judge/reference/mode targeted checks |
| Active six-section notebook and visual QA | Passed | nbconvert6/6, errors0, PNG6; viewed private contact sheet and retained required executed notebook |
| Result provenance and metrics | Passed | Original first/repeat/fresh labeled separately; formulas/gauges/threshold chronology explicit |
| README/plans/source semantic consistency | Passed | Derived math, actual execution flow, signature appendix and conventions review |
| Inventory/legal/temporary cleanup | Passed | Task template, all numerical files consumed/indexed, private originals retained, full known legal notices below |

Unselected upstream branches and missing physiological truth are scope limitations, not unexecuted mandatory gates. No GPU validation is claimed. Source licensing and recording attribution are preserved separately; no task-level LICENSE or NOTICE file is created, as requested.

## Preserved source attributions and legal notices

Final inventory: 75 task files including this validation document, 43 numeric fixture NPZs, nine identified reference-output NPZs and the required two output files. `final-task-inventory.json` is private, SHA256 `45579fb3ce5df70dba75f69c6bab98acbc7fee1584a60b12a956d387dc9bf6d5`, and indexes all task files except this self-referential document. No task-level LICENSE/NOTICE, raw TIFF, PDF, upstream checkout, screenshot, temporary executed notebook, `.npy`/`.mat`, Python bytecode or pytest cache remains. Code cells in the final required notebook are exactly equal to those in the successful archived execution; only Markdown gained the explicit even-mode boundary explanation afterward.

The final symbol/call-role audit distinguishes primitive use from core inverse mathematics: OpenCV DFT and inverse DFT supply transforms; `matchTemplate/minMaxLoc` supply generic normalized correlation/extrema only during small sampled-template initialization; `resize/warpAffine/remap/GaussianBlur/filter2D` supply general interpolation/filter kernels. The constrained displacement search, Fourier refinement, field construction, patch CNMF, support/noise conventions, background ALS, LARS path, OASIS active pools and quality/KDE pipeline remain local. NumPy/SciPy linear solves, SVD, FFT, sparse products, morphology, special CDF/quantile functions, root finding and percentile filters serve the documented mathematical substeps. Normal tests and notebooks consume frozen numeric references and local source only; no private capture helper is imported.

Core extraction adapts the CaImAn release under its recorded GPL version 2 terms. OASIS original author is Johannes Friedrich (2016). Low-level registration comes via SIMA and the scikit-image team, with the original complete notice below. General randomized-SVD/NMF/positive-LARS numerical conventions derive from scikit-learn's BSD code; explicit peak plateau detection derives from Lucas Hermann Negri's MIT PeakUtils. These are local readable numerical primitives, not imports of the original reconstruction frameworks. Original licenses, complete copyright headers and complete upstream source are private archival inputs; the following full relevant terms remain with the redistributed cleaned source through this documentation. Provider recording credit remains in README and provenance. No source copyright/permission notice is removed merely because standalone license filenames are omitted.

### CaImAn/SIMA GPL version 2

```text
GNU GENERAL PUBLIC LICENSE
                       Version 2, June 1991

 Copyright (C) 1989, 1991 Free Software Foundation, Inc.,
 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 Everyone is permitted to copy and distribute verbatim copies
 of this license document, but changing it is not allowed.

                            Preamble

  The licenses for most software are designed to take away your
freedom to share and change it.  By contrast, the GNU General Public
License is intended to guarantee your freedom to share and change free
software--to make sure the software is free for all its users.  This
General Public License applies to most of the Free Software
Foundation's software and to any other program whose authors commit to
using it.  (Some other Free Software Foundation software is covered by
the GNU Lesser General Public License instead.)  You can apply it to
your programs, too.

  When we speak of free software, we are referring to freedom, not
price.  Our General Public Licenses are designed to make sure that you
have the freedom to distribute copies of free software (and charge for
this service if you wish), that you receive source code or can get it
if you want it, that you can change the software or use pieces of it
in new free programs; and that you know you can do these things.

  To protect your rights, we need to make restrictions that forbid
anyone to deny you these rights or to ask you to surrender the rights.
These restrictions translate to certain responsibilities for you if you
distribute copies of the software, or if you modify it.

  For example, if you distribute copies of such a program, whether
gratis or for a fee, you must give the recipients all the rights that
you have.  You must make sure that they, too, receive or can get the
source code.  And you must show them these terms so they know their
rights.

  We protect your rights with two steps: (1) copyright the software, and
(2) offer you this license which gives you legal permission to copy,
distribute and/or modify the software.

  Also, for each author's protection and ours, we want to make certain
that everyone understands that there is no warranty for this free
software.  If the software is modified by someone else and passed on, we
want its recipients to know that what they have is not the original, so
that any problems introduced by others will not reflect on the original
authors' reputations.

  Finally, any free program is threatened constantly by software
patents.  We wish to avoid the danger that redistributors of a free
program will individually obtain patent licenses, in effect making the
program proprietary.  To prevent this, we have made it clear that any
patent must be licensed for everyone's free use or not licensed at all.

  The precise terms and conditions for copying, distribution and
modification follow.

                    GNU GENERAL PUBLIC LICENSE
   TERMS AND CONDITIONS FOR COPYING, DISTRIBUTION AND MODIFICATION

  0. This License applies to any program or other work which contains
a notice placed by the copyright holder saying it may be distributed
under the terms of this General Public License.  The "Program", below,
refers to any such program or work, and a "work based on the Program"
means either the Program or any derivative work under copyright law:
that is to say, a work containing the Program or a portion of it,
either verbatim or with modifications and/or translated into another
language.  (Hereinafter, translation is included without limitation in
the term "modification".)  Each licensee is addressed as "you".

Activities other than copying, distribution and modification are not
covered by this License; they are outside its scope.  The act of
running the Program is not restricted, and the output from the Program
is covered only if its contents constitute a work based on the
Program (independent of having been made by running the Program).
Whether that is true depends on what the Program does.

  1. You may copy and distribute verbatim copies of the Program's
source code as you receive it, in any medium, provided that you
conspicuously and appropriately publish on each copy an appropriate
copyright notice and disclaimer of warranty; keep intact all the
notices that refer to this License and to the absence of any warranty;
and give any other recipients of the Program a copy of this License
along with the Program.

You may charge a fee for the physical act of transferring a copy, and
you may at your option offer warranty protection in exchange for a fee.

  2. You may modify your copy or copies of the Program or any portion
of it, thus forming a work based on the Program, and copy and
distribute such modifications or work under the terms of Section 1
above, provided that you also meet all of these conditions:

    a) You must cause the modified files to carry prominent notices
    stating that you changed the files and the date of any change.

    b) You must cause any work that you distribute or publish, that in
    whole or in part contains or is derived from the Program or any
    part thereof, to be licensed as a whole at no charge to all third
    parties under the terms of this License.

    c) If the modified program normally reads commands interactively
    when run, you must cause it, when started running for such
    interactive use in the most ordinary way, to print or display an
    announcement including an appropriate copyright notice and a
    notice that there is no warranty (or else, saying that you provide
    a warranty) and that users may redistribute the program under
    these conditions, and telling the user how to view a copy of this
    License.  (Exception: if the Program itself is interactive but
    does not normally print such an announcement, your work based on
    the Program is not required to print an announcement.)

These requirements apply to the modified work as a whole.  If
identifiable sections of that work are not derived from the Program,
and can be reasonably considered independent and separate works in
themselves, then this License, and its terms, do not apply to those
sections when you distribute them as separate works.  But when you
distribute the same sections as part of a whole which is a work based
on the Program, the distribution of the whole must be on the terms of
this License, whose permissions for other licensees extend to the
entire whole, and thus to each and every part regardless of who wrote it.

Thus, it is not the intent of this section to claim rights or contest
your rights to work written entirely by you; rather, the intent is to
exercise the right to control the distribution of derivative or
collective works based on the Program.

In addition, mere aggregation of another work not based on the Program
with the Program (or with a work based on the Program) on a volume of
a storage or distribution medium does not bring the other work under
the scope of this License.

  3. You may copy and distribute the Program (or a work based on it,
under Section 2) in object code or executable form under the terms of
Sections 1 and 2 above provided that you also do one of the following:

    a) Accompany it with the complete corresponding machine-readable
    source code, which must be distributed under the terms of Sections
    1 and 2 above on a medium customarily used for software interchange; or,

    b) Accompany it with a written offer, valid for at least three
    years, to give any third party, for a charge no more than your
    cost of physically performing source distribution, a complete
    machine-readable copy of the corresponding source code, to be
    distributed under the terms of Sections 1 and 2 above on a medium
    customarily used for software interchange; or,

    c) Accompany it with the information you received as to the offer
    to distribute corresponding source code.  (This alternative is
    allowed only for noncommercial distribution and only if you
    received the program in object code or executable form with such
    an offer, in accord with Subsection b above.)

The source code for a work means the preferred form of the work for
making modifications to it.  For an executable work, complete source
code means all the source code for all modules it contains, plus any
associated interface definition files, plus the scripts used to
control compilation and installation of the executable.  However, as a
special exception, the source code distributed need not include
anything that is normally distributed (in either source or binary
form) with the major components (compiler, kernel, and so on) of the
operating system on which the executable runs, unless that component
itself accompanies the executable.

If distribution of executable or object code is made by offering
access to copy from a designated place, then offering equivalent
access to copy the source code from the same place counts as
distribution of the source code, even though third parties are not
compelled to copy the source along with the object code.

  4. You may not copy, modify, sublicense, or distribute the Program
except as expressly provided under this License.  Any attempt
otherwise to copy, modify, sublicense or distribute the Program is
void, and will automatically terminate your rights under this License.
However, parties who have received copies, or rights, from you under
this License will not have their licenses terminated so long as such
parties remain in full compliance.

  5. You are not required to accept this License, since you have not
signed it.  However, nothing else grants you permission to modify or
distribute the Program or its derivative works.  These actions are
prohibited by law if you do not accept this License.  Therefore, by
modifying or distributing the Program (or any work based on the
Program), you indicate your acceptance of this License to do so, and
all its terms and conditions for copying, distributing or modifying
the Program or works based on it.

  6. Each time you redistribute the Program (or any work based on the
Program), the recipient automatically receives a license from the
original licensor to copy, distribute or modify the Program subject to
these terms and conditions.  You may not impose any further
restrictions on the recipients' exercise of the rights granted herein.
You are not responsible for enforcing compliance by third parties to
this License.

  7. If, as a consequence of a court judgment or allegation of patent
infringement or for any other reason (not limited to patent issues),
conditions are imposed on you (whether by court order, agreement or
otherwise) that contradict the conditions of this License, they do not
excuse you from the conditions of this License.  If you cannot
distribute so as to satisfy simultaneously your obligations under this
License and any other pertinent obligations, then as a consequence you
may not distribute the Program at all.  For example, if a patent
license would not permit royalty-free redistribution of the Program by
all those who receive copies directly or indirectly through you, then
the only way you could satisfy both it and this License would be to
refrain entirely from distribution of the Program.

If any portion of this section is held invalid or unenforceable under
any particular circumstance, the balance of the section is intended to
apply and the section as a whole is intended to apply in other
circumstances.

It is not the purpose of this section to induce you to infringe any
patents or other property right claims or to contest validity of any
such claims; this section has the sole purpose of protecting the
integrity of the free software distribution system, which is
implemented by public license practices.  Many people have made
generous contributions to the wide range of software distributed
through that system in reliance on consistent application of that
system; it is up to the author/donor to decide if he or she is willing
to distribute software through any other system and a licensee cannot
impose that choice.

This section is intended to make thoroughly clear what is believed to
be a consequence of the rest of this License.

  8. If the distribution and/or use of the Program is restricted in
certain countries either by patents or by copyrighted interfaces, the
original copyright holder who places the Program under this License
may add an explicit geographical distribution limitation excluding
those countries, so that distribution is permitted only in or among
countries not thus excluded.  In such case, this License incorporates
the limitation as if written in the body of this License.

  9. The Free Software Foundation may publish revised and/or new versions
of the General Public License from time to time.  Such new versions will
be similar in spirit to the present version, but may differ in detail to
address new problems or concerns.

Each version is given a distinguishing version number.  If the Program
specifies a version number of this License which applies to it and "any
later version", you have the option of following the terms and conditions
either of that version or of any later version published by the Free
Software Foundation.  If the Program does not specify a version number of
this License, you may choose any version ever published by the Free Software
Foundation.

  10. If you wish to incorporate parts of the Program into other free
programs whose distribution conditions are different, write to the author
to ask for permission.  For software which is copyrighted by the Free
Software Foundation, write to the Free Software Foundation; we sometimes
make exceptions for this.  Our decision will be guided by the two goals
of preserving the free status of all derivatives of our free software and
of promoting the sharing and reuse of software generally.

                            NO WARRANTY

  11. BECAUSE THE PROGRAM IS LICENSED FREE OF CHARGE, THERE IS NO WARRANTY
FOR THE PROGRAM, TO THE EXTENT PERMITTED BY APPLICABLE LAW.  EXCEPT WHEN
OTHERWISE STATED IN WRITING THE COPYRIGHT HOLDERS AND/OR OTHER PARTIES
PROVIDE THE PROGRAM "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER EXPRESSED
OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.  THE ENTIRE RISK AS
TO THE QUALITY AND PERFORMANCE OF THE PROGRAM IS WITH YOU.  SHOULD THE
PROGRAM PROVE DEFECTIVE, YOU ASSUME THE COST OF ALL NECESSARY SERVICING,
REPAIR OR CORRECTION.

  12. IN NO EVENT UNLESS REQUIRED BY APPLICABLE LAW OR AGREED TO IN WRITING
WILL ANY COPYRIGHT HOLDER, OR ANY OTHER PARTY WHO MAY MODIFY AND/OR
REDISTRIBUTE THE PROGRAM AS PERMITTED ABOVE, BE LIABLE TO YOU FOR DAMAGES,
INCLUDING ANY GENERAL, SPECIAL, INCIDENTAL OR CONSEQUENTIAL DAMAGES ARISING
OUT OF THE USE OR INABILITY TO USE THE PROGRAM (INCLUDING BUT NOT LIMITED
TO LOSS OF DATA OR DATA BEING RENDERED INACCURATE OR LOSSES SUSTAINED BY
YOU OR THIRD PARTIES OR A FAILURE OF THE PROGRAM TO OPERATE WITH ANY OTHER
PROGRAMS), EVEN IF SUCH HOLDER OR OTHER PARTY HAS BEEN ADVISED OF THE
POSSIBILITY OF SUCH DAMAGES.

                     END OF TERMS AND CONDITIONS

            How to Apply These Terms to Your New Programs

  If you develop a new program, and you want it to be of the greatest
possible use to the public, the best way to achieve this is to make it
free software which everyone can redistribute and change under these terms.

  To do so, attach the following notices to the program.  It is safest
to attach them to the start of each source file to most effectively
convey the exclusion of warranty; and each file should have at least
the "copyright" line and a pointer to where the full notice is found.

    <one line to give the program's name and a brief idea of what it does.>
    Copyright (C) <year>  <name of author>

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License along
    with this program; if not, write to the Free Software Foundation, Inc.,
    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

Also add information on how to contact you by electronic and paper mail.

If the program is interactive, make it output a short notice like this
when it starts in an interactive mode:

    Gnomovision version 69, Copyright (C) year name of author
    Gnomovision comes with ABSOLUTELY NO WARRANTY; for details type `show w'.
    This is free software, and you are welcome to redistribute it
    under certain conditions; type `show c' for details.

The hypothetical commands `show w' and `show c' should show the appropriate
parts of the General Public License.  Of course, the commands you use may
be called something other than `show w' and `show c'; they could even be
mouse-clicks or menu items--whatever suits your program.

You should also get your employer (if you work as a programmer) or your
school, if any, to sign a "copyright disclaimer" for the program, if
necessary.  Here is a sample; alter the names:

  Yoyodyne, Inc., hereby disclaims all copyright interest in the program
  `Gnomovision' (which makes passes at compilers) written by James Hacker.

  <signature of Ty Coon>, 1 April 1989
  Ty Coon, President of Vice

This General Public License does not permit incorporating your program into
proprietary programs.  If your program is a subroutine library, you may
consider it more useful to permit linking proprietary applications with the
library.  If this is what you want to do, use the GNU Lesser General
Public License instead of this License.

```

### scikit-learn numerical primitive BSD notice

```text
BSD 3-Clause License

Copyright (c) 2007-2026 The scikit-learn developers.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

----

This binary distribution of scikit-learn also bundles the following software:

----

Name: GCC runtime library
Files: scikit_learn.libs/libgomp*.so*
Availability: https://gcc.gnu.org/git/?p=gcc.git;a=tree;f=libgomp

GCC RUNTIME LIBRARY EXCEPTION

Version 3.1, 31 March 2009

Copyright (C) 2009 Free Software Foundation, Inc. <http://fsf.org/>

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.

This GCC Runtime Library Exception ("Exception") is an additional
permission under section 7 of the GNU General Public License, version
3 ("GPLv3"). It applies to a given file (the "Runtime Library") that
bears a notice placed by the copyright holder of the file stating that
the file is governed by GPLv3 along with this Exception.

When you use GCC to compile a program, GCC may combine portions of
certain GCC header files and runtime libraries with the compiled
program. The purpose of this Exception is to allow compilation of
non-GPL (including proprietary) programs to use, in this way, the
header files and runtime libraries covered by this Exception.

0. Definitions.

A file is an "Independent Module" if it either requires the Runtime
Library for execution after a Compilation Process, or makes use of an
interface provided by the Runtime Library, but is not otherwise based
on the Runtime Library.

"GCC" means a version of the GNU Compiler Collection, with or without
modifications, governed by version 3 (or a specified later version) of
the GNU General Public License (GPL) with the option of using any
subsequent versions published by the FSF.

"GPL-compatible Software" is software whose conditions of propagation,
modification and use would permit combination with GCC in accord with
the license of GCC.

"Target Code" refers to output from any compiler for a real or virtual
target processor architecture, in executable form or suitable for
input to an assembler, loader, linker and/or execution
phase. Notwithstanding that, Target Code does not include data in any
format that is used as a compiler intermediate representation, or used
for producing a compiler intermediate representation.

The "Compilation Process" transforms code entirely represented in
non-intermediate languages designed for human-written code, and/or in
Java Virtual Machine byte code, into Target Code. Thus, for example,
use of source code generators and preprocessors need not be considered
part of the Compilation Process, since the Compilation Process can be
understood as starting with the output of the generators or
preprocessors.

A Compilation Process is "Eligible" if it is done using GCC, alone or
with other GPL-compatible software, or if it is done without using any
work based on GCC. For example, using non-GPL-compatible Software to
optimize any GCC intermediate representations would not qualify as an
Eligible Compilation Process.

1. Grant of Additional Permission.

You have permission to propagate a work of Target Code formed by
combining the Runtime Library with Independent Modules, even if such
propagation would otherwise violate the terms of GPLv3, provided that
all Target Code was generated by Eligible Compilation Processes. You
may then convey such a combination under terms of your choice,
consistent with the licensing of the Independent Modules.

2. No Weakening of GCC Copyleft.

The availability of this Exception does not imply any general
presumption that third-party software is unaffected by the copyleft
requirements of the license of GCC.

```

### PeakUtils explicit peak detection MIT notice

```text
The MIT License (MIT)

Copyright (c) 2014-2022 Lucas Hermann Negri

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

```

### Original registration attribution and BSD notice

```text
The functions apply_shifts_dft, register_translation, _compute_error, _compute_phasediff, and _upsampled_dft are from
SIMA (https://github.com/losonczylab/sima), licensed under the  GNU GENERAL PUBLIC LICENSE, Version 2, 1991.
These same functions were adapted from sckikit-image, licensed as follows:

Copyright (C) 2011, the scikit-image team
 All rights reserved.

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are
 met:

  1. Redistributions of source code must retain the above copyright
     notice, this list of conditions and the following disclaimer.
  2. Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in
     the documentation and/or other materials provided with the
     distribution.
  3. Neither the name of skimage nor the names of its contributors may be
     used to endorse or promote products derived from this software without
     specific prior written permission.

 THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
 IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT,
 INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
 IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 POSSIBILITY OF SUCH DAMAGE.
```


## Independent Coordinator Acceptance

The coordinator independently reviewed the acquisition model, full NoRMCorre/CNMF/OASIS and quality execution path, unit/gauge/assignment conventions, actual interfaces and both plans. A dimensionally invalid projected-residual display was corrected against patch and temporal source formulas. Judge threshold validation now rejects nonfinite, out-of-domain and boolean configurations before scoring; targeted independent cases and the final 79-test suite pass. OpenCV call roles were inspected: transforms, correlation/extrema, interpolation and filtering are generic primitives; constrained registration and inverse updates remain explicit local code. All six actual executed notebook figures were individually viewed. Only six Markdown headings subsequently gained the uniform numbering 1 through 6; executed code and scientific outputs did not change. The independent structural audit reports no errors or inventory warnings. The final standalone Judge invoked from the private coordinator directory exits0: detection F1=1, temporal/spatial NCC at least0.999999999979, largest reported source NRMSE7.041005775811564e-8, background NRMSE1.8534525623020457e-7 and motion error0. All eight required output quantities are scored through the declared factor gauges and assignment. Local cleaning is accepted; source detection remains a method-reference comparison, not a physiological ground-truth claim. Evidence and the updated final inventory are private coordinator/calcium-external-cwd-judge.json and calcium-final-review.json.
