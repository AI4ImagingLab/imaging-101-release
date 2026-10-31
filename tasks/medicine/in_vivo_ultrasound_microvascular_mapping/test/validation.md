# Provenance and Validation

## Scope and completion status

This is a real-data microvascular density and scalar path-speed reconstruction task, not a simulated single-bubble localization exercise. The bound acquisition contains all pixels and frames from30complete83x128x200IQblocks, i.e.6000frames. The final gate ledger is appended after fresh CLI, Judge, complete tests and notebook verification; until that ledger is populated, cleaning must not be declared complete.

Frozen scientific scope: PALA radial-symmetry localization with full tissue rejection, temporal filtering, dense detection, global assignment, trajectory processing and density/speed rendering. Weighted-average and integer-peak localization are active scientific controls on the same bound acquisition. This selection was recorded in private `screening.md` before local implementation. The paper's other interpolation/Gaussian-fitting localization families and its simulated PSF/flow studies are outside this selected task; no seven-algorithm aggregate ranking or complete published experiment is claimed. No absent method is plotted as zero or replaced with an archived output.

## Sources, versions, and archive index

Private maintainer archive root: `D:/yan1/agent/AutoSkill/imaging-101-maintainer/research10/task09/`. It is not imported by solver, Judge, tests or notebook. All original archives, complete checkouts, PDFs, capture scripts, logs, and diagnostics stay there.

- Upstream repository: https://github.com/AChavignon/PALA . Remote GitHub REST verified `main` commit **f3e7813767d747139e67a0d3f1a7c60b745a698a**, archived response `commit.json`. `git ls-remote` failed on an initial connection; remote freshness is established by the successful REST response, not a local tracking branch. Exact-commit archive `upstream_f3e7813.zip` is extracted under `exact/PALA-f3e7813767d747139e67a0d3f1a7c60b745a698a/`. The exploratory main archive is not used by oracle.
- Paper: Heiles et al., *Performance benchmarking of microbubble-localization algorithms for ultrasound localization microscopy*, Nature Biomedical Engineering6,605-616(2022), DOI10.1038/s41551-021-00824-8, first online17February2022. Publisher preview `paper.html`; full HAL author-manuscript v1 at https://hal.science/hal-03965571v1 , deposited31January2023/released7February2023. Actual downloaded45-page manuscript `paper_hal.pdf`, extracted `paper_hal.txt` and independent root extraction `root-paper-hal-text.txt`. This is identified as the HAL manuscript, not misrepresented as the final publisher-paginated PDF.
- Supplement: HAL annex https://hal.science/hal-03965571/file/supplemental_informations_NBME-20-1220D.pdf , `supplement_hal.pdf`,36pages,27159371bytes; full extraction `supplement_hal.txt`. Method1-6 and1-7 were read, and radial derivation page7 was rendered/inspected as `supplement-page7.png`.
- Addendum: DOI10.1038/s41551-023-01123-0, published online20October2023. Actual publisher PDF `addendum.pdf` and `addendum.txt` read completely. It adds the separate rat-brain dataset Zenodo7883227 with200000frames and raw/beamformed data. It does not modify this MouseTumor dataset or give a replacement MouseTumor reconstruction algorithm. It explicitly says a ULM reconstruction of that new dataset was not provided. Its details are not silently applied to these older files.
- Data: https://doi.org/10.5281/zenodo.4343435 , version1.1, record metadata `zenodo.json`, accompanying `data_readme.md`. Actual source package `PALA_data_InVivoMouseTumor.zip`,2436645325bytes, advertised whole-archive MD5`232b95622bf3d7a76eed0bd840bbe547`. **Whole ZIP was not downloaded and whole-archive MD5 was not verified.** Individual members001-030 were downloaded using range requests, decompressed using the original ZIP method, checked against exact member lengths and central-directory CRC32, and separately hashed SHA256. `root-complete-member-manifest.json` and `root-complete-download.log` document all30successful checks. No simulated member is used.

## Bound public input and data fidelity

`data/raw_data.npz` contains only `iq`, shape(1,30,83,128,200),complex64. All30members' complex64IQ arrays compare exactly elementwise with their public NPZ slices; `raw_npz_roundtrip.json` records shape,dtype,and exact equality per block. The final NPZ SHA256 is listed in the final inventory. There is no spatial crop, frame decimation, artificial noise, alternate acquisition, hidden clean observation, target-derived PSF, random seed/object definition, or hidden reference key in the public file. Acquisition scalars are written directly in README; no metadata JSON is shipped.

| Original member | Uncompressed bytes | ZIP CRC32 verified | SHA256 |
| --- | ---: | --- | --- |
| PALA_InVivoMouseTumor_001.mat | 15993765 | 49ff3aa5 | `122b73deb30a7689d118ea176d22f5350d3808fc6d54db0e09c9063fd499b478` |
| PALA_InVivoMouseTumor_002.mat | 16000786 | 405ca5c1 | `faf9dc98fa0bbb23992b91ba0af4e0ec4445557d165361f289b3af4810041c7b` |
| PALA_InVivoMouseTumor_003.mat | 16002486 | 82ca98d4 | `25ebd0ae3a99c88f5422f12a9f6d08bb99a112e12b1073d4747e701fcee90715` |
| PALA_InVivoMouseTumor_004.mat | 15996543 | deb6f538 | `26937a128e110dde3cd233d7636c1f13bc0059610dea545f62b750815d32b09a` |
| PALA_InVivoMouseTumor_005.mat | 16002067 | 09f3c2a1 | `2d91b29f320f3f89cb7046670cefc4f4eae2f4c9740dbdbb85326152c499b939` |
| PALA_InVivoMouseTumor_006.mat | 16003531 | 6d114658 | `84cfb96960990d88ba34230bd0ca4a83df0c06e8fa8f581142c44b43a2991f71` |
| PALA_InVivoMouseTumor_007.mat | 16002354 | fca85fc0 | `a0ee12363d722979848e05a2f1882659a97e40c7191f26e9a4b3a747eeaea649` |
| PALA_InVivoMouseTumor_008.mat | 15998021 | 81e9272c | `a71940acd4a3319d1e24a964c04ad1a77d1806642345cce22e4fc3b489f95478` |
| PALA_InVivoMouseTumor_009.mat | 16001276 | f4f34b06 | `ca0b4867c64008ef4eba349790671eed2ec8761574f172fd220b7ccb45ac2fe8` |
| PALA_InVivoMouseTumor_010.mat | 15999477 | 8de48d8f | `a3bc0aef2252d02c75fcf5e9ed736b7d1e1ace742094f0b3ea67f977513a682e` |
| PALA_InVivoMouseTumor_011.mat | 16001567 | a35114fa | `3eed502b3a10c891b7726c1c96123ecb75392cf1d029e3128ac9496ae3868689` |
| PALA_InVivoMouseTumor_012.mat | 16000583 | f05dfff0 | `ccf69fdc9b60dbef6b2ad8f6de9066e5613495e170133b76c3e0470cfd3560fa` |
| PALA_InVivoMouseTumor_013.mat | 15999436 | 2004edbb | `8a80011c4ff56785fd74bf79753a80c2a40394837548df3784df89c10aa8683c` |
| PALA_InVivoMouseTumor_014.mat | 16001401 | 9cccba35 | `7476e76149a670eab073f80e367c51964c887b2a95b8a505ea4199b65edd4d61` |
| PALA_InVivoMouseTumor_015.mat | 16003021 | 40e78873 | `5f203f01d2574ec439d922798449be5ed7e8e2992c37851f537018cafcca7a7e` |
| PALA_InVivoMouseTumor_016.mat | 16000154 | 360ed20d | `940c71b0fe30633c60794c94195847450d3b4f42822128bffbf68b2b3f649d75` |
| PALA_InVivoMouseTumor_017.mat | 16001112 | ef71145c | `9759bd9afe96f0ce25d119efb01a1fbfdc4b95a9b122b04c147aa02574df2f8e` |
| PALA_InVivoMouseTumor_018.mat | 16000191 | 81320271 | `e4b93dfdb40cd585ebf36713c45cd4ff66e7e5df6e0b9bf94989ba1c2f725b6a` |
| PALA_InVivoMouseTumor_019.mat | 15999887 | c897f0a9 | `86b7ddd30ab5b3096a44634ba28e5167f2cdef39a358900835d651f105c7bf09` |
| PALA_InVivoMouseTumor_020.mat | 15999039 | 1aa3417c | `107c2892a039b8c43d0a1d01dc5bdec9a4ac8cedfd8b9e8024fe3e784d78ecde` |
| PALA_InVivoMouseTumor_021.mat | 15997971 | 0d24c6f6 | `04a5c491c2fabfb30b1b3e8abbbdde34b8f5cc3939e27f36c9450a6d28e342ab` |
| PALA_InVivoMouseTumor_022.mat | 15999295 | c13e322e | `88e666988d9417878d34a41d3fba1ba381674ce8e22ed33195df82cbf1a37495` |
| PALA_InVivoMouseTumor_023.mat | 15999835 | 6852f2e4 | `02851a445c3c73874099d711aeecd56c104a6feaf8f0a074b93215b964d0d626` |
| PALA_InVivoMouseTumor_024.mat | 15998814 | 6a3e4c83 | `8333379a1e6e3d544f7096da614586353d54bf2c9b1b572bc819234bb7ed292e` |
| PALA_InVivoMouseTumor_025.mat | 15997244 | 15bbf4a4 | `cbb041d8e20472f3bf1e747786255969264fda4ac9c24714a33aedcda02e73e7` |
| PALA_InVivoMouseTumor_026.mat | 15997534 | 6fb495a0 | `08678b97ddfba6cfae3e08745ea9e425e905c38b65b8d2d692a472a8d5f224f6` |
| PALA_InVivoMouseTumor_027.mat | 15999683 | 166d992a | `0e121320b88a5015958da43e24c6529abcd57fcb00dfb1733837addfc6650cea` |
| PALA_InVivoMouseTumor_028.mat | 16000747 | e6220f5a | `4b8a9994f0072423718fedb6d3e0f05c9e34177974d0059669580936669e7254` |
| PALA_InVivoMouseTumor_029.mat | 15999322 | d9268df1 | `b82e25a68ce9071dba3e4e92dd4999e0527b8587c8cec1cfa615fbf0edfab67e` |
| PALA_InVivoMouseTumor_030.mat | 16001236 | 190870ec | `60aeec8df87c1d9923022946d6cfda4b4a71a1a64510653eb552e0aa162c1f6c` |

## Paper/code/data discrepancies and decisions

| Evidence | Discrepancy or limitation | Preserved convention / independent investigation |
| --- | --- | --- |
| Main manuscript OnlineMethods719-727 | General in-vivo text removes4of800modes | Selected MouseTumor code explicitly uses `[10,200]`, removing9of200; retain dataset-specific source rather than substitute the generic brain description |
| Main manuscript714-718; original UF | Paper says150blocks over150seconds; UF says BlockRepeatFrequency2Hz | Preserve UF setting as reported acquisition metadata; do not invent absolute timestamps or assume a calibrated total duration; never link across blocks |
| Main manuscript probe0.1mm; UF Pitch0.11mm; example wavelength1540/(15e6)m | Transducer pitch, paper rounded pitch and wavelength differ; no independent IQgrid calibration | Keep coordinates/speeds in acquisition pixel and pixel/s; no unsupported conversion to mm/s |
| SVDfilter and input dtype | Native-single Gram projector differs across BLAS/LAPACK backends beyond frozen parity gates | Explicit complex128 computation variant on unchanged complex64samples; original source functions remain unmodified in both oracle runs; native-single results retained as a distinct comparison |
| Original edge indexing | `IQ(:,[1 end])` on3Dinput collapses dimensions, not ordinary every-frame lateral edge zeroing | Preserve actual MATLAB indexing through a Fortran view; independent small-array unit case distinguishes it |
| Supplement method1-7 p7 | Printed point-line distance denominator is1+m^2 without square root; its following squared objective and source use1+m^2 | Derive correct perpendicular squared-distance objective and test explicit solution against independent weighted least squares; no source inverse update changed |
| Supplement method1-7 text vs source | Text describes classical weighted-average initial center; source weights midpoint coordinates by gradient energy | Preserve executable gradient-energy centroid; record distinction rather than claim prose and code are identical |
| Radial source NaN fallback | Unsmoothed-slope fallback sign differs from primary gradient slope | Derive/sign-check primary formula independently. Audit every candidate patch in all30blocks: zero NaN fallback and zero infinite slope calls. No bound-data result depends on that suspect branch; preserve it transparently for source fidelity, without claiming arbitrary degenerate input validity |
| Source tracking/pala speed | Speed divides interpolated smoothed path length by n*dt and colon endpoint can precede n | Preserve finite-track estimator and disclose downward bias; analytic10pointstraight-linecase gives44.8pixel/s |
| Source ULM parameter fwhm3 | MATLAB round(1.5)=2 yields5x5ROI | Preserve exact ROI,margin,maxima count and1.5offset rejection |
| Original PALA_multiULM | Tracking/interpolation computed before final float32track cast | Keep double localization,assignment,smoothing; quantize raw/interpolated rendering tracks only at source-matching point |
| Real experiment | No true anatomy/velocity field available;30of150blocks only | Hidden baseline is standard-method output, never truth; show saturation and sampling limitations |

## Numerical precision investigation, before completion

`preimplementation_criteria.json` fixed temporal-stage relative-L2<=1e-4, stable localization position difference<=0.001pixel, density NCC>=0.995/NRMSE<=0.005, speed NCC>=0.99/NRMSE<=0.01 before cleaned results were judged. Native-single first-block local/original SVD discrepancy0.0027401724 and radial density NCC0.9807439 failed those gates. Feeding the captured original Gram directly to a different single LAPACK backend still gave about0.001883 relative SVD-output discrepancy. An original repeat produced exactly equal first-block SVD output, so stochastic variability was not used as an explanation.

Root accepted an explicit double-computation variant after the independently measured ninth/tenth Gram singular-value gap and projector identities ruled out a degenerate cutoff. `precision_diagnostic.json` records first-block gap0.1400585298, retained/removed orthogonality4.30e-15, Hermitian residual1.51e-16, idempotence2.93e-15, gesvd/gesdd projector difference3.58e-15, and complement-projection identity9.50e-14. Original-source double input versus cleaned double computation gives first-block SVD relative4.71e-12, causal-filter relative6.02e-12,34175positions with maximum coordinate discrepancy2.27e-10pixel, and444retained radial tracks.

`allblock_precision_and_fallback_audit.json` independently examines all30blocks. Summary is inserted below; it uses general linear algebra to test physical/mathematical identities rather than generating expected parity outputs from the cleaned inverse.

- Minimum relative ninth/tenth Gram singular-value gap: 0.069399899421217756.
- Maximum double gesvd/gesdd projector relative difference: 4.2504139484203869e-15.
- Total candidate-patch NaN-slope fallback count: 0.
- Total candidate-patch infinite-slope count: 0.

The coefficient-level comparison uses5e-14relative/5e-15absolute tolerance for independent Octave/SciPy Butterworth design: observed numerator coefficient difference was2.13e-14, with transfer stages matching about6e-12relative. An initial unnecessarily strict1e-14coefficient test failed and remains recorded; this roundoff-scale unit tolerance does not alter the frozen1e-4stage or output-quality boundaries. No quality threshold was weakened to accommodate the native-single failure.

## Original runs and independent fixtures

**Original native-single full run.** `run_oracle.m` with unmodified exact-commit numerical functions; driver `run_oracle_driver.py`. Command captured in `oracle-run-status.json`: WSL Ubuntu24.04, Octave8.4.0, image2.14.0, signal1.4.5, control4.0.1; system BLAS/LAPACK3.12.0(`libblas.so.3.12.0`, `liblapack.so.3.12.0`), not an assumed OpenBLAS installation. CPU single thread settings `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`. Exit0, wall2324.752598s; method processing log elapsed2322.677s. All30exactbound blocks and allthreefrozencomparison methods ran. Raw/speed/density stages captured under `oracle/`, final`oracle/final.mat`. Counts: radial1019519positions/15621tracks; WA1068663/15931; integer1068663/18117. Qt runtime-directory/WSL localhost-proxy warnings are environmental and did not change exit status. No solver seed is used.

**Original double-computation full run.** `run_oracle_double.m` and `run_oracle_double_driver.py` use the exact same unmodified MATLAB functions and observations, with only explicit `double(IQ)` at the driver input. Captures are separate under `oracle_double/`, never relabeled native-single outputs. The final section records actual exit/runtime after completion. Per-block artifacts preserve all3localizationtables,raw/interpolatedtracks,counts,maps,timings; firstblock`stages001.mat` preserves complexSVD/filter outputs and coefficients.

**Compatibility adapter.** Only `compat/smooth.m` supplies MATLAB's default moving-average behavior unavailable in Octave. It reduces even spans, shrinks odd endpoint windows, and is checked independently with constant,ramp,quadratic,short-track cases. The source core is not patched. `capture_math_cases.m` executes original Munkres,radiallocalizer and adapter to freeze cases; random assignment input seed121 is recorded in that script and is unrelated to observation generation. No private script or upstream source is imported by final runtime/tests/notebook.

| Frozen task artifact | Source and documented consumer |
| --- | --- |
| test/fixtures/upstream_block001.npz | Original double `oracle_double/block001.mat` and `stages001.mat`; temporal/localization/assignment/rendering parity tests. All expected output keys start output_, coefficients start param_ |
| test/fixtures/upstream_math_cases.npz | Original `oracle/math_cases.mat`; input_cost/output_assignment tie,rectangular,forbidden and partial assignment cases; input_patches/output_centers; smoothing input/output. Consumed by independent math-case parity test |
| notebooks/reference_outputs/upstream.npz | Full original double final maps,counts,timing; plus saturation and track-length summaries computed solely from captured original per-block outputs. Consumed by active notebook comparisons |
| notebooks/reference_outputs/upstream_single_precision.npz | Full identified native-single original maps/counts; consumed only by notebook precision comparison |
| evaluation/baseline_reference.npz | Original double radial density and speed, batch-first float64; consumed only by standalone Judge. No actual truth file is invented |

## Equation/stage traceability

| Paper or selected method stage | Exact-commit upstream function/file | Local function | Independent case / validation |
| --- | --- | --- | --- |
| MainFig2, supplementFig1-1steps1-2; OnlineMethods724-727 | SVDfilter.m; PALA_VivoMulti MouseTumor row | original_boundary_zero,svd_filter,temporal_filter | original full-block complex outputs; Hermitian/idempotence/complement and cutoff-gap checks; Butter coefficients |
| Detection and ROIselection in source | ULM_localization2D.m steps1-3 | regional_maximum,localize | original detections allthree methods; independent connected-plateau case and margin/indexing tests |
| Supplement method1-7, pp7-8 finite differences, weighted perpendicular-distance normal equations | localizeRadialSymmetry.m / lsradialcenterfit | radial_centers | original numerical patches; analytic Gaussian/reflection/intensity-scale cases; independently constructed weighted-lstsq solution |
| Supplement method1-6,p6 weighted first moments | ULM_localization2D / LocWeightedAverage | localize(method='wa') | original full-block weighted-centroid positions,tracks,maps |
| Integer-grid scientific control | ULM_localization2D / NoLocalization | localize(method='integer') | original no-shift positions,exact tie-sensitive assignment cases |
| Main OnlineMethods729-735,Kuhn-Munkres squared-distance matching | hungarianlinker.m,munkres.m,simpletracker.m | link_frames,munkres,track_localizations | frozen original rectangular/tie/forbidden cases; original rawtrack arrays and length order; hardgate/no-gap/strictlength analytic cases |
| Source pala trajectory postprocessing | ULM_tracking2D.m pala branch | smooth_track,interpolate_tracks | original rawtracks-to-density/speed parity; independent moving-average endpoints and44.8pixel/sstraight-line case |
| Main OnlineMethods737-740; supplementFig1-1steps5-6 | ULM_Track2MatOut.m mode2D_velmean; PALA_VivoMulti block accumulation | render_tracks,reconstruct | original block/fullmaps; independent distinct-crossing and weighted-speed case |
| MainFig5,in-vivo saturation/trajectorylength/spatialgrid discussion | Original captured blockmaps/rawtracks | plot_sampling and active notebook | original/fresh method counts,trajectories,saturation,spatialspectra; displayonlysqrttransform labeled |
| Benchmark metrics, not paper synthetic localisation RMSE | Independent benchmark convention | visualization.metrics; independent Judge | known cosine/dynamic-range cases,malformed and poor submission contract tests |

## Environment, interfaces, and intended verification commands

Cleaned CPU runtime uses the shared research environment: Python/NumPy/SciPy/Matplotlib exact versions and numerical configuration are archived in `cleaned_environment.json`. Requirements contains only NumPy2.4.6,SciPy1.17.1,Matplotlib3.11.1. No domain package is installed/imported; pytest/Jupyter are shared development tools. CPU execution is the supported path; no GPU compatibility is claimed.

From the task root:

```text
python -m src.main
python evaluation/judge.py --prediction output/reconstruction.npz --report output/judge_report.json
python -m pytest test/ -v
python -m nbconvert --to notebook --execute notebooks/in_vivo_ultrasound_microvascular_mapping.ipynb --ExecutePreprocessor.kernel_name=imaging-101-research --ExecutePreprocessor.timeout=1800 --output review_executed.ipynb --output-dir <private archive>
```

Private writable TEMP/TMP and the shared research kernel path are supplied by the maintainer driver; public files contain no private environment path. Notebook uses `%matplotlib inline`; library imports never change the Matplotlib backend. Allsix H2sections are Problem,Data,Physics,Method,Results,Conclusion. Both required output quantities and allthreecomparison methods must be freshlycomputed and plotted. No cached-preview mode satisfies the reconstruction gate.

The Judge is standalone and validates required keys,exactshape,float64,finite/nonnegative values. Both arrays require NCC and dynamic-rangeNRMSE; there is no crop,rescale,alignment,mask,phasechange,or magnitude conversion. Zero estimate has NCC0. Malformed inputs/evaluator errors produce empty metrics/code2; valid poor predictions fail/code1; pass/code0. CWD-independent hidden assets,reportparentcreation,reportwritefailure and known metriccases are tested. File/report JSON uses no NaN or Infinity numbers.

## License and attribution evidence

The dataset record metadata says CC BY4.0, whereas the accompanying data README and PALA source say CC BY-NC-SA4.0. Both evidence sources and both complete license texts are retained below. This task is prepared for authorized local research and makes no claim of unrestricted external redistribution. The more restrictive accompanying terms are retained rather than interpreting metadata as permission to discard them.

Original attribution: Arthur Chavignon,Baptiste Heiles,Vincent Hingot; CNRS,Sorbonne Universite,INSERM; directed by Olivier Couture. Code version1.1dated2020-12-17. Cite Heiles,Chavignon,Hingot,Lopez,Teston,Couture, DOI10.1038/s41551-021-00824-8. Source-specific adaptations: explicit local numerical extraction,shape/interface normalization,complex128computation variant,vectorized independent-ROI operations,retained Munkres column-major ordering,independent Judge and active scientific review. No standalone LICENSE/NOTICE file is included, as requested; required notices are consolidated here and code attribution is preserved in module docstrings.

### Original PALA disclaimer

THIS SOFTWARE IS PROVIDED BY THE AUTHORS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS AND CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

### Original simpletracker/Munkres terms

```text
Copyright (c) 2016, Jean-Yves Tinevez
Copyright (c) 2009, Yi Cao
Copyright (c) 2011, Jean-Yves Tinevez
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in
      the documentation and/or other materials provided with the distribution

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.

```

### Full CC BY-NC-SA4.0 legal text

```text
Attribution-NonCommercial-ShareAlike 4.0 International

=======================================================================

Creative Commons Corporation ("Creative Commons") is not a law firm and
does not provide legal services or legal advice. Distribution of
Creative Commons public licenses does not create a lawyer-client or
other relationship. Creative Commons makes its licenses and related
information available on an "as-is" basis. Creative Commons gives no
warranties regarding its licenses, any material licensed under their
terms and conditions, or any related information. Creative Commons
disclaims all liability for damages resulting from their use to the
fullest extent possible.

Using Creative Commons Public Licenses

Creative Commons public licenses provide a standard set of terms and
conditions that creators and other rights holders may use to share
original works of authorship and other material subject to copyright
and certain other rights specified in the public license below. The
following considerations are for informational purposes only, are not
exhaustive, and do not form part of our licenses.

     Considerations for licensors: Our public licenses are
     intended for use by those authorized to give the public
     permission to use material in ways otherwise restricted by
     copyright and certain other rights. Our licenses are
     irrevocable. Licensors should read and understand the terms
     and conditions of the license they choose before applying it.
     Licensors should also secure all rights necessary before
     applying our licenses so that the public can reuse the
     material as expected. Licensors should clearly mark any
     material not subject to the license. This includes other CC-
     licensed material, or material used under an exception or
     limitation to copyright. More considerations for licensors:
    wiki.creativecommons.org/Considerations_for_licensors

     Considerations for the public: By using one of our public
     licenses, a licensor grants the public permission to use the
     licensed material under specified terms and conditions. If
     the licensor's permission is not necessary for any reason--for
     example, because of any applicable exception or limitation to
     copyright--then that use is not regulated by the license. Our
     licenses grant only permissions under copyright and certain
     other rights that a licensor has authority to grant. Use of
     the licensed material may still be restricted for other
     reasons, including because others have copyright or other
     rights in the material. A licensor may make special requests,
     such as asking that all changes be marked or described.
     Although not required by our licenses, you are encouraged to
     respect those requests where reasonable. More considerations
     for the public:
    wiki.creativecommons.org/Considerations_for_licensees

=======================================================================

Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International
Public License

By exercising the Licensed Rights (defined below), You accept and agree
to be bound by the terms and conditions of this Creative Commons
Attribution-NonCommercial-ShareAlike 4.0 International Public License
("Public License"). To the extent this Public License may be
interpreted as a contract, You are granted the Licensed Rights in
consideration of Your acceptance of these terms and conditions, and the
Licensor grants You such rights in consideration of benefits the
Licensor receives from making the Licensed Material available under
these terms and conditions.


Section 1 -- Definitions.

  a. Adapted Material means material subject to Copyright and Similar
     Rights that is derived from or based upon the Licensed Material
     and in which the Licensed Material is translated, altered,
     arranged, transformed, or otherwise modified in a manner requiring
     permission under the Copyright and Similar Rights held by the
     Licensor. For purposes of this Public License, where the Licensed
     Material is a musical work, performance, or sound recording,
     Adapted Material is always produced where the Licensed Material is
     synched in timed relation with a moving image.

  b. Adapter's License means the license You apply to Your Copyright
     and Similar Rights in Your contributions to Adapted Material in
     accordance with the terms and conditions of this Public License.

  c. BY-NC-SA Compatible License means a license listed at
     creativecommons.org/compatiblelicenses, approved by Creative
     Commons as essentially the equivalent of this Public License.

  d. Copyright and Similar Rights means copyright and/or similar rights
     closely related to copyright including, without limitation,
     performance, broadcast, sound recording, and Sui Generis Database
     Rights, without regard to how the rights are labeled or
     categorized. For purposes of this Public License, the rights
     specified in Section 2(b)(1)-(2) are not Copyright and Similar
     Rights.

  e. Effective Technological Measures means those measures that, in the
     absence of proper authority, may not be circumvented under laws
     fulfilling obligations under Article 11 of the WIPO Copyright
     Treaty adopted on December 20, 1996, and/or similar international
     agreements.

  f. Exceptions and Limitations means fair use, fair dealing, and/or
     any other exception or limitation to Copyright and Similar Rights
     that applies to Your use of the Licensed Material.

  g. License Elements means the license attributes listed in the name
     of a Creative Commons Public License. The License Elements of this
     Public License are Attribution, NonCommercial, and ShareAlike.

  h. Licensed Material means the artistic or literary work, database,
     or other material to which the Licensor applied this Public
     License.

  i. Licensed Rights means the rights granted to You subject to the
     terms and conditions of this Public License, which are limited to
     all Copyright and Similar Rights that apply to Your use of the
     Licensed Material and that the Licensor has authority to license.

  j. Licensor means the individual(s) or entity(ies) granting rights
     under this Public License.

  k. NonCommercial means not primarily intended for or directed towards
     commercial advantage or monetary compensation. For purposes of
     this Public License, the exchange of the Licensed Material for
     other material subject to Copyright and Similar Rights by digital
     file-sharing or similar means is NonCommercial provided there is
     no payment of monetary compensation in connection with the
     exchange.

  l. Share means to provide material to the public by any means or
     process that requires permission under the Licensed Rights, such
     as reproduction, public display, public performance, distribution,
     dissemination, communication, or importation, and to make material
     available to the public including in ways that members of the
     public may access the material from a place and at a time
     individually chosen by them.

  m. Sui Generis Database Rights means rights other than copyright
     resulting from Directive 96/9/EC of the European Parliament and of
     the Council of 11 March 1996 on the legal protection of databases,
     as amended and/or succeeded, as well as other essentially
     equivalent rights anywhere in the world.

  n. You means the individual or entity exercising the Licensed Rights
     under this Public License. Your has a corresponding meaning.


Section 2 -- Scope.

  a. License grant.

       1. Subject to the terms and conditions of this Public License,
          the Licensor hereby grants You a worldwide, royalty-free,
          non-sublicensable, non-exclusive, irrevocable license to
          exercise the Licensed Rights in the Licensed Material to:

            a. reproduce and Share the Licensed Material, in whole or
               in part, for NonCommercial purposes only; and

            b. produce, reproduce, and Share Adapted Material for
               NonCommercial purposes only.

       2. Exceptions and Limitations. For the avoidance of doubt, where
          Exceptions and Limitations apply to Your use, this Public
          License does not apply, and You do not need to comply with
          its terms and conditions.

       3. Term. The term of this Public License is specified in Section
          6(a).

       4. Media and formats; technical modifications allowed. The
          Licensor authorizes You to exercise the Licensed Rights in
          all media and formats whether now known or hereafter created,
          and to make technical modifications necessary to do so. The
          Licensor waives and/or agrees not to assert any right or
          authority to forbid You from making technical modifications
          necessary to exercise the Licensed Rights, including
          technical modifications necessary to circumvent Effective
          Technological Measures. For purposes of this Public License,
          simply making modifications authorized by this Section 2(a)
          (4) never produces Adapted Material.

       5. Downstream recipients.

            a. Offer from the Licensor -- Licensed Material. Every
               recipient of the Licensed Material automatically
               receives an offer from the Licensor to exercise the
               Licensed Rights under the terms and conditions of this
               Public License.

            b. Additional offer from the Licensor -- Adapted Material.
               Every recipient of Adapted Material from You
               automatically receives an offer from the Licensor to
               exercise the Licensed Rights in the Adapted Material
               under the conditions of the Adapter's License You apply.

            c. No downstream restrictions. You may not offer or impose
               any additional or different terms or conditions on, or
               apply any Effective Technological Measures to, the
               Licensed Material if doing so restricts exercise of the
               Licensed Rights by any recipient of the Licensed
               Material.

       6. No endorsement. Nothing in this Public License constitutes or
          may be construed as permission to assert or imply that You
          are, or that Your use of the Licensed Material is, connected
          with, or sponsored, endorsed, or granted official status by,
          the Licensor or others designated to receive attribution as
          provided in Section 3(a)(1)(A)(i).

  b. Other rights.

       1. Moral rights, such as the right of integrity, are not
          licensed under this Public License, nor are publicity,
          privacy, and/or other similar personality rights; however, to
          the extent possible, the Licensor waives and/or agrees not to
          assert any such rights held by the Licensor to the limited
          extent necessary to allow You to exercise the Licensed
          Rights, but not otherwise.

       2. Patent and trademark rights are not licensed under this
          Public License.

       3. To the extent possible, the Licensor waives any right to
          collect royalties from You for the exercise of the Licensed
          Rights, whether directly or through a collecting society
          under any voluntary or waivable statutory or compulsory
          licensing scheme. In all other cases the Licensor expressly
          reserves any right to collect such royalties, including when
          the Licensed Material is used other than for NonCommercial
          purposes.


Section 3 -- License Conditions.

Your exercise of the Licensed Rights is expressly made subject to the
following conditions.

  a. Attribution.

       1. If You Share the Licensed Material (including in modified
          form), You must:

            a. retain the following if it is supplied by the Licensor
               with the Licensed Material:

                 i. identification of the creator(s) of the Licensed
                    Material and any others designated to receive
                    attribution, in any reasonable manner requested by
                    the Licensor (including by pseudonym if
                    designated);

                ii. a copyright notice;

               iii. a notice that refers to this Public License;

                iv. a notice that refers to the disclaimer of
                    warranties;

                 v. a URI or hyperlink to the Licensed Material to the
                    extent reasonably practicable;

            b. indicate if You modified the Licensed Material and
               retain an indication of any previous modifications; and

            c. indicate the Licensed Material is licensed under this
               Public License, and include the text of, or the URI or
               hyperlink to, this Public License.

       2. You may satisfy the conditions in Section 3(a)(1) in any
          reasonable manner based on the medium, means, and context in
          which You Share the Licensed Material. For example, it may be
          reasonable to satisfy the conditions by providing a URI or
          hyperlink to a resource that includes the required
          information.
       3. If requested by the Licensor, You must remove any of the
          information required by Section 3(a)(1)(A) to the extent
          reasonably practicable.

  b. ShareAlike.

     In addition to the conditions in Section 3(a), if You Share
     Adapted Material You produce, the following conditions also apply.

       1. The Adapter's License You apply must be a Creative Commons
          license with the same License Elements, this version or
          later, or a BY-NC-SA Compatible License.

       2. You must include the text of, or the URI or hyperlink to, the
          Adapter's License You apply. You may satisfy this condition
          in any reasonable manner based on the medium, means, and
          context in which You Share Adapted Material.

       3. You may not offer or impose any additional or different terms
          or conditions on, or apply any Effective Technological
          Measures to, Adapted Material that restrict exercise of the
          rights granted under the Adapter's License You apply.


Section 4 -- Sui Generis Database Rights.

Where the Licensed Rights include Sui Generis Database Rights that
apply to Your use of the Licensed Material:

  a. for the avoidance of doubt, Section 2(a)(1) grants You the right
     to extract, reuse, reproduce, and Share all or a substantial
     portion of the contents of the database for NonCommercial purposes
     only;

  b. if You include all or a substantial portion of the database
     contents in a database in which You have Sui Generis Database
     Rights, then the database in which You have Sui Generis Database
     Rights (but not its individual contents) is Adapted Material,
     including for purposes of Section 3(b); and

  c. You must comply with the conditions in Section 3(a) if You Share
     all or a substantial portion of the contents of the database.

For the avoidance of doubt, this Section 4 supplements and does not
replace Your obligations under this Public License where the Licensed
Rights include other Copyright and Similar Rights.


Section 5 -- Disclaimer of Warranties and Limitation of Liability.

  a. UNLESS OTHERWISE SEPARATELY UNDERTAKEN BY THE LICENSOR, TO THE
     EXTENT POSSIBLE, THE LICENSOR OFFERS THE LICENSED MATERIAL AS-IS
     AND AS-AVAILABLE, AND MAKES NO REPRESENTATIONS OR WARRANTIES OF
     ANY KIND CONCERNING THE LICENSED MATERIAL, WHETHER EXPRESS,
     IMPLIED, STATUTORY, OR OTHER. THIS INCLUDES, WITHOUT LIMITATION,
     WARRANTIES OF TITLE, MERCHANTABILITY, FITNESS FOR A PARTICULAR
     PURPOSE, NON-INFRINGEMENT, ABSENCE OF LATENT OR OTHER DEFECTS,
     ACCURACY, OR THE PRESENCE OR ABSENCE OF ERRORS, WHETHER OR NOT
     KNOWN OR DISCOVERABLE. WHERE DISCLAIMERS OF WARRANTIES ARE NOT
     ALLOWED IN FULL OR IN PART, THIS DISCLAIMER MAY NOT APPLY TO YOU.

  b. TO THE EXTENT POSSIBLE, IN NO EVENT WILL THE LICENSOR BE LIABLE
     TO YOU ON ANY LEGAL THEORY (INCLUDING, WITHOUT LIMITATION,
     NEGLIGENCE) OR OTHERWISE FOR ANY DIRECT, SPECIAL, INDIRECT,
     INCIDENTAL, CONSEQUENTIAL, PUNITIVE, EXEMPLARY, OR OTHER LOSSES,
     COSTS, EXPENSES, OR DAMAGES ARISING OUT OF THIS PUBLIC LICENSE OR
     USE OF THE LICENSED MATERIAL, EVEN IF THE LICENSOR HAS BEEN
     ADVISED OF THE POSSIBILITY OF SUCH LOSSES, COSTS, EXPENSES, OR
     DAMAGES. WHERE A LIMITATION OF LIABILITY IS NOT ALLOWED IN FULL OR
     IN PART, THIS LIMITATION MAY NOT APPLY TO YOU.

  c. The disclaimer of warranties and limitation of liability provided
     above shall be interpreted in a manner that, to the extent
     possible, most closely approximates an absolute disclaimer and
     waiver of all liability.


Section 6 -- Term and Termination.

  a. This Public License applies for the term of the Copyright and
     Similar Rights licensed here. However, if You fail to comply with
     this Public License, then Your rights under this Public License
     terminate automatically.

  b. Where Your right to use the Licensed Material has terminated under
     Section 6(a), it reinstates:

       1. automatically as of the date the violation is cured, provided
          it is cured within 30 days of Your discovery of the
          violation; or

       2. upon express reinstatement by the Licensor.

     For the avoidance of doubt, this Section 6(b) does not affect any
     right the Licensor may have to seek remedies for Your violations
     of this Public License.

  c. For the avoidance of doubt, the Licensor may also offer the
     Licensed Material under separate terms or conditions or stop
     distributing the Licensed Material at any time; however, doing so
     will not terminate this Public License.

  d. Sections 1, 5, 6, 7, and 8 survive termination of this Public
     License.


Section 7 -- Other Terms and Conditions.

  a. The Licensor shall not be bound by any additional or different
     terms or conditions communicated by You unless expressly agreed.

  b. Any arrangements, understandings, or agreements regarding the
     Licensed Material not stated herein are separate from and
     independent of the terms and conditions of this Public License.


Section 8 -- Interpretation.

  a. For the avoidance of doubt, this Public License does not, and
     shall not be interpreted to, reduce, limit, restrict, or impose
     conditions on any use of the Licensed Material that could lawfully
     be made without permission under this Public License.

  b. To the extent possible, if any provision of this Public License is
     deemed unenforceable, it shall be automatically reformed to the
     minimum extent necessary to make it enforceable. If the provision
     cannot be reformed, it shall be severed from this Public License
     without affecting the enforceability of the remaining terms and
     conditions.

  c. No term or condition of this Public License will be waived and no
     failure to comply consented to unless expressly agreed to by the
     Licensor.

  d. Nothing in this Public License constitutes or may be interpreted
     as a limitation upon, or waiver of, any privileges and immunities
     that apply to the Licensor or You, including from the legal
     processes of any jurisdiction or authority.

=======================================================================

Creative Commons is not a party to its public
licenses. Notwithstanding, Creative Commons may elect to apply one of
its public licenses to material it publishes and in those instances
will be considered the “Licensor.” The text of the Creative Commons
public licenses is dedicated to the public domain under the CC0 Public
Domain Dedication. Except for the limited purpose of indicating that
material is shared under a Creative Commons public license or as
otherwise permitted by the Creative Commons policies published at
creativecommons.org/policies, Creative Commons does not authorize the
use of the trademark "Creative Commons" or any other trademark or logo
of Creative Commons without its prior written consent including,
without limitation, in connection with any unauthorized modifications
to any of its public licenses or any other arrangements,
understandings, or agreements concerning use of licensed material. For
the avoidance of doubt, this paragraph does not form part of the
public licenses.

Creative Commons may be contacted at creativecommons.org.


```

### Full CC BY4.0 legal text

```text
Attribution 4.0 International

=======================================================================

Creative Commons Corporation ("Creative Commons") is not a law firm and
does not provide legal services or legal advice. Distribution of
Creative Commons public licenses does not create a lawyer-client or
other relationship. Creative Commons makes its licenses and related
information available on an "as-is" basis. Creative Commons gives no
warranties regarding its licenses, any material licensed under their
terms and conditions, or any related information. Creative Commons
disclaims all liability for damages resulting from their use to the
fullest extent possible.

Using Creative Commons Public Licenses

Creative Commons public licenses provide a standard set of terms and
conditions that creators and other rights holders may use to share
original works of authorship and other material subject to copyright
and certain other rights specified in the public license below. The
following considerations are for informational purposes only, are not
exhaustive, and do not form part of our licenses.

     Considerations for licensors: Our public licenses are
     intended for use by those authorized to give the public
     permission to use material in ways otherwise restricted by
     copyright and certain other rights. Our licenses are
     irrevocable. Licensors should read and understand the terms
     and conditions of the license they choose before applying it.
     Licensors should also secure all rights necessary before
     applying our licenses so that the public can reuse the
     material as expected. Licensors should clearly mark any
     material not subject to the license. This includes other CC-
     licensed material, or material used under an exception or
     limitation to copyright. More considerations for licensors:
    wiki.creativecommons.org/Considerations_for_licensors

     Considerations for the public: By using one of our public
     licenses, a licensor grants the public permission to use the
     licensed material under specified terms and conditions. If
     the licensor's permission is not necessary for any reason--for
     example, because of any applicable exception or limitation to
     copyright--then that use is not regulated by the license. Our
     licenses grant only permissions under copyright and certain
     other rights that a licensor has authority to grant. Use of
     the licensed material may still be restricted for other
     reasons, including because others have copyright or other
     rights in the material. A licensor may make special requests,
     such as asking that all changes be marked or described.
     Although not required by our licenses, you are encouraged to
     respect those requests where reasonable. More considerations
     for the public:
    wiki.creativecommons.org/Considerations_for_licensees

=======================================================================

Creative Commons Attribution 4.0 International Public License

By exercising the Licensed Rights (defined below), You accept and agree
to be bound by the terms and conditions of this Creative Commons
Attribution 4.0 International Public License ("Public License"). To the
extent this Public License may be interpreted as a contract, You are
granted the Licensed Rights in consideration of Your acceptance of
these terms and conditions, and the Licensor grants You such rights in
consideration of benefits the Licensor receives from making the
Licensed Material available under these terms and conditions.


Section 1 -- Definitions.

  a. Adapted Material means material subject to Copyright and Similar
     Rights that is derived from or based upon the Licensed Material
     and in which the Licensed Material is translated, altered,
     arranged, transformed, or otherwise modified in a manner requiring
     permission under the Copyright and Similar Rights held by the
     Licensor. For purposes of this Public License, where the Licensed
     Material is a musical work, performance, or sound recording,
     Adapted Material is always produced where the Licensed Material is
     synched in timed relation with a moving image.

  b. Adapter's License means the license You apply to Your Copyright
     and Similar Rights in Your contributions to Adapted Material in
     accordance with the terms and conditions of this Public License.

  c. Copyright and Similar Rights means copyright and/or similar rights
     closely related to copyright including, without limitation,
     performance, broadcast, sound recording, and Sui Generis Database
     Rights, without regard to how the rights are labeled or
     categorized. For purposes of this Public License, the rights
     specified in Section 2(b)(1)-(2) are not Copyright and Similar
     Rights.

  d. Effective Technological Measures means those measures that, in the
     absence of proper authority, may not be circumvented under laws
     fulfilling obligations under Article 11 of the WIPO Copyright
     Treaty adopted on December 20, 1996, and/or similar international
     agreements.

  e. Exceptions and Limitations means fair use, fair dealing, and/or
     any other exception or limitation to Copyright and Similar Rights
     that applies to Your use of the Licensed Material.

  f. Licensed Material means the artistic or literary work, database,
     or other material to which the Licensor applied this Public
     License.

  g. Licensed Rights means the rights granted to You subject to the
     terms and conditions of this Public License, which are limited to
     all Copyright and Similar Rights that apply to Your use of the
     Licensed Material and that the Licensor has authority to license.

  h. Licensor means the individual(s) or entity(ies) granting rights
     under this Public License.

  i. Share means to provide material to the public by any means or
     process that requires permission under the Licensed Rights, such
     as reproduction, public display, public performance, distribution,
     dissemination, communication, or importation, and to make material
     available to the public including in ways that members of the
     public may access the material from a place and at a time
     individually chosen by them.

  j. Sui Generis Database Rights means rights other than copyright
     resulting from Directive 96/9/EC of the European Parliament and of
     the Council of 11 March 1996 on the legal protection of databases,
     as amended and/or succeeded, as well as other essentially
     equivalent rights anywhere in the world.

  k. You means the individual or entity exercising the Licensed Rights
     under this Public License. Your has a corresponding meaning.


Section 2 -- Scope.

  a. License grant.

       1. Subject to the terms and conditions of this Public License,
          the Licensor hereby grants You a worldwide, royalty-free,
          non-sublicensable, non-exclusive, irrevocable license to
          exercise the Licensed Rights in the Licensed Material to:

            a. reproduce and Share the Licensed Material, in whole or
               in part; and

            b. produce, reproduce, and Share Adapted Material.

       2. Exceptions and Limitations. For the avoidance of doubt, where
          Exceptions and Limitations apply to Your use, this Public
          License does not apply, and You do not need to comply with
          its terms and conditions.

       3. Term. The term of this Public License is specified in Section
          6(a).

       4. Media and formats; technical modifications allowed. The
          Licensor authorizes You to exercise the Licensed Rights in
          all media and formats whether now known or hereafter created,
          and to make technical modifications necessary to do so. The
          Licensor waives and/or agrees not to assert any right or
          authority to forbid You from making technical modifications
          necessary to exercise the Licensed Rights, including
          technical modifications necessary to circumvent Effective
          Technological Measures. For purposes of this Public License,
          simply making modifications authorized by this Section 2(a)
          (4) never produces Adapted Material.

       5. Downstream recipients.

            a. Offer from the Licensor -- Licensed Material. Every
               recipient of the Licensed Material automatically
               receives an offer from the Licensor to exercise the
               Licensed Rights under the terms and conditions of this
               Public License.

            b. No downstream restrictions. You may not offer or impose
               any additional or different terms or conditions on, or
               apply any Effective Technological Measures to, the
               Licensed Material if doing so restricts exercise of the
               Licensed Rights by any recipient of the Licensed
               Material.

       6. No endorsement. Nothing in this Public License constitutes or
          may be construed as permission to assert or imply that You
          are, or that Your use of the Licensed Material is, connected
          with, or sponsored, endorsed, or granted official status by,
          the Licensor or others designated to receive attribution as
          provided in Section 3(a)(1)(A)(i).

  b. Other rights.

       1. Moral rights, such as the right of integrity, are not
          licensed under this Public License, nor are publicity,
          privacy, and/or other similar personality rights; however, to
          the extent possible, the Licensor waives and/or agrees not to
          assert any such rights held by the Licensor to the limited
          extent necessary to allow You to exercise the Licensed
          Rights, but not otherwise.

       2. Patent and trademark rights are not licensed under this
          Public License.

       3. To the extent possible, the Licensor waives any right to
          collect royalties from You for the exercise of the Licensed
          Rights, whether directly or through a collecting society
          under any voluntary or waivable statutory or compulsory
          licensing scheme. In all other cases the Licensor expressly
          reserves any right to collect such royalties.


Section 3 -- License Conditions.

Your exercise of the Licensed Rights is expressly made subject to the
following conditions.

  a. Attribution.

       1. If You Share the Licensed Material (including in modified
          form), You must:

            a. retain the following if it is supplied by the Licensor
               with the Licensed Material:

                 i. identification of the creator(s) of the Licensed
                    Material and any others designated to receive
                    attribution, in any reasonable manner requested by
                    the Licensor (including by pseudonym if
                    designated);

                ii. a copyright notice;

               iii. a notice that refers to this Public License;

                iv. a notice that refers to the disclaimer of
                    warranties;

                 v. a URI or hyperlink to the Licensed Material to the
                    extent reasonably practicable;

            b. indicate if You modified the Licensed Material and
               retain an indication of any previous modifications; and

            c. indicate the Licensed Material is licensed under this
               Public License, and include the text of, or the URI or
               hyperlink to, this Public License.

       2. You may satisfy the conditions in Section 3(a)(1) in any
          reasonable manner based on the medium, means, and context in
          which You Share the Licensed Material. For example, it may be
          reasonable to satisfy the conditions by providing a URI or
          hyperlink to a resource that includes the required
          information.

       3. If requested by the Licensor, You must remove any of the
          information required by Section 3(a)(1)(A) to the extent
          reasonably practicable.

       4. If You Share Adapted Material You produce, the Adapter's
          License You apply must not prevent recipients of the Adapted
          Material from complying with this Public License.


Section 4 -- Sui Generis Database Rights.

Where the Licensed Rights include Sui Generis Database Rights that
apply to Your use of the Licensed Material:

  a. for the avoidance of doubt, Section 2(a)(1) grants You the right
     to extract, reuse, reproduce, and Share all or a substantial
     portion of the contents of the database;

  b. if You include all or a substantial portion of the database
     contents in a database in which You have Sui Generis Database
     Rights, then the database in which You have Sui Generis Database
     Rights (but not its individual contents) is Adapted Material; and

  c. You must comply with the conditions in Section 3(a) if You Share
     all or a substantial portion of the contents of the database.

For the avoidance of doubt, this Section 4 supplements and does not
replace Your obligations under this Public License where the Licensed
Rights include other Copyright and Similar Rights.


Section 5 -- Disclaimer of Warranties and Limitation of Liability.

  a. UNLESS OTHERWISE SEPARATELY UNDERTAKEN BY THE LICENSOR, TO THE
     EXTENT POSSIBLE, THE LICENSOR OFFERS THE LICENSED MATERIAL AS-IS
     AND AS-AVAILABLE, AND MAKES NO REPRESENTATIONS OR WARRANTIES OF
     ANY KIND CONCERNING THE LICENSED MATERIAL, WHETHER EXPRESS,
     IMPLIED, STATUTORY, OR OTHER. THIS INCLUDES, WITHOUT LIMITATION,
     WARRANTIES OF TITLE, MERCHANTABILITY, FITNESS FOR A PARTICULAR
     PURPOSE, NON-INFRINGEMENT, ABSENCE OF LATENT OR OTHER DEFECTS,
     ACCURACY, OR THE PRESENCE OR ABSENCE OF ERRORS, WHETHER OR NOT
     KNOWN OR DISCOVERABLE. WHERE DISCLAIMERS OF WARRANTIES ARE NOT
     ALLOWED IN FULL OR IN PART, THIS DISCLAIMER MAY NOT APPLY TO YOU.

  b. TO THE EXTENT POSSIBLE, IN NO EVENT WILL THE LICENSOR BE LIABLE
     TO YOU ON ANY LEGAL THEORY (INCLUDING, WITHOUT LIMITATION,
     NEGLIGENCE) OR OTHERWISE FOR ANY DIRECT, SPECIAL, INDIRECT,
     INCIDENTAL, CONSEQUENTIAL, PUNITIVE, EXEMPLARY, OR OTHER LOSSES,
     COSTS, EXPENSES, OR DAMAGES ARISING OUT OF THIS PUBLIC LICENSE OR
     USE OF THE LICENSED MATERIAL, EVEN IF THE LICENSOR HAS BEEN
     ADVISED OF THE POSSIBILITY OF SUCH LOSSES, COSTS, EXPENSES, OR
     DAMAGES. WHERE A LIMITATION OF LIABILITY IS NOT ALLOWED IN FULL OR
     IN PART, THIS LIMITATION MAY NOT APPLY TO YOU.

  c. The disclaimer of warranties and limitation of liability provided
     above shall be interpreted in a manner that, to the extent
     possible, most closely approximates an absolute disclaimer and
     waiver of all liability.


Section 6 -- Term and Termination.

  a. This Public License applies for the term of the Copyright and
     Similar Rights licensed here. However, if You fail to comply with
     this Public License, then Your rights under this Public License
     terminate automatically.

  b. Where Your right to use the Licensed Material has terminated under
     Section 6(a), it reinstates:

       1. automatically as of the date the violation is cured, provided
          it is cured within 30 days of Your discovery of the
          violation; or

       2. upon express reinstatement by the Licensor.

     For the avoidance of doubt, this Section 6(b) does not affect any
     right the Licensor may have to seek remedies for Your violations
     of this Public License.

  c. For the avoidance of doubt, the Licensor may also offer the
     Licensed Material under separate terms or conditions or stop
     distributing the Licensed Material at any time; however, doing so
     will not terminate this Public License.

  d. Sections 1, 5, 6, 7, and 8 survive termination of this Public
     License.


Section 7 -- Other Terms and Conditions.

  a. The Licensor shall not be bound by any additional or different
     terms or conditions communicated by You unless expressly agreed.

  b. Any arrangements, understandings, or agreements regarding the
     Licensed Material not stated herein are separate from and
     independent of the terms and conditions of this Public License.


Section 8 -- Interpretation.

  a. For the avoidance of doubt, this Public License does not, and
     shall not be interpreted to, reduce, limit, restrict, or impose
     conditions on any use of the Licensed Material that could lawfully
     be made without permission under this Public License.

  b. To the extent possible, if any provision of this Public License is
     deemed unenforceable, it shall be automatically reformed to the
     minimum extent necessary to make it enforceable. If the provision
     cannot be reformed, it shall be severed from this Public License
     without affecting the enforceability of the remaining terms and
     conditions.

  c. No term or condition of this Public License will be waived and no
     failure to comply consented to unless expressly agreed to by the
     Licensor.

  d. Nothing in this Public License constitutes or may be interpreted
     as a limitation upon, or waiver of, any privileges and immunities
     that apply to the Licensor or You, including from the legal
     processes of any jurisdiction or authority.


=======================================================================

Creative Commons is not a party to its public
licenses. Notwithstanding, Creative Commons may elect to apply one of
its public licenses to material it publishes and in those instances
will be considered the “Licensor.” The text of the Creative Commons
public licenses is dedicated to the public domain under the CC0 Public
Domain Dedication. Except for the limited purpose of indicating that
material is shared under a Creative Commons public license or as
otherwise permitted by the Creative Commons policies published at
creativecommons.org/policies, Creative Commons does not authorize the
use of the trademark "Creative Commons" or any other trademark or logo
of Creative Commons without its prior written consent including,
without limitation, in connection with any unauthorized modifications
to any of its public licenses or any other arrangements,
understandings, or agreements concerning use of licensed material. For
the avoidance of doubt, this paragraph does not form part of the
public licenses.

Creative Commons may be contacted at creativecommons.org.


```

## Archived source document hashes

| Private artifact | SHA256 |
| --- | --- |
| upstream_f3e7813.zip | `d657a6157fe80c5fd3a19790bfbeac8b80e24c155f1de62225e3f0e9d750ef93` |
| paper_hal.pdf | `37604f0ddf4124c86dde7eac8f435569a72eca47361966244ae1d6de9d919f3f` |
| supplement_hal.pdf | `1f075fcb9d494a22de386e055f054be76d7bc487c8784fcd86264ba6e1bbe17a` |
| addendum.pdf | `f0ec8a10cd9b4794c450ac70eafaf9645f9a5cea6281d039f97c89d0b7d37200` |
| data_readme.md | `7690aa430f868da6fc03c42ffe62db4ab7755034cfbd443fe99f6471414e8cc4` |
| root-complete-member-manifest.json | `c9d09dfe9a96bbef03d6f483ca7cb203e9d00b6e7c272f4d3842885d0c864db9` |
| preimplementation_criteria.json | `6b92f9cd438b9c35e8f1fdea21476a9d485b71cb4e8bf7efa026d62c5d6b847b` |
| precision_diagnostic.json | `28de396811d840ffb9f3abc34473b8309d1b146296fcb936b683b149aa2839c6` |
| allblock_precision_and_fallback_audit.json | `c8e2aa4c418c7881a080c74e42132c842ab8eb0907c86ddb4c49c07b6ec62cd9` |


## Final executed validation and gate ledger

The selected double-computation original run completed successfully: exit 0, wall 2776.478530 s, with all 30 exact real observation blocks and all three scientific comparison methods. Its source functions remain unmodified; `double(IQ)` is the explicitly identified driver-level precision variant. The previously failed single cross-backend parity is preserved as a disclosed numerical finding, not relabeled as a successful single reproduction. Source-single and source-double results remain separately identified. No frozen stage or output criterion was relaxed.

Original-double total [accepted positions, retained tracks], in method order radial/weighted centroid/integer: `[[1019468.0, 15622.0], [1068647.0, 15931.0], [1068647.0, 18107.0]]`. These are measurements of the selected finite acquisition and processing functional, not anatomical bubble counts or biological ground truth.

| Executed check | Exit code | Wall seconds | Private evidence |
| --- | ---: | ---: | --- |
| cli | 0 | 90.684892 | `cli.log` and `cli.stderr` |
| judge | 0 | 0.721406 | `judge.log` and `judge.stderr` |
| pytest | 0 | 16.550669 | `pytest.log` and `pytest.stderr` |
| notebook | 0 | 209.502754 | `notebook.log` and `notebook.stderr` |

Final unit/parity/physics/Judge/pipeline suite: **41 passed, zero failed, zero skipped**. The additional bool-threshold cases were explicitly requested during independent source review; both `True` and `False` configuration values are rejected. Original first-block parity and frozen independent numeric cases pass. Final source, tests and executed notebook use the same shipped implementation.

| Fresh required-output metric | Measured value |
| --- | ---: |
| reconstruction_ncc | 1 |
| reconstruction_nrmse | 0 |
| speed_ncc | 0.99999999999999978 |
| speed_nrmse | 6.3979330615304972e-10 |

The active notebook completed the full three-method 6000-frame path. It contains exactly the numbered six sections Problem, Data, Physics, Method, Results and Conclusion. Every code cell executed; no error output exists. Eight PNG outputs were extracted privately and individually inspected: actual acquisition snapshots; raw/clutter-filtered views and time series; three-method density comparisons; three-method speed comparisons; paired localization/track counts, profiles and saturation; observed-point versus interpolated-path crossing support; original-single versus original-double precision differences; and trajectory-length/spatial-spectrum comparisons. Units, transform labels, separate upstream/fresh identities, meaningful differences and legends are visible. Figure review details are indexed in `figure-review.json`; the only executed notebook shipped is the primary review notebook.

SHA256 protection before the final CLI and notebook and after their completion confirmed unchanged raw input, original references and fixtures. The exact hashes and complete commands are in `final-verification-status.json`. `final-task-manifest.json` indexes the final task files after this ledger is written; archive evidence includes commands, stdout/stderr and the temporary executed notebook retained outside the task.

The supplementary radial-distance typography was checked against the weighted squared-line-distance derivation, upstream normal equations and an independently assembled least-squares/KKT case. The local code preserves the correct source objective, not a squared version of the printed missing-square-root distance. The unused source fallback-sign discrepancy is disclosed above and exhaustively absent on every bound candidate patch. The actual source collapsed-dimension zeroing, nine-mode removal, strict length criterion, no-gap assignment, endpoint smoothing, frame-abscissa interpolation, finite-track speed denominator and rendering precision were reviewed semantically against the plans. Array-shape/signature matching alone was not used as evidence of mathematical correctness.

| Mandatory gate | Status | Evidence |
| --- | --- | --- |
| Paper, exact revision, data hashes, traceability | passed | Exact remote commit and manuscript/supplement/addendum; thirty member CRC/SHA256 records and equation/function/test table above. |
| Original and cleaned runs on bound observations | passed | Both original single and selected original double runs process all 30 complete blocks; fresh canonical CLI and active notebook exit 0. |
| Independent upstream parity and physics | passed | Frozen original first-block stages/all-three methods, independent math cases, analytic/KKT/projector checks, and full-reference comparisons pass unchanged frozen criteria. |
| Constants, units, coordinates, precision | passed | README constants and source boundary behavior tested; complex128 variant identified; acquisition pixels and pixel/s retained. |
| CLI output and fresh Judge | passed | Canonical CLI and notebook independently write both declared arrays; standalone Judge passes all four metrics. |
| Judge failure and contract handling | passed | Missing keys/files, wrong shape/dtype, nonfinite/negative arrays, poor speed, bool/negative thresholds, missing/degenerate assets, CWD/report failures tested. |
| Public solvability | passed | README defines target functional, all observations/acquisition constants and required trajectory/grid/precision conventions without hidden reference access. |
| No public leakage | passed | Only unchanged measured IQ is public NPZ; no truth, target-derived calibration, target seed, reference arrays, or public GitHub URLs. |
| README metadata and contract | passed | Exactly one public key with batch-first table, explicit acquisition table, two-output contract; no metadata JSON. |
| Minimal validated CPU environment | passed | Recorded shared Python/NumPy/SciPy/Matplotlib versions; no upstream domain dependency in cleaned runtime. |
| Explicit local mathematics | passed | Local temporal projection, filter, regional maxima, radial weighted-line inverse, Munkres, smoothing/interpolation/rendering; source/Judge/test/notebook call-role review passed. |
| Complete test suite | passed | Final pytest log: 41 passed, no failures and no skips. |
| Active executed notebook | passed | All three methods freshly reconstruct every frame; all code cells executed without errors; eight scientific figures individually inspected. |
| Result and metric provenance | passed | Separate original native-single, selected original-double and fresh local measurements; identical NCC/NRMSE conventions and unchanged frozen boundaries. |
| Plans/source semantic consistency | passed | Manually checked forward model, weighted-distance denominator/signs, complete padded assignment costs including all-zero degeneracy, smoothing endpoints, 0.08 frame abscissae, n/500 denominator, float32 raster input, units and all actual signatures/flows. |
| Final inventory and required notices | passed | Task follows required template; legal notices consolidated here; no private source/PDF/archive/capture/temporary executed notebook or duplicate reference files in task. |


## Independent coordinator acceptance

The coordinator independently reviewed the shipped SVD/filtering, radial least-squares, explicit padded Munkres assignment, endpoint-aware smoothing, frame-index interpolation, float32 rendering quantization, units, signatures and plans. The ambiguous unconstrained partial-assignment wording was corrected to the actual padded complete-assignment cost convention before final notebook execution. All eight actual executed-notebook PNG figures were individually inspected: real IQ observations/filtering, three-method density and speed parity, paired counts/profiles/sampling, observed versus interpolated positions, original single versus double computation, and track-length/spatial-spectrum comparisons. Structural and notebook call-role reviews found no errors or forbidden solver shortcuts. A separate standalone Judge invocation from outside the task workspace passed density NCC=1/NRMSE=0 and speed NCC=0.9999999999999998/NRMSE=6.397933061530497e-10. Evidence: private research10/coordinator/ulm-external-cwd-judge.json and ulm-final-review.json. This accepts local CPU cleaning of the stated first30-block benchmark; independent anatomical truth, mm/pixel calibration and the documented licensing discrepancy are not inferred from reference agreement.
