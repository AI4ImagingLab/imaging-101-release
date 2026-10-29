# Provenance and validation

Status: implementation and initial 27 tests passed; independent full notebook, visual and final consistency gates pending. Do not mark task complete until those evidence records exist.

## Bound paper, original code and real observations

Paper: Makabe et al., Spectral Sensitivity Estimation with an Uncalibrated Diffraction Grating, ICCV 2025; https://arxiv.org/abs/2508.00330v1 (version 1). Original repository: https://github.com/lilika-makabe/camera-sensitivity-estimation-with-grating . Exact remotely verified commit 3316ba9ca3ccd93950444ea87eb07d5ba4a79ea9; remote tree and downloaded code are archived privately. This is not a local tracking-revision freshness claim.

Private archive root: research_batch11_20261027/root/sources/grating, relative to the maintainer directory. paper.pdf contains the supplied paper; upstream/ contains original source and data; real.zip and camspec_dataset.zip preserve author downloads; remote_tree.json records source inventory. Author-linked data folder: https://drive.google.com/drive/folders/1RbSQoDIuygaC_tRaqMGOTYMZop8xvgE8 . Real archive SHA256 60e15da0c28119df02ae74d78f18c1ff107d656bb64be7f9f2e3ee3cccf8165f; population archive SHA256 98307344b9c48582b472c1c58f8a8c9ed5d97a604a10db0d6e18bdc64c6aefc3.

No explicit code/data redistribution license was found in the verified repository tree or README. No permissive license is invented. This is a local research maintenance package; publication rights are not established. Source attribution is retained in NOTICE. Paper availability does not itself license code/data.

Images are author's processed real linear EXR crops, not synthesized from the target. All four full released image crops are retained. Public population samples are from 44 other cameras, with excluded names recorded in package_data_manifest.json. Public observations SHA256: 2cb05233aaceadcb92ebdcbc6ac84c73e94c3f5595826eb8380fa112605935af. Independent CamSPECS ground truth is sampled at the exact 31 wavelengths, with one global RGB peak normalization per camera. No reconstruction is relabeled truth.

## Original execution before extraction

Private root/run_grating_original.py runs unchanged src/estimate.py -m real_godox_CAMERA for EOS RP, Sony, iPhone and DJI. Each real_godox_*.execution.json records exact command, successful exit 0, runtime and isolated environment; matching stdout/stderr are retained. Native Python 3.11.11, NumPy 2.4.6, SciPy 1.17.1; Windows CPU, one OMP/OpenBLAS/MKL thread. Shared OpenCV 5 could not decode EXR; private oracle_cv410 contains the already installed OpenCV 4.10.0.84 used only by the original oracle. Private YAML parser is likewise upstream-only. The cleaned task has neither dependency and reads NPZ. No shared package was upgraded.

root/capture_grating.py uses read-only Python return tracing of unchanged original functions, recording numerical locals and returned arrays in upstream_capture.npz, with stage environment/revision information in upstream_capture.json. The four baseline original CLI runs are separately recorded, so tracing is not a replacement for original execution. root/capture_grating_comparisons.py records all 20 alternate original reconstructions in upstream_comparisons.npz/json. Its terminal PowerShell redirection reported NativeCommandError from progress-bar stderr despite reaching CAPTURED2404; this shell status is not presented as a successful independent CLI exit. The explicit baseline subprocess exit statuses are the original-run gate evidence.

frozen_acceptance.json predates local extraction. Preprocessing tolerance 1e-12, mapping 1e-8, matrix rtol 1e-8/atol 1e-10, normalized output atol 1e-7. The initial proposed reference used a pseudoinverse-projected extended-grid target; this was rejected before any local extraction because extrapolation mixes the final bins. initial_design_pinv_reference_rejected.json preserves that design; actual measured 400--700 nm points are now the reference. No cleaned result was used to relax criteria.

## Traceability

| Paper stage | Original file/function | Local function | Validation |
|---|---|---|---|
| Real profile normalization | src/loader/realworld_dataset.py, load_images | preprocessing.observations | test_parity full camera stages |
| Population subspace | src/utils/camspec_util.py, compute_camspec_basis | preprocessing.population_basis | frozen bases and spectral products |
| Wavelength calibration | original get_initial_pix2wave_params / pix2wave_mat | solvers.peak_mapping; physics_model.wavelength_kernel | four maps, affine interpolation, kernel support |
| Smooth reciprocal throughput | original get_fourier_basis | physics_model.fourier_basis | analytic harmonic samples and frozen matrices |
| Supplement constrained spectral solve | src/estimate.py, estimate_sensitivity_core | physics_model.spectral_system; solvers.reconstruct_camera | all KKT entries, constraints, solutions, outputs |
| LED-only polynomial ICP | original mapping estimator | solvers.icp_mapping | four independently captured 500-step outputs |
| Basis comparisons | original estimator with alternate basis counts | reconstruct_camera keyword parameters | twenty frozen original results |
| Relative response evaluation | independent CamSPECS calibration | standalone judge and visualization.metrics | independently known numeric metric cases and malformed input tests |

The upstream code refers to Supplement E, while the supplied paper places estimation details in Supplement F; this editorial mismatch does not change equations. Eigenbasis signs are arbitrary; comparisons cover reconstructed operators/results, not a new physical convention. The method has unconstrained signed sensitivity and nuisance efficiency. Sony's negative fitted efficiency is exposed, not clipped, and is not claimed to be validated physical throughput.

## Reference consumers and metric provenance

notebooks/reference_outputs/upstream.npz contains only normalized results from the independent original run, consumed by notebook parity and response panels. test/fixtures/upstream_stages.npz contains captured original stages consumed by comparing_test/test_parity.py. test/fixtures/upstream_comparisons.npz contains the 20 original alternative results consumed by the same tests. evaluation/ground_truth.npz contains independently measured response points, consumed by the standalone Judge and review notebook. No baseline_reference is needed. All full source measurements/capture logs remain in the private archive, not duplicated in the package.

The upstream paper metric uses a projected response and Frobenius/channel relative error. It is not relabeled NRMSE. This task's fresh and reproduced baseline metrics use actual 31-point measured truth, cosine NCC without centering, and RMS error divided by reference range, with no rescale/alignment. Per-camera frozen original metrics are recorded in evaluation/metrics.json; boundaries are 0.9 baseline NCC and 1.1 baseline NRMSE. Each camera plus global peak normalization must pass. Signed coefficients remain in scoring.

## Current gate ledger

| Gate | State | Evidence |
|---|---|---|
| Exact paper/revision/data and stage mapping | passed | private original archive, this table, hashes below |
| Original full bound observations | passed | four original CLI exit-0 JSON logs |
| Local stage/full parity and analytic physics | passed | 27-test initial run; normalized output max difference 3.76e-14 |
| Judge good/poor/malformed/evaluator contracts | passed | test_physics_judge tests |
| Fresh local CLI and Judge | passed | output/reconstruction.npz, judge_report.json |
| Runtime without upstream packages | passed | NPZ-only NumPy/SciPy/Matplotlib local imports |
| Full fresh notebook and visual inspection | unverified | independent parent execution pending |
| Public leak/metadata/interface consistency | unverified | final parent review pending |
| Plans mathematics/call semantics | unverified | final source review pending |
| Final inventory and complete acceptance | unverified | no completion claim |

## Packaged numerical hashes

| File | SHA256 |
|---|---|
| `data/raw_data.npz` | `2cb05233aaceadcb92ebdcbc6ac84c73e94c3f5595826eb8380fa112605935af` |
| `evaluation/ground_truth.npz` | `b59130655a834b06b4752c593a4850086ff1a881db81bd643ce23423182ee8d1` |
| `notebooks/reference_outputs/upstream.npz` | `d2bd2c7e77927dd1cb2459aac63475f812c5694839b07c5dc2ddf7038682f0d2` |
| `output/reconstruction.npz` | `d38424f49dc3e2955bf2e3a3e312571b8ce36ee234e5f025a4c61d5fced4a957` |
| `test/fixtures/upstream_comparisons.npz` | `9dd836a3d414f197195a4a0d52baf8a115614bdce6261fedb4366cd16e68d83c` |
| `test/fixtures/upstream_stages.npz` | `51189b452e2663c521ccee946835a1a75c921644ee15f4e250c95f546324e607` |

## Original data and source inventory

| Archived path | SHA256 |
|---|---|
| `upstream\data\camspec_dataset\Canon_10D_Huynh-2007.csv` | `ae0354e36c86f408e8c8b0a28c8970e280106b3b653edbb972fef88f13944a82` |
| `upstream\data\camspec_dataset\Canon_10D_Kawakami-et-al-2013.csv` | `5f6c2df3031ecf4e230bbc217e03af5cba63fa04fa9245a20b492a48533f3685` |
| `upstream\data\camspec_dataset\Canon_1D-Mk-III_Jiang-et-al-2013.csv` | `0dcbc15414575341d31293674ae1e530293e6f3f6c2b87ec538e7ee2d8fecf11` |
| `upstream\data\camspec_dataset\Canon_1Ds-Mk-II_Akkaynak-et-al-2011.csv` | `ee34944e58583e75e1f336e92fcb6382aa63685fcb1bd72a4528235fc252caee` |
| `upstream\data\camspec_dataset\Canon_1Ds-Mk-II_Brady-2009.csv` | `e894e045c36254706bf6c0d2cc40084d4cec87b09bc73e6eebe12ca8b1fb2f88` |
| `upstream\data\camspec_dataset\Canon_20D_Jiang-et-al-2013.csv` | `a90fcb3103c752607c9ee1c7316916cb445a44c1c6726921abc3a15152a79070` |
| `upstream\data\camspec_dataset\Canon_300D_Jiang-et-al-2013.csv` | `6988de798c66d38d490184a81b79dcd247e985bd34c2b3d3243693770f07fe17` |
| `upstream\data\camspec_dataset\Canon_400D_Lebourgeois-2008.csv` | `4f61cd9b21ea30b696991b015632687dee35361a8595a9a18fdb23bd1b468fb9` |
| `upstream\data\camspec_dataset\Canon_40D_Jiang-et-al-2013.csv` | `cfb1fbb9e2d900244844ce093bfc24a3916307f7a353c0c542922dc1ba8fcd2b` |
| `upstream\data\camspec_dataset\Canon_40D_Mauer-2009.csv` | `d62eeec30f900baf5495eca7a52c885db393c045c80dc39bc0fae61920eb2dde` |
| `upstream\data\camspec_dataset\Canon_500D_Jiang-et-al-2013.csv` | `15f7dd488e4afcc4ea6c655a33c4b815f1ebf25c02abcb03e123ba6bbb60dc9b` |
| `upstream\data\camspec_dataset\Canon_50D_Jiang-et-al-2013.csv` | `4f51db9abd6d68cce2eb90fd13fee327a0ffaf5cc340a14f328dd0eba8d1400c` |
| `upstream\data\camspec_dataset\Canon_5D-Mk-II_Jiang-et-al-2013.csv` | `79a7dd176a626bcb78dd087aa4925d870f3ffc7d7d60b46b772800965a7b2837` |
| `upstream\data\camspec_dataset\Canon_5D-Mk-II_Kawakami-et-al-2013.csv` | `4c3f241a52e761805546dac49ea8b41615725be8594cf9fa3c0a32ea70d38745` |
| `upstream\data\camspec_dataset\Canon_5D_Kawakami-et-al-2013.csv` | `6a2e062a3fbab9ee63c645145ae39e7cc985267c11d0cb81b4b615eab0dfa1fd` |
| `upstream\data\camspec_dataset\Canon_600D_Jiang-et-al-2013.csv` | `41618ea51ac52fbe4158aa76e268664993f3ddf99d7197ca893e31e0340b6430` |
| `upstream\data\camspec_dataset\Canon_60D_Jiang-et-al-2013.csv` | `2f81f1a1968b5a594953afe82fb8f68da8775ba4f2116594fb49c93de59361d1` |
| `upstream\data\camspec_dataset\iPhone_10_Tominaga_etal_2021.csv` | `c7c1ff318946a3bbd5241f05beb5d63ae195b8530c17c139c45471d8a34a7dfa` |
| `upstream\data\camspec_dataset\iPhone_11_Tominaga_etal_2021.csv` | `ba3059960f36fe3b3b40727ac0d79be6fcd2b627d8b6ad90356d130480e4d95f` |
| `upstream\data\camspec_dataset\iPhone_12-Pro-Max_Tominaga_etal_2021.csv` | `4cf716d73cd95512d4ccad4991d6a07ada566915b75c0c1618d930c8fa625819` |
| `upstream\data\camspec_dataset\iPhone_8_Tominaga_etal_2021.csv` | `02b7ffe9b2a28523cc4b93f0f5a1bfd5e06a60c573701109fdb8579172de3680` |
| `upstream\data\camspec_dataset\Leica_M8_Mauer-2009.csv` | `6db864a7bc9d3e700f0fa5396f74f8119f33aa5ce366cf4af2bb8fa9564ae113` |
| `upstream\data\camspec_dataset\Nikon_D1x_Kawakami-et-al-2013.csv` | `41d8a64687af2d9979a19918f6d8d21e2b21d4e82274d2e1c4385a080f523ff0` |
| `upstream\data\camspec_dataset\Nikon_D200_Jiang-et-al-2013.csv` | `218a8a9a62a1ad13eeeebaf4b1eb4535f35f114674f65493f28c02709a68fd15` |
| `upstream\data\camspec_dataset\Nikon_D200_Mauer-2009.csv` | `e61e0ed282d14a32766c955f44435c9ed439077566e0de753c782712ed7ff994` |
| `upstream\data\camspec_dataset\Nikon_D300s_Jiang-et-al-2013.csv` | `a12c3245d021970eaa8d6cdcaa6242b63c778529a40f9791a917543a38a794af` |
| `upstream\data\camspec_dataset\Nikon_D3X_Jiang-et-al-2013.csv` | `195d9e5bc23128e3a10beca513958aa9d9458ed9bcabc8c7a53998175de77dd2` |
| `upstream\data\camspec_dataset\Nikon_D3_Brady-2009.csv` | `9be322913a382695dc793a14630beeaf022911638aca6551cd666b6230bb1ad5` |
| `upstream\data\camspec_dataset\Nikon_D3_Jiang-et-al-2013.csv` | `c89908eb72d213cfc54b9086b509685c529c0089c716977aaf0b7dd1cd792753` |
| `upstream\data\camspec_dataset\Nikon_D40_Jiang-et-al-2013.csv` | `459bee43e0f267ed3e2a3fac1e6cdfd944572a3d8d5ce2973aeeb2eca9ae887b` |
| `upstream\data\camspec_dataset\Nikon_D50_Jiang-et-al-2013.csv` | `2689cce7b9d4107dede0ff4c46a90100cdb3c8163ab63dd74683d5f721583c66` |
| `upstream\data\camspec_dataset\Nikon_D5100_Jiang-et-al-2013.csv` | `34e749509a73401fb5a6ebc0db6f62d83b6016abea1e7a729329096201f780e9` |
| `upstream\data\camspec_dataset\Nikon_D700_Jiang-et-al-2013.csv` | `4a1e15cb1f2b4d6859d9463b788a5b223ede47ae51c33b0a0b75936ff295e4de` |
| `upstream\data\camspec_dataset\Nikon_D700_Mauer-2009.csv` | `7cb3f1fd3876f15d0dfcaee053cc42e80cec20f0d9269727af804047c1931b97` |
| `upstream\data\camspec_dataset\Nikon_D70_Huynh-2007.csv` | `6b20872c745dd1ef1ca5173a8be26cf12f9ec6bf7fce37df01ae59461f84a8c0` |
| `upstream\data\camspec_dataset\Nikon_D70_Kawakami-et-al-2013.csv` | `ce18de899e11d389182fc1dbabc1641f125ee5a1dd7e6e7ca1bb3dc2fbadb090` |
| `upstream\data\camspec_dataset\Nikon_D80_Jiang-et-al-2013.csv` | `1775a764bb83a25be656a53ffa69844c9e069b034a6477b251f30552bf1c9588` |
| `upstream\data\camspec_dataset\Nikon_D90_Jiang-et-al-2013.csv` | `a0e49479714d58442be5edbc1a28169ae5fd5c3c8db2fc0fd38c7ada9ca1c4d3` |
| `upstream\data\camspec_dataset\Olympus_EPL2_Jiang-et-al-2013.csv` | `02bb3a687810ee7f0622622d11956def5a0ca5c8b4c4b02718216d9488748945` |
| `upstream\data\camspec_dataset\Panasonic_DMC-LX3_Mauer-2009.csv` | `ceeeea4201e6ec747580c5e04baff18beb97e0a86e7fb909bcf0f58defe6ec53` |
| `upstream\data\camspec_dataset\Panasonic_DMCL-X5_Berra-et-al-2015.csv` | `3cbd683b32ce62e8990614d03546ae1d7cf41bb41a0d034dd914cb084c32f7ad` |
| `upstream\data\camspec_dataset\Pentax_K5_Jiang-et-al-2013.csv` | `388bd6b48360779ceba12274ea83eff7429b120f4d654d79db3dda8cf4e72ab5` |
| `upstream\data\camspec_dataset\Pentax_Q_Jiang-et-al-2013.csv` | `b734c24e406127d7c2c48f0acf8472d8dc35ef5967621c79756031aeb86d8918` |
| `upstream\data\camspec_dataset\Samsung_Galaxy-S20_Tominaga_etal_2021.csv` | `0af012d07f3eb5322ef78bad37d81cf220f45e09603d32ddeb7622ce85837830` |
| `upstream\data\camspec_dataset\Samsung_Galaxy-S7-Edge_Tominaga_etal_2021.csv` | `90c681c7fd4a9e5a9831f80deb9902afd3c67f3edd3184e4b4378a6f3bcae3a2` |
| `upstream\data\camspec_dataset\Samsung_Galaxy-S8_Burggraaff_etal_2019.csv` | `f4707ec69398aacd86ca904763573f6f26885ea683afec7ca3662d610f715bb9` |
| `upstream\data\camspec_dataset\Samsung_Galaxy-S9_Tominaga_etal_2021.csv` | `906967c317dc8e7c5c93c2802ccba0df89b62c62c7df2c3f4e864d24b6e45432` |
| `upstream\data\camspec_dataset\Sony_NEX5N_Jiang-et-al-2013.csv` | `d3f171aac201e6137ed0b531d3f8dadd0807359f78e3f951dfb7452e5f652dcc` |
| `upstream\data\camspec_dataset\Sony_NEX7_Berra-et-al-2015.csv` | `4691cb47196ffdaa8c77b2c03a820839fa44424ed314cd53b5c74a62ea344d9c` |
| `upstream\data\cam_calib\dji_pocket3.txt` | `3fe292fdb76fdc7af65f1ec78dad6d5c82b82c8297d59d686280a6bdc11458a0` |
| `upstream\data\cam_calib\eos_rp.txt` | `266ff28697ac8cfe332577cd6e387fb0695706b2c4538a7ed18161ceda64022e` |
| `upstream\data\cam_calib\iphone15promax.txt` | `8deb1dbb70e0cf381de9297b0a784a4f1eef01a1251201c9a3e29be447eac507` |
| `upstream\data\cam_calib\sony_alpha1.txt` | `c93dbd9708edcf76792c76c9127e6501f1a2b1806f5c48799023a3e53f95490c` |
| `upstream\data\light_calib\fluorescent.csv` | `1067d51a0a56f33753a3ee2de0e04728a7d2e928dff64308c9cad4be1757e981` |
| `upstream\data\light_calib\godox_3200.csv` | `91d0ef2264ce730d4caf1ed6c04ce9f793d560d934f9f47843ba445aa1eb074b` |
| `upstream\data\light_calib\godox_6500.csv` | `2cd58415b35a49da9eea8597274d33bc2cc59f596565abf71c05e531c74b8a60` |
| `upstream\data\light_calib\magsafe_3200.csv` | `8c9b1829a8040e8390573e5e17cfb65d1079dc1eb8e47f16bcf90e15f51f7860` |
| `upstream\data\light_calib\magsafe_6500.csv` | `5cd9a03df7a3115115c6079a410618a2a804842640bb3b48fd8b279598faa394` |
| `upstream\data\real\godox_dji_pocket3\godox_dji_pocket3_6500.yaml` | `a61de60116d75913824ffd353e8b2f3f3ac4c79a6f04d865e98af3440a78da55` |
| `upstream\data\real\godox_dji_pocket3\images\fluorescent_image.exr` | `d744655a6ce1dddbfe81c53809f45c90dc774cbe87bf5c04d26f82129c040442` |
| `upstream\data\real\godox_dji_pocket3\images\6500k\diffracted_image.exr` | `ef17daae04e0b5be3e15fb1f66fe429705b0a35275c7c024f0a2e4926df8197e` |
| `upstream\data\real\godox_dji_pocket3\images\6500k\direct_image.exr` | `aea08d3b99617d5d075e5d673cad17dc7f70a4ce0508fe0686e6e954d7bd5b54` |
| `upstream\data\real\godox_eos\godox_eos_6500.yaml` | `14348fee757a9319a38dff8d69f0a544fee570cc973720fa916ab6af4ed79b01` |
| `upstream\data\real\godox_eos\images\fluorescent_image.exr` | `41ba58d94923c9a1e5ada6eb7ed27fd6d59e969e809c7c72737996785f5af5e0` |
| `upstream\data\real\godox_eos\images\6500k\diffracted_image.exr` | `40f34a3abda96e1f5120f45f84c11c75ecc0513471adadf22e42092eefe02bba` |
| `upstream\data\real\godox_eos\images\6500k\direct_image.exr` | `d780905a310ae28119e810c6b3923a4d4705f46b323e27886946c2731d18fa1e` |
| `upstream\data\real\godox_iphone15promax\godox_iphone15promax_6500.yaml` | `7d26a8ec36e5ec7852b512e7d5ce5caae6e2673602407a90d1bcedeeb39e0a08` |
| `upstream\data\real\godox_iphone15promax\images\fluorescent_image.exr` | `ba13101fceaa722e37eb60c04327474a781f0501f98df50c36c2855b2b5ff812` |
| `upstream\data\real\godox_iphone15promax\images\6500k\diffracted_image.exr` | `1c0bc8b459712b2bbef392b4f2259b2fad579646a9c8e842dd0e48037caa57e8` |
| `upstream\data\real\godox_iphone15promax\images\6500k\direct_image.exr` | `c29355f268cbc6fb2109c571bdbfd3252273647dd55f69055fe13f1f817b60c5` |
| `upstream\data\real\godox_sony\godox_sony_6500.yaml` | `63c931276aac9414654da69016414db1d4451e0b65b8798f6cb808331e9c3fd7` |
| `upstream\data\real\godox_sony\images\fluorescent_image.exr` | `9bc53e6f44db41e79fae070ec8312997b00d8e32642ada9076a0296d00325d23` |
| `upstream\data\real\godox_sony\images\6500k\diffracted_image.exr` | `e86f2b014f738d37b21bd7e1d6b5263c1fec285695d6e5e6e81af6a2162f8278` |
| `upstream\data\real\godox_sony\images\6500k\direct_image.exr` | `0a46b34128e27a86cc6ff6475fbe3d9cf2a93a4303e36d74addb660d117e2093` |
| `upstream\src\estimate.py` | `180a7267734fa20461e402d75db615f9188d47d92785fc5964abb176709735c5` |
| `upstream\src\loader\camspec_loader.py` | `8ed9044b5c48f306af9a0d1ae4aa33818a19cae6754adab77231b9a1e6f0daeb` |
| `upstream\src\loader\diffraction_base_dataset.py` | `04deec0fd64b14d2daa7c2a491aaa1f8c8260cbb56fbed965be18e3ce62d5d51` |
| `upstream\src\loader\lightspec_loader.py` | `2edf455015c338fe70049f40270b3bd87b8628c9d7168a86fb5dcd609f3b7fd8` |
| `upstream\src\loader\realworld_dataset.py` | `d12fb7394811174580a388721961ccd9d7aada3c45772e8fe45ac59e983b0670` |
| `upstream\src\utils\camspec_util.py` | `d919b8a642b795db5cc36bf4b22af225d9e8677eef90573b77c6daf6f5b58c55` |
| `upstream\src\utils\io_utils.py` | `b335b03c47f117bbf3a0d6832e05bed0b448522a13ba84da3abec3cdba32ee66` |
| `upstream\src\utils\__init__.py` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |

## Final gate resolution

Earlier pending notebook, visual, public interface, semantic plans and inventory gates are now passed by parent full execution and independent material_mechanics read-only source review. All27tests,7active notebook cells,6computed figures and5math formulas passed; all six figures were individually inspected. Licensing remains an explicitly unverified public redistribution condition, not an invented grant. Local maintenance acceptance does not authorize publication.


## Independent parent acceptance

Canonical kernel metadata is portable python3; all executed cells, counts and scientific outputs are preserved from the private shared-runtime execution.

CLI, standalone Judge, pytest, active nbconvert and dual-engine math rendering were independently rerun: all exit0. Parent notebook: 7 executed code cells, 6 computed PNG outputs, 5 formulas. Private evidence: research_batch11_20261027/acceptance/grating_camera_spectral_response/execution.json and semantic_review.json. All panels actually inspected by parent; source/physics/GT roles and limitations semantically checked. Canonical notebook is this parent-executed artifact. No new source or notebook execution is inferred from cached upstream outputs.
