# Provenance and validation: complete Compton attitude sky intensity

This is a locally validated standard-method benchmark on author-existing simulation observations. The selected method is published ordinary fixed-background Poisson Richardson–Lucy, Siegert2020 Section4.1 Eq7, with step1. The earlier intended advanced variant did not complete and is not claimed as a delivered comparison. No actual sky truth asset is available; the hidden reference is an independently reproduced standard-method result.

## Source, paper, data rights and archive

Repository https://github.com/cositools/cosipy exact acquired commit **cf0cf6cfda77c50e4d737879474fafd9a0c14bdc**. This is an actually acquired revision, not a claim that this old revision is the current remote HEAD. Current-source acquisition record and ZIP SHA are below. Full unchanged source, paper PDFs, capture scripts and logs are outside this final task in the maintainer private directory `groups/physics_materials/cosi`, relative to research_batch8_20261022. `private_archive_index.json` contains file sizes and SHA256 for its original archives, logs, source capture scripts, source-only numerical evidence and papers. `final_source_math_integrity.json` independently checks all four original mathematical modules against the acquired ZIP. Their numerical source is byte identical. The final task contains no upstream checkout, domain-library package, PDF, source-capture script, screenshot or executed scratch notebook.

Code is Apache-2.0; original LICENSE is retained. Data are author-provided public challenge research observations: official tutorial assets under the author public reader paths below. A separate raw-data redistribution license has not been established. The software license is **not** applied to the data. This staging package is for local research and maintainer review; public redistribution requires a separate rights resolution. No acquired or invented hidden sky truth is exposed.

```json
{
  "url": "https://codeload.github.com/cositools/cosipy/zip/cf0cf6cfda77c50e4d737879474fafd9a0c14bdc",
  "commit": "cf0cf6cfda77c50e4d737879474fafd9a0c14bdc",
  "bytes": 77192005,
  "sha256": "b9a18c95b3ed7f9a0202ff81344c76654b41625e4374769e671a3db4bfd7fab2",
  "seconds": 0.6632366999983788,
  "status": "source_acquired_not_original_inverse_success",
  "source_root": "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research_batch8_20261022\\groups\\physics_materials\\cosi\\native_src",
  "previous_failure": "Windows long-path extraction in current_source; original ZIP preserved and extracted byte-identically to shorter native_src"
}
```
### Paper versions

```json
{
  "url": "https://arxiv.org/pdf/2005.10950v1",
  "version": "arXiv2005.10950v1 (2020-05-22), related journal DOI10.3847/1538-4357/ab9607",
  "title": "Imaging the511keV positron annihilation sky with COSI",
  "sha256": "b15af98c85ddb034831a6ff6be39538a3c4d7cf2ba56e8588628fc7f325ccf16",
  "bytes": 3539412,
  "seconds": 4.4699961000005715
}
```
```json
{
  "url": "https://arxiv.org/pdf/2412.04721v1",
  "version": "arXiv2412.04721v1 (2024-12-06), related journal ApJ979116 published2025-01-21 DOI10.3847/1538-4357/ad968a",
  "title": "Imaging and Spectral Fitting of Bright Gamma-ray Sources with the COSI Balloon Payload",
  "sha256": "06aa2fcb64bc28e49d7bbc52bde16976d9adb19576799790e8805f13961114a6",
  "bytes": 1762135,
  "seconds": 3.7261536000005435
}
```
The selected inverse is Siegert2020; Roberts arXiv2412.04721v1 supplies related response/imaging context, not the source of this satellite acquisition. Paper PDFs are archived privately and their exact hashes are above. The local plans define the actual equations and numerical conventions, so neither paper equation numbers nor citations substitute for executable mathematics.

### Original asset hashes

- `response.h5`: source `COSI-SMEX/cosipy_tutorials/Data/Responses/SMEXv12.511keV.HEALPixO4.binnedimaging.imagingresponse.h5`, SHA256 `fb458d0370552fc9407e44e8b36de50a582727fb65d7d131d7c5a7a9b38366c3`, 225031708 bytes.
- `ccm.hdf5`: source `COSI-SMEX/cosipy_tutorials/image_deconvolution/511keV/ScAttBinning/ccm_dc2.hdf5`, SHA256 `eaf12b585954c1864a2fb08e4c7e4085a6984333cef0347bf0de1de37b83f23f`, 30655124 bytes.
- `511keV_scatt_bkg.hdf5`: source `COSI-SMEX/cosipy_tutorials/image_deconvolution/511keV/ScAttBinning/511keV_dc2_scatt_bkg.hdf5`, SHA256 `b37c21b05f81578f37555cc5fc193de8b2e263ec90d33023295ad738e8e1b024`, 1280790464 bytes.
- `511keV_scatt_event.hdf5`: source `COSI-SMEX/cosipy_tutorials/image_deconvolution/511keV/ScAttBinning/511keV_dc2_scatt_event.hdf5`, SHA256 `6044b7f69e22f8b71d2252c09b61fbb839b1a8b2bfdc68b44208fa4b05d89c37`, 1280790464 bytes.

### Physical window, overflow and public packing

The bound acquisition is the entire author-selected physical histogram view, **278 x1 x60 x3072**, not all original under/overflow slots. Its observed total is649141 and known background422069; source227072. Original out-of-window counts are nonzero and remain archived. The original Histogram.open physical view and storage bridge are bitwise equal. No claim of zero overflow is made.

```json
[
  {
    "source": "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research_batch8_20261022\\groups\\physics_materials\\cosi\\dc2_scatt\\511keV_scatt_event.hdf5",
    "target": "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research_batch8_20261022\\groups\\physics_materials\\cosi\\dc2_scatt\\511keV_scatt_event_physical.hdf5",
    "physical_shape": [
      278,
      1,
      60,
      3072
    ],
    "dtype": "float64",
    "physical_nonzero_count": 617504,
    "physical_count_sum": 649141.0,
    "physical_squared_sum": 728949.0,
    "maximum_count": 7.0,
    "overflow_nonzero_count": 30334488,
    "overflow_sum": 50522564.0,
    "physical_bytes_sha256": "c1824bc462345b7d448bc8d0a87a17a17675784d7bc8bc2f567720a6700d7f00",
    "roundtrip_bitwise": true,
    "seconds": 2.519100699981209,
    "no_inverse_executed": true
  },
  {
    "source": "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research_batch8_20261022\\groups\\physics_materials\\cosi\\dc2_scatt\\511keV_scatt_bkg.hdf5",
    "target": "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research_batch8_20261022\\groups\\physics_materials\\cosi\\dc2_scatt\\511keV_scatt_bkg_physical.hdf5",
    "physical_shape": [
      278,
      1,
      60,
      3072
    ],
    "dtype": "float64",
    "physical_nonzero_count": 399515,
    "physical_count_sum": 422069.0,
    "physical_squared_sum": 482591.0,
    "maximum_count": 7.0,
    "overflow_nonzero_count": 29327544,
    "overflow_sum": 48407926.0,
    "physical_bytes_sha256": "3cb80f68f5ee83e8a724291615afec6e6c4f3133f0d5bf0ceaf9634b135b33ba",
    "roundtrip_bitwise": true,
    "seconds": 3.584459700010484,
    "no_inverse_executed": true
  }
]
```
All nonzero attitude-conversion coordinates are physical; subtracting one converts original underflow-offset indices. Public integers preserve exact counts/tallies; area and dwell calibration explicitly select the upstream-supported float32 API variant before inverse extraction. Public exposure is target-independent A^T1; no reconstructed field, hidden seed/object, target-derived calibration or true sky occurs in raw_data.npz. No newly synthetic observation is generated, so src/generate_data.py is correctly omitted.

Public NPZ file SHA256: `a3f2a224396cb9af698ec96d6297b2fd1e1b58c690091e2a947d1c1a8474892b`.

```json
{
  "seconds": 14.110919646002003,
  "keys": [
    {
      "key": "events",
      "shape": [
        1,
        278,
        1,
        60,
        3072
      ],
      "dtype": "uint8",
      "numeric_bytes_sha256": "72a220c335b8a2bcd59c4af2541c440afc972c9fd0fc2c5034003f3d4b22fe11"
    },
    {
      "key": "background",
      "shape": [
        1,
        278,
        1,
        60,
        3072
      ],
      "dtype": "uint8",
      "numeric_bytes_sha256": "fb6caf0e197354f5ff2380ba2a60ff46ee9141d31826369c4e6093d2437f9252"
    },
    {
      "key": "response_counts",
      "shape": [
        1,
        3072,
        1,
        1,
        60,
        3072
      ],
      "dtype": "uint16",
      "numeric_bytes_sha256": "c93d44def1768b517b8299755eea1abe8a77098fafe0cef4ea58acab9d3db3e4"
    },
    {
      "key": "response_area_cm2",
      "shape": [
        1,
        1
      ],
      "dtype": "float32",
      "numeric_bytes_sha256": "cff63e8fc4a9ccf5162d5548d02535f1866e871c6d0671f9ffdf36b0ef607396"
    },
    {
      "key": "attitude_coordinates",
      "shape": [
        1,
        3,
        3416064
      ],
      "dtype": "int32",
      "numeric_bytes_sha256": "4bb845374c18e9165c8dd71615fa3cf64e08214696b538fb2ef35fb15e8bacad"
    },
    {
      "key": "attitude_weights_s",
      "shape": [
        1,
        3416064
      ],
      "dtype": "float32",
      "numeric_bytes_sha256": "624a31d98de4b3e57f5702c44169b1cb878d90a6cb1e2b5bc6f8a00f9a1239fe"
    },
    {
      "key": "exposure_cm2_s_sr",
      "shape": [
        1,
        3072
      ],
      "dtype": "float32",
      "numeric_bytes_sha256": "c3ed4301d11ed938905a8164c27a07e2d015c137019eb163f55c857536fe72c6"
    }
  ],
  "scope": "Public counts and target-independent instrument calibration only; native physical-bin observations exactly preserved; float32 calibration variant explicitly selected before inverse extraction",
  "npz_bytes": 248864755
}
```
## Traceability: equations, source functions, local implementation and tests

| Required stage | Unchanged source function / paper role | Local function | Independent validation consumer |
|---|---|---|---|
| Physical count extraction and coordinate convention | Histogram.open physical contents; CoordsysConversionMatrix physical sparse view | preprocessing.load_observations | test_original_physical_coordinate_view; native_histogram_view_audit.json |
| Mean counts A x + b; response in cm2, C in s, solid angle | data_interfaces/dataIF_COSI_DC2.py calc_source_expectation / calc_bkg_expectation | ComptonOperator.forward | test_original_full_operators; native_full_operator_fixtures.npz; independent explicit toy sum |
| Ordinary transpose and exposure A^T1 | dataIF_COSI_DC2.py calc_T_product / _calc_exposure_map | ComptonOperator.adjoint / compute_exposure | test_original_full_operators; test_adjoint_and_exposure; native_full_operator_checks.json |
| Siegert2020 Sec4.1 Eq7 delta, fixed background, additive float32 update | algorithms/RichardsonLucy.py Mstep; RichardsonLucyBasic constraints | solvers.reconstruct | test_original_first_inverse_update; first delta fixture; all30 local source-parity measurements |
| Total Poisson log likelihood and stopping | dataIF_COSI_DC2.calc_log_likelihood; RichardsonLucyAdvanced.check_stopping_criteria | poisson_log_likelihood / reconstruct | full original30 objective history; active notebook curves; complete CLI test |
| Ring order / physical labels | equal-area HEALPix ring discretization | ring_pixel_angles | test_equal_area_ring_geometry; acquisition metadata contract |
| Output, metric and failure contract | benchmark-defined, independent of source solver | evaluation/judge.py evaluate | test_judge.py; independently known repeated three-value metric case shared with visualization |

## Original execution and configuration audit

Both full30 original-source runs used the same exact observations, float32 public API, background normalization one, flat1e-4 intensity, all optional flags false, minimum intensity zero, stopping increment.01, budget30. There is no randomized solver initialization or solver RNG seed. The observations are pre-existing author simulation; no hidden generation seed is inferred or provided. The source scripts use original mathematical functions and explicit external cyclic garbage collection only to collect unreachable Histogram cycles; first/7update fixture parity and the full independent process repeat verify this storage intervention.

The initial capture passed a plain dict where a configuration wrapper was needed, so advanced flags silently defaulted false. The misleading historical scope strings in original raw logs are preserved as evidence, **superseded by this audit**. Correcting Configurator reached advanced expectation deepcopy and exceeded the resource budget. A generic ndarray memmap copy adapter did not intercept Histpy._replace/_contents.copy; the failed advanced runs/adapters are archived. They are not passed gates or plotted advanced results. The explicit basic-method rebinding and frozen tolerances preceded local inverse extraction: see prelocal_freeze.json.

`run_original_float32_gc_execution.json` (the stored scope phrase “Source loading audit only” is an old supervisor label; per-update archives prove the complete30step result):

```json
{
  "command": [
    "wsl",
    "-d",
    "Ubuntu-24.04",
    "--",
    "env",
    "OMP_NUM_THREADS=1",
    "OPENBLAS_NUM_THREADS=1",
    "MKL_NUM_THREADS=1",
    "/mnt/d/yan1/agent/AutoSkill/imaging-101-maintainer/research_batch8_20261022/groups/physics_materials/cosi/oracle_env/bin/python",
    "/mnt/d/yan1/agent/AutoSkill/imaging-101-maintainer/research_batch8_20261022/groups/physics_materials/cosi/run_original_float32_gc.py"
  ],
  "returncode": 0,
  "seconds": 195.38964740000665,
  "resource_abort": false,
  "minimum_physical_GiB": 7.1678466796875,
  "minimum_commit_GiB": 20.80260467529297,
  "scope": "Source loading audit only; no inverse success claim"
}
```
`run_original_float32_gc_repeat_execution.json` (the stored scope phrase “Source loading audit only” is an old supervisor label; per-update archives prove the complete30step result):

```json
{
  "command": [
    "wsl",
    "-d",
    "Ubuntu-24.04",
    "--",
    "env",
    "OMP_NUM_THREADS=1",
    "OPENBLAS_NUM_THREADS=1",
    "MKL_NUM_THREADS=1",
    "/mnt/d/yan1/agent/AutoSkill/imaging-101-maintainer/research_batch8_20261022/groups/physics_materials/cosi/oracle_env/bin/python",
    "/mnt/d/yan1/agent/AutoSkill/imaging-101-maintainer/research_batch8_20261022/groups/physics_materials/cosi/run_original_float32_gc_repeat.py"
  ],
  "returncode": 0,
  "seconds": 196.6800712000113,
  "resource_abort": false,
  "minimum_physical_GiB": 7.092842102050781,
  "minimum_commit_GiB": 21.245121002197266,
  "scope": "Source loading audit only; no inverse success claim"
}
```
Full numerical source capture `float32_gc_complete.json` records164.855s inverse execution; independent repeat complete file records its actual time. These elapsed values exclude/ include loading according to their respective fields; do not conflate them. Original process exits are both0; full repeat maps/model/delta/exposure are bitwise identical. Logs with warnings are indexed; failed prior runs remain identified rather than overwritten.

```json
{
  "array_checks": {
    "maps": true,
    "model": true,
    "delta": true,
    "exposure": true
  },
  "all30_model_arrays_bitwise": true,
  "scope": "Independent fresh original full30process restart same observations, float32API, outerGC, no local code"
}
```
### Original environment

WSL Ubuntu24.04, CPU, OMP/OPENBLAS/MKL threads1. Isolated oracle_env only; upstream dependencies are absent from cleaned runtime. Environment queried on restore from the exact oracle executable, not inferred from a package file.

```json
{
  "python": "3.12.3 (main, Mar 23 2026, 19:04:32) [GCC 13.3.0]",
  "platform": "Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.39",
  "packages": {
    "cosipy": "0.4.2.dev0",
    "histpy": "2.0.7",
    "numpy": "2.5.4",
    "scipy": "1.18.1",
    "astropy": "8.0.2",
    "healpy": "1.20.1"
  },
  "blas": {
    "Compilers": {
      "c": {
        "name": "gcc",
        "linker": "ld.bfd",
        "version": "14.2.1",
        "commands": "cc"
      },
      "cython": {
        "name": "cython",
        "linker": "cython",
        "version": "3.3.0",
        "commands": "cython"
      },
      "c++": {
        "name": "gcc",
        "linker": "ld.bfd",
        "version": "14.2.1",
        "commands": "c++"
      }
    },
    "Machine Information": {
      "host": {
        "cpu": "x86_64",
        "family": "x86_64",
        "endian": "little",
        "system": "linux"
      },
      "build": {
        "cpu": "x86_64",
        "family": "x86_64",
        "endian": "little",
        "system": "linux"
      }
    },
    "Build Dependencies": {
      "blas": {
        "name": "scipy-openblas",
        "found": true,
        "version": "0.3.34.106.0",
        "detection method": "pkgconfig",
        "include directory": "/opt/_internal/cpython-3.12.13/lib/python3.12/site-packages/scipy_openblas64/include",
        "lib directory": "/opt/_internal/cpython-3.12.13/lib/python3.12/site-packages/scipy_openblas64/lib",
        "openblas configuration": "OpenBLAS 0.3.34.106.0  USE64BITINT DYNAMIC_ARCH NO_AFFINITY SkylakeX MAX_THREADS=64",
        "pc file directory": "/project/.openblas"
      },
      "lapack": {
        "name": "scipy-openblas",
        "found": true,
        "version": "0.3.34.106.0",
        "detection method": "pkgconfig",
        "include directory": "/opt/_internal/cpython-3.12.13/lib/python3.12/site-packages/scipy_openblas64/include",
        "lib directory": "/opt/_internal/cpython-3.12.13/lib/python3.12/site-packages/scipy_openblas64/lib",
        "openblas configuration": "OpenBLAS 0.3.34.106.0  USE64BITINT DYNAMIC_ARCH NO_AFFINITY SkylakeX MAX_THREADS=64",
        "pc file directory": "/project/.openblas"
      }
    },
    "Python Information": {
      "path": "/tmp/build-env-ptu49_tp/bin/python",
      "version": "3.12"
    },
    "SIMD Extensions": {
      "baseline": [
        "X86_V2"
      ],
      "found": [
        "X86_V3"
      ],
      "not found": [
        "X86_V4",
        "AVX512_ICL",
        "AVX512_SPR"
      ]
    }
  }
}
```
### Cleaned environment

Windows CPU, shared prepared research runtime, numerical-library thread count1. requirements.txt is the exact actually used runtime subset; pytest/Jupyter remain development tools. No package was installed or upgraded for this cleaning.

```json
{
  "executable": "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research-runtime\\Scripts\\python.exe",
  "python": "3.11.11 (main, Mar 17 2025, 21:01:29) [MSC v.1943 64 bit (AMD64)]",
  "platform": "Windows-10-10.0.26200-SP0",
  "packages": {
    "numpy": "2.4.6",
    "scipy": "1.17.1",
    "matplotlib": "3.11.1",
    "pytest": "9.1.1",
    "nbconvert": "7.17.1",
    "ipykernel": "7.3.0"
  },
  "blas": {
    "Compilers": {
      "c": {
        "name": "msvc",
        "linker": "link",
        "version": "19.44.35226",
        "commands": "cl"
      },
      "cython": {
        "name": "cython",
        "linker": "cython",
        "version": "3.2.4",
        "commands": "cython"
      },
      "c++": {
        "name": "msvc",
        "linker": "link",
        "version": "19.44.35226",
        "commands": "cl"
      }
    },
    "Machine Information": {
      "host": {
        "cpu": "x86_64",
        "family": "x86_64",
        "endian": "little",
        "system": "windows"
      },
      "build": {
        "cpu": "x86_64",
        "family": "x86_64",
        "endian": "little",
        "system": "windows"
      }
    },
    "Build Dependencies": {
      "blas": {
        "name": "scipy-openblas",
        "found": true,
        "version": "0.3.31.188.0",
        "detection method": "pkgconfig",
        "include directory": "C:/Users/runneradmin/AppData/Local/Temp/cibw-run-6ukx7exl/cp311-win_amd64/build/venv/Lib/site-packages/scipy_openblas64/include",
        "lib directory": "C:/Users/runneradmin/AppData/Local/Temp/cibw-run-6ukx7exl/cp311-win_amd64/build/venv/Lib/site-packages/scipy_openblas64/lib",
        "openblas configuration": "OpenBLAS 0.3.31.188.0  USE64BITINT DYNAMIC_ARCH NO_AFFINITY SkylakeX MAX_THREADS=24",
        "pc file directory": "D:/a/numpy-release/numpy-release/.openblas"
      },
      "lapack": {
        "name": "scipy-openblas",
        "found": true,
        "version": "0.3.31.188.0",
        "detection method": "pkgconfig",
        "include directory": "C:/Users/runneradmin/AppData/Local/Temp/cibw-run-6ukx7exl/cp311-win_amd64/build/venv/Lib/site-packages/scipy_openblas64/include",
        "lib directory": "C:/Users/runneradmin/AppData/Local/Temp/cibw-run-6ukx7exl/cp311-win_amd64/build/venv/Lib/site-packages/scipy_openblas64/lib",
        "openblas configuration": "OpenBLAS 0.3.31.188.0  USE64BITINT DYNAMIC_ARCH NO_AFFINITY SkylakeX MAX_THREADS=24",
        "pc file directory": "D:/a/numpy-release/numpy-release/.openblas"
      }
    },
    "Python Information": {
      "path": "C:\\Users\\runneradmin\\AppData\\Local\\Temp\\build-env-jxn7bo0g\\Scripts\\python.exe",
      "version": "3.11"
    },
    "SIMD Extensions": {
      "baseline": [
        "X86_V2"
      ],
      "found": [
        "X86_V3"
      ]
    }
  }
}
```
## Frozen acceptance and mathematical checks

Before local inverse implementation, full-map NRMSE<=1e-5, cosineNCC>=.999999, stage relativeRMS<=1e-5, adjoint and dwell relativeerror<=1e-5, source closure<=.001 were frozen. Independent full original repeat is bitwise, but a nonzero roughly84float32epsilon tolerance permits backend/contraction roundoff; self-comparison zero is not used as an error boundary. No threshold changed after seeing local results.

```json
{
  "adjoint_relative_error": 1.029062746679617e-07,
  "adjoint_left": 10097.223353457612,
  "adjoint_right": 10097.222314389972,
  "ccm_max_relative_duration_variation": 5.6339452332249287e-08,
  "integrated_exposure_s_min": 7979954.967153102,
  "integrated_exposure_s_max": 7979955.030147314,
  "source_expectation_max": 0.5624901056289673,
  "response_unit": "cm2",
  "ccm_unit": "s",
  "model_unit": "1 / (s sr cm2)",
  "source_unit": "",
  "seconds": 16.029076339000312,
  "scope": "Unchanged original full operator/transpose and independent inner-product plus attitude-duration conservation checks; no local inverse extraction"
}
```
```json
{
  "source_relative_l2": 7.787700313483552e-08,
  "transpose_relative_l2": 6.710165361458184e-08,
  "exposure_relative_l2": 6.587306943646371e-08,
  "seconds": 26.32047639999655
}
```
Independent full original adjoint inner-product error1.029e-7 and max relative dwell variation5.634e-8 pass. This is independent mathematics, not merely two implementations of a copied formula. Total dwell is7979954.967..7979955.030seconds per sky pixel. The full local forward/adjoint/exposure relativeerrors are recorded above; integer response tiling retains the whole reduction dimension. Local full30 solver parity records every update; maximum per-update dynamic-rangeNRMSE is below2.25e-8; finalNCC.9999999999999276 and finaldelta relativeL2=5.539185733975091e-6.

```json
{
  "returncode": 0,
  "resource_abort": false,
  "seconds": 524.8676083000028,
  "minimum_physical_GiB": 7.396503448486328,
  "minimum_commit_GiB": 21.449588775634766,
  "peak_child_rss_GiB": 1.6940574645996094
}
```
### Physics and scientific limits

```json
{
  "metrics": {
    "source_count_observed": 227072.0,
    "source_count_predicted": 227165.68374839812,
    "unmodeled_source_counts": 0.0,
    "negative_predicted_bins": 0,
    "model_flux_cm2_s": 0.0025570614047381616,
    "attitude_pearson_per_bin": 2.4744787624243325,
    "attitude_covered_count_fraction": 0.9994010710259301,
    "scatter_angle_pearson_per_bin": 1.3205717507626218,
    "scatter_angle_covered_count_fraction": 1.0,
    "scattered_direction_pearson_per_bin": 1.0594844059468336,
    "scattered_direction_covered_count_fraction": 0.9980975197294251,
    "poisson_source_deviance": 1797355.059038517,
    "maximum_intensity_lon_lat_deg": [
      0.0,
      -2.388015463268786
    ]
  },
  "seconds": 8.946928119999939,
  "noise_convention": "Original shared additive albedo realization is treated as exactly known; signal=event-background. Not independent background-template noise.",
  "scope": "Actual original full-forward on complete selected observations; scientific acceptance pending"
}
```
```json
{
  "observed_deviance": 1797355.059038517,
  "fixed_model_expected_deviance": 1817830.6329512394,
  "fixed_model_standard_deviation": 3198.0288064586207,
  "standardized_difference": -6.40256081226366,
  "maximum_predicted_mean": 0.5624901056289673,
  "positive_and_zero_bins_included": 51240960,
  "poisson_series_maximum_count": 20,
  "note": "Means were fitted to these data, so this fixed-model comparison is descriptive, not an independent calibrated p-value. No51-million-dof Gaussian chi-square approximation.",
  "seconds": 23.38475849997485
}
```
No negative predicted bins or unmodeled source counts were found. Fixed background is exactly the injected realization, not independent noisy background data. Count-scale closure is meaningful, but per-bin exact agreement is not: finite response Monte Carlo and averaged-attitude approximation cause descriptive attitude overdispersion. Sparse-Poisson deviance includes all51240960 bins; its fixed-fitted-mean comparison is not a calibrated p-value. Reference agreement proves method reproduction, not true-sky accuracy. Poor maps are rejected without relaxing gates: zero/flat/.95scale/permutation.

## Fixtures and reference measurements

| File | Source and documented consumer |
|---|---|
| test/fixtures/native_ccm_physical_view.npz | source-only native physical sparse coordinate capture; test_original_physical_coordinate_view |
| test/fixtures/native_full_operator_fixtures.npz | unchanged source full forward/transpose, captured before local code; test_original_full_operators |
| test/fixtures/float32_first_iteration.npz | unchanged source first model/delta/exposure; test_original_first_inverse_update |
| notebooks/reference_outputs/upstream_runs.npz | original complete30 maps/reconstruction/likelihood, used by active notebook original/fresh panels and iteration comparisons |
| notebooks/reference_outputs/fresh_review_measurements.npz | active notebook fresh30 maps/reconstruction/objectives/runtime; notebook execution audit and reproducibility record, never loaded as a substitute for active reconstruction |
| evaluation/baseline_reference.npz | unchanged original complete30 model, no true sky claim; independent Judge reference |

All fixtures are frozen numeric NPZ, allow_pickle=False. Source capture scripts remain private; normal tests do not import oracle packages. All retained fixture/reference files have these consumers. Full per-file hashes are in the final inventory.

## CLI, Judge, tests and notebook evidence

Standard command `python -m src.main` was executed by test_standard_full_cli_and_independent_judge with default30updates. It freshly wrote output/reconstruction.npz; independent CLI Judge passed. First default pytest attempt:6passed22setup errors caused by Windows default temporary-directory permissions; private release_pytest_default_tmp_failure.log retained. The unchanged tests were rerun with explicit private --basetemp.

Final full pre-contract-extension suite: ======================= 28 passed in 312.12s (0:05:12) ========================

Additional independent public metadata and nontrivial Judge/visualization metric tests: ============================== 2 passed in 0.73s ==============================. At this stage only the physics_model docstring and these extra tests changed after the full28suite; no mathematics/runtime behavior changed. Total30 tests passed across those original checks; subsequent parent and leaf checks are recorded below.

```json
{
  "returncode": 0,
  "seconds": 258.66227610001806
}
```
```json
{
  "status": "pass",
  "passed": true,
  "metrics": {
    "ncc": 0.9999999999999276,
    "nrmse": 1.7742092871240987e-08,
    "source_count_relative_error": 0.00041252269820016914
  },
  "thresholds": {
    "ncc": {
      "comparison": ">=",
      "boundary": 0.999999
    },
    "nrmse": {
      "comparison": "<=",
      "boundary": 1e-05
    },
    "source_count_relative_error": {
      "comparison": "<=",
      "boundary": 0.001
    }
  },
  "errors": []
}
```
Judge unit tests reject missing file/key, bad shape, integer/complex/float16 dtype, nonfinite/negative values, missing evaluator assets, invalid metric configuration, poor images and report-write failure; foreign CWD and truth precedence pass. Independently known repeated vectors[1,2,3] and[2,1,4] yield cosine16/sqrt(294) and dynamic-rangeNRMSE.5 in both visualization and Judge.

Notebook execution uses shared-runtime private kernel imaging101-research and `%matplotlib inline`; frontend jupyter-nbconvert was discovered through the existing Anaconda PATH, but the actual scientific kernel executable is shared research-runtime/Scripts/python.exe (process checked). No solver library changes the backend.

```json
{
  "command": [
    "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research-runtime\\Scripts\\python.exe",
    "-m",
    "jupyter",
    "nbconvert",
    "--to",
    "notebook",
    "--execute",
    "notebooks/compton_attitude_sky_intensity.ipynb",
    "--output",
    "compton_attitude_sky_intensity_executed.ipynb",
    "--output-dir",
    "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research_batch8_20261022\\groups\\physics_materials\\cosi",
    "--ExecutePreprocessor.timeout=3600",
    "--ExecutePreprocessor.kernel_name=imaging101-research"
  ],
  "returncode": 0,
  "resource_abort": false,
  "seconds": 234.3550984999747,
  "minimum_physical_GiB": 10.796318054199219,
  "minimum_commit_GiB": 26.140933990478516,
  "peak_child_rss_GiB": 1.8773574829101562,
  "kernel_executable": "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research-runtime\\Scripts\\python.exe"
}
```
Notebook has six chapters and six actual scientific figures: observations/calibration, original-vs-fresh updates1/7/30, signed image parity plus scatter, fresh forward projections and residuals, objective/flux/finite-iteration convergence, and measured negative controls. No required supplied upstream scientific comparison is dropped: the source active notebook supplied a30step response-based imaging path; optional advanced runs were never completed and are disclosed, not invented. `final_figures/` is private and contains PNG/SVG/PDF plus exact NPZ source arrays/JSON per figure. The canonical notebook contains the genuinely executed cells and six inline scientific outputs; its active code recomputes all30 updates when executed. The original successful execution artifact is additionally preserved privately. Timeout3600seconds reflects measured multi-minute solve.

## Semantic review, dependency audit and completion gates

Reviewed local source against approach.md and design.md: units of C,R,Omega and output; one-based physical-coordinate removal; floor excluded from transpose; e=A^T1; background normalization; total objective sign/no normalization; additive delta/step1/nonnegativity; stopping sign; float32 core and float64 evaluator; complete dot-product dimensions; exact signatures/defaults/return fields; caller paths; no notebook cache substitution. Library modules never select a Matplotlib backend. Runtime imports only standard library, NumPy, SciPy, Matplotlib (review plots). Source, Judge, tests and notebook were inspected for actual symbol/call roles: sparse COO/CSR, matrix product, norm/sum/log and plotting are general primitives; no COSIpy/Histpy/Healpy/domain reconstruction, cross-task import, opaque solver, privileged truth input or checkpoint inverse exists. The source spherical-smoothing docstring was stale and corrected; no spherical smoothing implementation is shipped.

| Gate | Status |
|---|---|
| Paper/revision/data hashes/traceability | passed |
| Original complete selected method on bound observations | passed |
| Source independent repeat; frozen parity/physics checks | passed |
| Fresh full local30 updates and default CLI/Judge | passed |
| Judge malformed/error/negative controls | passed |
| Public observation/calibration + README acquisition/output contract | passed |
| No truth/target calibration/hidden seed/GitHub public URL leak | passed |
| Minimal shared CPU environment and no domain-library shortcut | passed |
| 28full tests +2independent contract tests | passed |
| Active six-chapter notebook native execution and visual inspection | passed |
| Original/reproduced/fresh provenance and meaningful metrics | passed |
| README/plans actual semantic/signature consistency | passed |
| Final inventory/no scratch/source notice preservation | passed |
| Independent parent acceptance | unverified; parent gate |
| Public raw-data redistribution rights | unverified; local research only |

## Parent-discovered leaf corrections, 2026-10-24

The parent independently ran all30 then-existing tests, including a complete fresh inverse pipeline, exit0 in240.7568618s. Its later notebook audit correctly failed: the previous worker handoff retained a successful executed artifact privately but left the unexecuted input at the canonical path. This was a packaging omission, not a later generator overwrite. The parent failure logs remain unchanged under qa/compton_attitude_sky_intensity-parent-audit.json. Before copying, every cell type/source was compared against the genuine native0 executed artifact; all were identical. All6code cells executed with6inlinePNG outputs and zeroerrors. Exported iteration/parity/convergence source arrays were checked exactly against active measurements and current reconstruction. The actual executed artifact was copied byte-for-byte to the canonical path. No inverse, forward or notebook computation was repeated. The canonical SHA and checks are below.

```json
{
  "issue": "Successful executed artifact was retained privately while unexecuted input notebook was shipped as canonical; packaging handoff omission, not a subsequent generator overwrite. Parent audit correctly rejected current canonical.",
  "old_canonical_sha256": "e874e40dc15293663251862dc7ac8913228ce2d2a78e0a89b3f580f73ad45f97",
  "executed_artifact_sha256": "bf27e13a3bf6a6dbfbcb703483e6fafc0b805846bc813982c6c58a713b53526b",
  "canonical_sha256": "bf27e13a3bf6a6dbfbcb703483e6fafc0b805846bc813982c6c58a713b53526b",
  "cell_sources_identical_before_copy": true,
  "executed_code_cells": 6,
  "inline_pngs": 6,
  "execution_errors": 0,
  "numeric_figure_sources_match_active_measurements": true,
  "output_matches_active_measurements": true,
  "correction": "Copied actual native0 executed artifact to canonical after validation; no RL/forward/notebook recomputation",
  "original_execution_seconds": 234.3550984999747
}
```
Parent complete30suite evidence:

```json
{
  "command": [
    "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research-runtime\\Scripts\\python.exe",
    "-m",
    "pytest",
    "test/",
    "-v",
    "--basetemp",
    "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research_batch8_20261022\\qa\\compton_attitude_sky_intensity_pytest_temp"
  ],
  "cwd": "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research_batch8_20261022\\staging\\tasks\\compton_attitude_sky_intensity",
  "exit_code": 0,
  "seconds": 240.75686180003686
}
```
Two additional real regressions were added before changing the Judge: an existing ground_truth.npz directory previously silently selected the baseline (code0 instead of2), and a finite1e308 prediction produced nonfinite derived metrics (code1 instead of structured error2). `judge_leaf_red.log` records both failures on the old code. The minimal repair falls back only when truth neither exists nor is a symlink; an existing non-file truth therefore reaches strict loading and errors. Metric arithmetic catches floating overflow/invalid operations and explicitly rejects nonfinite norms or metrics before publishing measured values. Normal arithmetic, metric formulas, thresholds, observations, inverse and figures remain unchanged. The huge-input regression also runs the actual CLI and verifies exit2, UTF-8 JSON, statuserror, passedfalse, empty metrics and descriptive errors with allow_nan=False.

Affected Judge/public-contract collection after repair: ============================= 25 passed in 3.20s ==============================. Its25tests comprise23previous cases plus the2newcases. Therefore distinct validated tests are **32 = parent full30 +2new regressions**, not55 or a claim that all32 ran the inverse again. Red/green exit files and logs are private and preserved.

The earlier validation placeholders arose from reading PowerShell UTF-16 logs as UTF-8; decoding now detects the BOM and the actual28+2 results are rendered above. This corrects the evidence presentation without inventing an additional test run. Parent acceptance of these final changes remains a separate gate.


## Parent independent private acceptance,2026-10-24

The complete parent30-case suite exited0 in 240.756861800s, including fresh full30-update CLI. Following two independently reproduced evaluator edge failures, the parent ran all25affected Judge/public-contract tests again: exit0/3.455515400s. This is a32-distinct-case union, not a claim of a single full32-case run; no numerical reconstruction was repeated for the leaf fixes. Parent foreign-working-directory Judge passed, strict inventory/public-input audit found no errors, and all88formulas passed both KaTeX and MathJax. Parent individually opened all6final PNG groups (acquisition, original/fresh iterations, full-field signed parity, forward consistency, objective/flux/iteration sensitivity and negative controls) and read the complete physics/preprocessing modules, both plans, README and provenance. Independent peer semantic review SHA256a7f72eea232ff52849aabd4ee2306685faa4428e2425afc708d5fb989e59629a supports the source/basic-method interpretation and full operator conventions. Canonical notebook SHA256bf27e13a3bf6a6dbfbcb703483e6fafc0b805846bc813982c6c58a713b53526b is byte-identical to the actual successful234.3550985s executed artifact, with6executed code cells/6embedded figures/no errors. The earlier missing copy-back was an artifact-handoff error, not a failed scientific run; its unexecuted version and audit are retained privately. Both leaf fixes leave methods, observations, figures and numeric outputs unchanged. Truth-directory and finite-input-overflow failures now produce evaluator error2/empty metrics/valid JSON. Source method is ordinary fixed-background RL on author-existing simulated DC2, not the failed advanced variant or real measured sky. No independent sky truth or calibrated per-bin goodness-of-fit claim is made. Data redistribution rights remain unresolved separately from Apache software licensing. Accepted privately pending the complete20-task release.


## Public-contract scientific isolation 2026-10-29

The current strict evaluation identifies a30-update method output rather than a method-independent Poisson estimand. Benchmark eligible=false. Scientific review is completed; blind-agent execution is a separate unexecuted gate and is not the reason for isolation. Removed README recipe, where present, is archived privately under benchmark_readiness_20261029/private_readme_recipes. Existing arrays/Judge thresholds unchanged.


## Current benchmark contract v2 (2026-10-29)

The current v2 benchmark evaluates a nonnegative measurement-consistent conditional sky, not unique spatial truth. All 51,240,960 bins enter independently evaluated float64 factorized Poisson deviance; the fixed attainable boundary is 2.729057677375829 (measured original-field value 2.7290566773758287 plus predeclared 1e-6). Original source-count closure <=0.001 is unchanged. No response/noise/prior is invented. Response tallies actually have uint16 dtype and maximum1158, so conversion to uint8 would be lossy. The evaluator streams the exact integer array into a read-only temporary memmap and converts only 2048-column float64 tiles, enforcing at least6GiB available-host headroom. It imports no solver and never reads a private forward cache. Unsupported positive-count/zero-mean bins produce a finite reporting penalty and quality fail1, not NaN/Inf.

The original30-RL algorithm, raw observations, stored output and method baseline are unchanged. Historical pixel NCC/NRMSE and30-step comparisons remain explicitly labeled v1 reproduction diagnostics, not current pass gates. A scalar gain profile was essentially stationary (improvement only6e-14); it is not treated as meaningful new inversion evidence. An independent one-dimensional contrast direction with complete-forward line search lowers normalized deviance by4.455e-5 while preserving original source counts; an actual standalone full-data recomputation accepts it. Flux-matched wrong/flat maps are rejected by deviance despite exact count closure. This is a conditional improvement control, not a fresh blind reconstruction.

Full commands, failed memory-guard diagnostic, current tests, full-bin controls, independent scalar contractions and final hashes are recorded in the external benchmark_readiness_20261029/compton_v2 archive. Historical full active30-update notebook evidence remains valid for algorithm reproduction; only current evaluation and explicitly historical diagnostic cells are re-executed for this semantic revision.
