# Shipped interfaces and execution

All modules are local NumPy/Matplotlib runtime code; no qMRLab import or archived script executes in the task. Standard command `python -m src.main` loads raw data, reconstructs, creates output if needed, and writes one NPZ. Imports do not select a plotting backend.

| Function | Arguments and defaults | Return and role |
| --- | --- | --- |
| preprocessing.load_data | path='data/raw_data.npz' | Dict PDw/T1w/MTw float64(128,128,96), scanner units; validates shape, finite values and dtype after removing batch axis |
| preprocessing.analysis_mask | data dict | bool(128,128,96), fixed observation-only region; pure function |
| physics_model.flash_signal | amplitude,r1,delta,angle,tr; broadcast numeric arrays | Approximate signal in scanner units; delta fraction, angle radians, tr seconds |
| physics_model.exact_flash_signal | amplitude,r1,angle,tr | Independent no-MT Bloch signal for physics comparison |
| physics_model.predict_from_parameters | data,mtsat,r1,mask | Tuple amplitude(N,),predicted(3,N) in scanner units; order PD/T1/MT; maskN voxels |
| solvers.reconstruct | data,repetition_times=None | Dict reconstruction,r1 float64(1,128,128,96); zero outside domain; percent and s^-1; optional three repetition times only for labeled acquisition-error experiments |
| solvers.mtr | data,mask | float64(128,128,96) percent intensity ratio, not saturation |
| visualization.comparison_metrics | a,b,mask; unbatched arrays | Dict rmse,ncc using selected entries |
| visualization.image_grid | images,titles,limits=None,cmap='viridis',z=48 | Matplotlib Figure with spatial maps and colorbars; no file side effect |
| main.main | no arguments | Reconstruction dict; writes output/reconstruction.npz |
| judge.load_arrays | path | Strict batch-first float64 dict for both required outputs; rejects malformed input |
| judge.metrics | prediction,reference | Independent RMSE/cosine/zero/observation metrics using public observations |
| judge.evaluate | prediction_path | (report,exitcode); hidden assets relative to judge, user paths relative to cwd |
| judge.main | argv=None | CLI parse/write UTF8 report; code0pass/1qualityfail/2error |

No generator exists because observations are reused. Default constants live in physics_model; geometric/conditioning constants live in preprocessing and are repeated independently in Judge. README concordance tests guard them. Judge uses only NumPy and standard library, never imports solver or submitted code. Actual truth would take precedence if present; this task supplies baseline_reference only.

Notebook sections Problem/Data/Physics/Method/Results/Conclusion actively load raw inputs and execute reconstruct, then write fresh cleaned_measurements in notebooks/reference_outputs. It loads upstream.npz only for explicitly labeled comparisons. It performs fresh MTR, wrong-TR, Bloch-approximation and synthetic-noise sensitivity experiments. No cached value is presented as a fresh result. Tests consume test/fixtures/upstream.npz and no original packages. All source/download/oracle logs and conversion scripts remain outside the task in the maintainer archive.
