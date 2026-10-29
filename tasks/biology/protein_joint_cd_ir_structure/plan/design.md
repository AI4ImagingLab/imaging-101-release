# Runtime design

`python -m src.main` loads public NPZ, selects/scales spectral blocks, runs complete local SELCON3 for four targets, then writes both required arrays to output/reconstruction.npz. No upstream import, file download, cross-task import, hidden calibration or domain library is used. The reference task uses NumPy and Matplotlib; development pytest/nbconvert remain outside requirements. Importing main does not select a plotting backend.

| Callable | Signature/defaults | Shape/units/return |
| --- | --- | --- |
| preprocessing.load_data | path='data/raw_data.npz' | Validated batch-first float32dict observations(1,4,267),library_spectra(1,24,267),library_fractions(1,24,4) |
| preprocessing.prepare | data,ir_scale=15.,mode='joint' | A(M,24),F(4,24),q(M,4),float32; M127joint,66CD-only,201IR-only |
| physics_model.spectral_svd | A | U,S,Vh; full singular vectors, float64rectangular S |
| physics_model.structure_estimate | F,q,decomposition,rank | Four estimated fractions; explicit truncated inverse |
| physics_model.spectral_refit | decomposition,rank | First column of rank-truncated spectral matrix, shape(M,) |
| physics_model.rmsd | a,b | Float RMS difference in inputs' units |
| solvers.select_candidates | A,F,q | Lists of fractions and zero-based(lastcolumn,basis) pairs after full variable-subspace selection |
| solvers.helix_selection | solutions,hj_helix | Integer indexes into SELCON2candidate list |
| solvers.selcon | A,F,q,ir_only=False | Diagnosticdict fractions,spectrum,spread,ensemble_spectra,hj5,ix,sel1,sel2,sel3,Mean_refit_prot,refit_np_param,refit_rule,attempts,fallback |
| solvers.reconstruct | data,ir_scale=15.,mode='joint' | (submissiondict,4diagnosticdicts); default outputs float64 reconstruction(1,4,4),spectra(1,4,127); optional CD-only spectra(1,4,66),IR-only(1,4,201) are notebook comparisons, not accepted submissions |
| visualization.rms_error | a,b | Float RMS difference |
| visualization.fractions_plot | groups,title | Figure of4protein grouped bars; no file output |
| main.main | noarguments | Submissiondict, writes output/reconstruction.npz |
| judge.load_arrays | path | Strict exactshape/float64/finite loader for both keys |
| judge.metrics | prediction,reference |5explicit scalar metrics |
| judge.evaluate | prediction_path | (report,exitcode), hidden assets relative to Judge |
| judge.main | argv=None | CLI/reportwrite; exit0pass,1qualityfail,2error |

Notebook actively computes default and all28scientific comparisons using local modules. It loads original outputs and withheld crystal annotations only from notebooks/reference_outputs/upstream.npz and explicitly labels them. It writes fresh cleaned_measurements.npz. Unit/parity tests use frozen numeric fixtures and original comparison results, no oracle execution. Original source, capture scripts, logs and source measurement files remain in the external private maintainer archive. There is no synthetic generator, because the benchmark observations are measured and reused.
