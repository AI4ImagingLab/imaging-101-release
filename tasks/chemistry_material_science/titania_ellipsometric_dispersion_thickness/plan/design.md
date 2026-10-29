# Shipped interfaces and execution flow

The task runs independently with NumPy2.4.6, SciPy1.17.1 and Matplotlib3.11.1 in the validated shared CPU environment. There is no domain optical solver, lmfit, Pandas, database download, widget or upstream dependency in the cleaned runtime. No generated-data script is needed because existing real measurements are reused.

## Standard entry point

`python -m src.main` parses optional `--data` and `--output`. `main(argv=None)` sets OMP/BLAS thread counts to1before numerical imports. `run(data_path=None,output_path=None,initial=None)` uses task-relative default paths data/raw_data.npz and output/reconstruction.npz, calls load_observations then reconstruct, creates the output parent and writes both required arrays in oneNPZ, returning(output,record). Explicit paths are caller-relative. Importing src.main has no plotting/backend side effect; no library calls matplotlib.use().

## Observation loading

`load_observations(path=None)` returns a dictionary with wavelength_nm,psi_deg,delta_deg as float64(926,) and silicon_index as complex128(926,). It loads allow_pickle=False, validates exact public batch shapes/dtypes/finiteness, increasing400–800nm wavelengths, physicalPsi and passiveSi calibration, then removes only the declared batchaxis. It reads no hidden references or external files. `measured_rho(data)` returns complex128(926,) using tan(Psi)*exp(-iDelta) with degreeconversion.

## Local optics

`cauchy_index(wavelength,parameters)` accepts wavelength values in nm and a4vector(n0,a,b,d), returning the real dimensionless nfield with explicit source100/1e7scaling. d does not affect this dispersion equation.

`fresnel(n_i,n_t,theta_i,theta_t)` accepts broadcastable complexindices/radianangles and returns(rs,rp,ts,tp). `forward_angle(index,angle)` returns the passive branch boolean. `reflection(wavelength,parameters,silicon_index,angle_deg=70.,oxide_thickness=276.36)` validates parameter/calibration shapes/finiteness and returns complex128(rs,rp), each wavelength-shaped1Darray. It explicitly constructs source root/Snell branches, interfaces, two finite-layer phases and2x2matrix products. Ambient and substrate are half-spaces. `predict_rho(data,parameters,angle_deg=70.)` calls reflection and returns rp/rs; it never alters data or performs inversion.

## Inverse

`initial_parameters()` returns float64(4,) authorstart. LOWER/UPPER are the exact4bounds. `to_internal(parameters)` validates the finite4vector withinbounds and applies the arcsine transform. `from_internal(internal)` applies the corresponding scalar sine transforms and returns float64(4,). `residual(parameters,data)` returns float64(1852,), all real then all imaginary observed-minus-modeled complexratio values.

`reconstruct(data,initial=None)` calls boundtransformed MINPACK leastsq with the exact approach.md defaults, including10002raw callbacks corresponding to the source10000adjusted nfev budget. Return output has `parameters` float64(1,4), mixed declared coordinates, and `reconstruction` float64(1,926), dimensionless n. record contains nfev(int MINPACKreport), success(bool),message,status(int),residual(float641852),seconds(walltime),evaluations(float64Ncallsx5 with n0,a,b,d,SSE),covariance(float644x4orNone),initial(float644). Evaluation traces include finite-difference and setup calls; they are not iterations. No solver writes files. Optional initializations are used only by explicit notebook sensitivity/comparisons, not the standardCLI.

## Independent evaluation

The standalone evaluation/judge.py is owned separately from the solver and imports no submitted solver code. It loads exact required output/reference arrays, computes each mixed-unit parameter metric separately, nfielddynamicrangeNRMSE and formula consistency, then evaluates whole-band complexrho using a separate recursive-interface implementation and publicSi calibration. Reference paths are relative to its own file, CLIpaths to the caller. Actual truth would take priority; shipped baseline_reference.npz is a validated method estimate. Every declared boundary in metrics.jsonmust pass, with strict finiteJSON and descriptive errors for invalid arrays or evaluator assets. CLIreturns0pass,1validbelowthreshold,2malformed/evaluator/reportfailure. Unit tests cover metric/Judge cases independently of the reconstruction implementation.

## Review notebook and figures

The canonical notebook actively calls cleaned main.run on bound public observations and grades the fresh output. All retained source arrays in notebooks/reference_outputs have identified consumers and provenance in test/validation.md. The notebook compares fullobservations, initialmodel, original2x2and4x4sourcefits, and freshlocalfit; it computes all additional starts/condition/sensitivity plots it labels fresh. It uses an explicit sharedPython private kernelspec and inline Matplotlib. Its runner captures actual nativeexit/stdout/stderr/runtime, and the final canonical file retains executed scientificoutputs.

`src/visualization.py` handles display, forward sensitivity and diagnostic finite differences; it never performs inversion or mutates the source data. Scientific figures are exported outside the task to a private caller-specified directory as PNG/SVG/PDF plus actual numeric NPZ and JSON provenance. The notebook's six code cells have zero-based cell indices 1,3,5,7,9,11. Its local `show` helper calls the exporter and displays/closes each figure. The following are the shipped interfaces; all inputs called data use the dictionary described above, all spectral arrays have926samples, and parameters are the four declared coordinates.

| Callable signature | Inputs, returns and units | Notebook caller / effects |
|---|---|---|
| `observed_rho(data)` | Complex128(926,) measured ratio from degree-valued angles. | Cells1,9; no writes. |
| `rho_angles(rho)` | Complex spectrum to tuple(Psi,Delta), real degree arrays; Delta in[-180,180). | Drawing helpers; no writes. |
| `comparison_metrics(data,rho)` | Dict of relative complex-rho RMS and Psi/circular-Delta RMS in degrees. | Cells5,11 and helpers; no writes. |
| `export_figure(fig,destination,arrays,metadata)` | Matplotlib figure, path stem, named numeric arrays, JSON-compatible provenance; returnsNone. | Cell1's show helper creates private parents and writes matching PNG180dpi/SVG/PDF/NPZ/JSON. |
| `acquisition_figure(data)` | Returns(figure,arrays): wavelengths nm, measured degree angles and complexrho. | Cell1, all926observations. |
| `spectrum_figure(data,fresh_rho,source,initial_rho,source4x4)` | Original dictionaries contain actual2x2/4x4fits; returns(figure,arrays), all complexratio/degree spectra and four source4x4parameters. | Cell3; actual initial forward, both original fits and freshfit, no inverse execution. |
| `residual_figure(data,fresh_rho,source_rho)` | Returns(figure,arrays,metrics), signed Psi/complex residuals and circular degree-phase residuals. | Cell5; no writes until exporter. |
| `scaled_jacobian(data,parameters,angle_deg=70.)` | Float64(1852,4), centered forward differences with step1e-5*max(abs(parameter),1), columns multiplied by the same scale. | properties helper; descriptive condition diagnostic only, not optimizer Jacobian. |
| `properties_figure(data,parameters,source_parameters,source_covariance,source_singular_values)` | Covariance4x4, singular-values4; returns(figure,arrays) with dimensionless index/correlation and scaled sensitivity spectrum. | Cell7; covariance is archived conditional source covariance, not a new truth estimate. |
| `multistart_figure(data,starts,solutions,labels,objectives,evaluations)` | Starts/solutions(M,4), labelsM, SSE/nfev(M,); returns(figure,arrays). | Cell9, four actually executed fits; displays spectra, thicknessnm, SSE and function-evaluation counts. |
| `sensitivity_figure(data,parameters,angles_deg,source_rho)` | Angle vectordegrees and sourcecomplexrho; returns(figure,arrays) with held-parameter complex responses, reflectances and angle-dependent relative residual. | Cell11; evaluates the forward model at both70/70.06and requested angles, no refit. |

No extra task-level report or test directory is created.
