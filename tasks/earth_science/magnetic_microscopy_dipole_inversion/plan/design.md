# Shipped interfaces and flow

All scientific arrays are float64 and finite. Public arrays are batch-first only on disk; internal arrays omit the singleton batch. The CLI assumes the task directory as working directory. No generator exists because measured data are reused.

| Callable | Inputs / defaults | Returns / side effects |
| --- | --- | --- |
| preprocessing.load_data(path='data/raw_data.npz') | NPZ path | Bz (600,960) nT, x (960,) µm, y (600,) µm; validates field geometry/finite values; reads only input file. |
| preprocessing.upward(field,spacing=2.35,displacement=5.0) | 2-D field nT, equal pixel spacing µm and signed vertical displacement µm | Continued field same shape/dtype nT; no disk writes. |
| preprocessing.prepare(field,x,y) | Field nT, coordinates µm | Tuple continued f,fx,fy,fz,TGA,stretched TGA; all shape (600,960); gradient units nT/µm; stretched dimensionless. |
| preprocessing.detect_windows(stretched,x,y) | 2-D dimensionless image, coordinate axes µm | (100,4) xmin,xmax,ymin,ymax windows µm; intensity-ordered. |
| physics_model.dipole_jacobian(x,y,z,position) | matching 2-D x,y µm, scalar or matching z µm, position (3,) µm | (N,3) coefficient rows T/(A m²), rejects coincident source/sensor. |
| physics_model.forward(x,y,z,positions,moments) | x,y shape H×W, scalar/matching z, positions (K,3) µm, moments (K,3) A m² | H×W Bz nT, summed sources. |
| solvers.euler(field,dx,dy,dz,x,y,z=10.0) | Local field nT and gradients nT/µm, local axes µm, sensor z µm | Position (3,) µm and background scalar nT. |
| solvers.invert_moment(field,x,y,position,z=10.0) | Background-subtracted local field nT, axes/position µm | Moment (3,) A m², covariance (3,3) A² m⁴, R2, SNR dB. |
| solvers.reconstruct(field,x,y) | Complete bound acquisition | Positions (75,3), moments (75,3), bases (75,), scores (75,2) R2/SNR, accepted windows (75,4), covariance (75,3,3), prepare tuple, all detection windows (100,4). |
| visualization.angles(moments) | K×3 signed vector moments | Inclination/declination (K,) degrees. |
| visualization.review(field,x,y,result,reference) | Acquisition, fresh reconstruct tuple, upstream NPZ mapping with batch-first positions/moments | Two Matplotlib figures; no backend selection, no file writes. |
| main.main() | No arguments; reads default public path | Writes output/reconstruction.npz with positions/moments each (1,75,3), creates output/ if needed. |

`python -m src.main` → load_data → reconstruct → prepare/detect_windows → each window's euler and invert_moment → discard z>0 → NPZ. CLI writes only physical required arrays; diagnostics remain available to notebook callers. Importing src.main does not select a plotting backend.

The active notebook calls the same load_data and reconstruct functions on the public observations, times the fresh run, saves a fresh reconstruction, loads notebooks/reference_outputs/upstream.npz only for explicit comparison, calls independent Judge evaluation, renders the acquisition and interpretation comparisons, and tabulates upstream versus fresh quality counts. Its all-source field predictions call the local forward model. No upstream package is imported.

The standalone evaluation/judge.py imports only stdlib/NumPy/SciPy, resolves metrics and hidden references beside its own file, and evaluates every required output array after exact float64 shape and finite validation. CLI paths are caller-relative. It returns exit 0/1/2 for pass/quality-fail/error and emits strict finite UTF-8 JSON. Hidden ground_truth takes precedence if present; this task has only baseline_reference. Judge geometric matching is the only alignment and has no effect on physical scale or direction. Unit tests import the Judge by file path; comparing tests consume the independent frozen upstream arrays and analytic physical identities.
