# Shipped execution and interfaces

preprocessing.load_data(path="data/raw_data.npz") returns a dictionary from nominal angle to (detuning, observed S0). Each is a complete float64 vector; condition order is 0,45,90 and detuning is in MHz. Matching shapes, dtypes and finite values are validated.

physics_model.py implements the atomic constants, angular momentum matrices, Kronecker Hamiltonians, thermal populations, dipole overlaps, Voigt profiles and transverse dielectric propagation. susceptibility(detuning,temperature,magnetic_field) takes MHz, Celsius and Gauss, and returns complex128(3,N) in Left/Right/Z order. transmission(detuning,temperature,magnetic_field,ex,ey,length=.001) returns normalized real float64(N). It imports no original or domain library.

solvers.from_internal(values) maps three internal coordinates to theta/Celsius/Gauss with the original sine bounds. fit_sweep(detuning,observations,nominal_angle,seed=20261029) runs full DE and returns a length-3 vector [B tesla,T kelvin,theta rad], the complete modeled spectrum, and convergence/evaluation/generation/runtime/RMS records. The nominal angle is a condition label, not a fixed physical parameter. Objective, bounds, budget and stopping conventions are in approach.md. The inverse never accesses a reference.

main.run(output="output/reconstruction.npz") fits all three complete observations, writes all six output arrays and returns (outputs,records). python -m src.main is the standard CLI and prints the records. The notebook calls run afresh, evaluates the result, then computes complete forward perturbation experiments. Imports do not select a Matplotlib backend. visualization.save_review_figure(figure,path) is an optional export helper; the notebook directly saves its computed figures at 150 dpi.

The standalone Judge imports no solver. It loads hidden assets beside itself; an actual ground_truth.npz with the public output contract takes precedence if supplied. Otherwise each reference member is a complete paired parameter/response outcome. For each condition, all parameter and response tolerances must pass against the same member, together with independent observation RMS. No alignment is fitted. Exact float64 shapes, finite values and physical support are validated; all metrics and positive boundaries must be finite. Invalid input/assets/overflow return empty-metric error JSON and exit2; valid poor predictions exit1, passes exit0. Report-write failure emits stderr and exits2.

Tests consume frozen original atomic/Voigt stages and DE population fixtures, plus independent angular momentum, hyperfine, normalization and dielectric-mode checks. Nine original parameter-to-response cases are recomputed on every native frequency. Complete inverse execution is additionally demonstrated by separate actual CLI and notebook runs after freezing. Unit tests do not repeat the 15-minute optimization after those two full executions and do not replace them with a smaller benchmark.

Actual local physical signatures:

| Callable | Arguments/defaults |
|---|---|
| raising_operator | jj |
| jx | jj |
| jy | jj |
| jz | jj |
| sz | L, S, I |
| lz | L, S, I |
| Iz | L, S, I |
| Hfs | L, S, I |
| Hhfs | L, S, I |
| Bbhfs | L, S, I |
| numDenRb | T |
| FreqStren | groundLevels, excitedLevels, groundDim, excitedDim, Dline, hand, BoltzmannFactor=True, T=293.16 |
| add_voigt | d, DoppTemp, atomMass, wavenumber, gamma, voigtwidth, ltransno, lenergy, lstrength, rtransno, renergy, rstrength, ztransno, zenergy, zstrength |
| Hamiltonian.__init__ | self, Isotope, Trans='D2', gL=1.0, Bfield=4200.0 |
| Hamiltonian.groundStateManifold | self, gI, I, A_hyp_coeff, IsotopeShift, Bfield |
| Hamiltonian.excitedStateManifold | self, gL, gI, I, A_hyp_coeff, B_hyp_coeff, Bfield |
| susceptibility | detuning, temperature, magnetic_field |
| transmission | detuning, temperature, magnetic_field, ex, ey, length=0.001 |
