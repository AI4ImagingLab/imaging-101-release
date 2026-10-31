# Implemented mathematics

The complete benchmark reconstructs apo ecSecB from 63 peptides at seven native exposures. The forward model, raw control correction and units are fully defined in README. All source arrays retain their precision. X is 63 by 146, k is a146-vector of rates in1/s, t is seven seconds, G is a146-vector in J/mol; a is 63 by 7 Da. R=8.31446261815324 J/(mol K), T=303.15 K and f=.9. Protection exp(G/RT) enters the denominator 1+protection. MSE is normalized by 441, with no factor 1/2. TV is 2e-4 times the mean of 145 absolute adjacent differences; no support restriction or periodic wrap is applied.

## Initialization

`initial_energy(raw,temperature=303.15)` forms relative fractional uptake u/F, applies exchange-masked membership divided by peptide capacity, then normalizes each residue's weights across peptides. Boundaries are all changes of membership between neighboring residues. Each nonempty block is fit to 1-r exp(-k1 t)-(1-r)exp(-k2 t). Initial r=.5. k1 is an fsolve root at native time index2 starting .5; k2 is an fsolve root at penultimate exposure starting .05. Powell minimizes half the sum of seven squared fractional residuals, tolerance1e-9. Original symbolic lexical parameter ordering is preserved explicitly. Out-of-domain fits, nonfinite parameters or twice the fit loss above .2 trigger original differential evolution: rand1bin, population multiplier40, mutation(.423,1.053), recombination.95, Latin-hypercube initialization, no polish, private reproducibility stream43. Admissible mixture is[0,1]; rate bounds are ln2/(50 max positive t) and50ln2/(min positive t). Defaults of the installed general optimizer are retained and recorded in the source.

The mixture's half-time is found by fsolve in log time, starting at log(k1) if r>.5 else log(k2), preserving the original convention despite its dimensional starting-guess oddity. Rate=ln2/half-time. G=RT log(max(kint/rate-1,0)); nonfinite energies are interpolated over residue indices from finite neighbors. This initialization uses only public observations and independent calibration, never a frozen initial field.

## Joint inverse updates and stopping

Float64 PyTorch computes the explicit physical forward model and full objective. SGD uses learning rate10000, momentum.5, dampening0 and Nesterov correction. If g is the current objective gradient, b0=g on the first step and thereafter b=.5 b+g; update G <- G - 10000*(g + .5*b). Thus the first step is -15000g. There is no projection or line search. After each step the pre-step MSE and TV are appended. Let d be previous pre-step objective minus current; a counter increases if d<1e-5, otherwise resets. Stop when counter>100 or after200000 epochs. Final quantities are recomputed after the last update. Native optimizer nondifferentiable absolute-value convention uses zero subgradient for exact ties.

## Versioned scientific correction

Original PyHDX0.4.0b3 multiplies the fully exchanged capacity by f=.9, but its energy forward asymptote is the unlabelled site count. Independent t -> infinity and capacity tests identify the missing factor, not an auxiliary-mask issue. V1 is retained as a source counterfactual, not the selected physical standard. V2 multiplies the forward uptake by f while leaving raw data, control, initial field, 146-site adjacency and raw TV coefficient unchanged.

The independent V2 source run calls its original forward and SGD kernels through a unit adapter: native y=yraw/f, native r1=2/f^2, native learning rate10000*f^2, native stopping tolerance1e-5/f^2. Since original TV=1e-4 r1 mean|difference|, Jraw=f^2*Jnative and grad Jraw=f^2*grad Jnative. Momentum and Nesterov updates coincide with the local physical objective, and the absolute progress rule scales identically. All other budgets are unchanged. This is an explicitly corrected protocol, not an unmodified paper reproduction.

## Evaluation

No independently measured residue truth is available. Original corrected-source result is a method reference. Acceptance uses J <= 1.0361689959301181, fixed as1.01 times the independently measured V2 standard1.025909896960513 before the local V2 run. All146 entries contribute to TV, all441 predictions to fidelity. Reported uptake must agree with the independent NumPy forward calculation with norm(error)/norm(corrected observations) <= 1e-9. Similarity to a particular nuisance field is not required. If actual truth is later supplied, it takes precedence and requires an explicit versioned truth metric configuration; no silent fallback occurs.
