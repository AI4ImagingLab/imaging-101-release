# Implemented periodic charge-flipping mathematics

## Bound observations and physical scale

All4508 indexed real reflections of author dipyridyl disulfide are retained,
including136 negative integrated intensities. Before binding, divide both
the original measured Fo-squared values and their reported uncertainties by
the RMS of all original Fo-squared values. The original FCF global scale was
matched to a structural model; this normalization removes that unavailable
absolute gain. No calculated amplitudes, atomic coordinates, phases or direct
beam coefficient enter the public input or local initialization. The uncertainty
array is legitimate reported measurement information; the selected source
charge-flipping method does not weight its amplitude projection by it.

The actual author-identified symmetry is P21 unique-axis b, rather than the
preliminary P21/c teaching model. Cell lengths are15.8489,5.5008,23.1175 angstrom,
angles90,96.916,90 degrees. Let B have the three direct lattice vectors as
Cartesian columns, with a along x, b along positive y, c in x-z with positive z.
The reciprocal metric G=(B transpose B) inverse gives d(h)=1/sqrt(h transpose G h)
for Miller vector h. Explicit metric coefficients preserve source ordering
G11,G22,G33,2G12,2G13,2G23. No crystallographic library is used.

The source grid formula is round(cell lengths/dmin), add1, round upward to a
multiple of4, then double. With dmin=.5 angstrom, half-grid is(36,12,48), and
the complete output is(72,24,96). These are fractional lattice axes, excluding
the endpoint. They are not orthogonal Cartesian image axes. All165888 samples
are reconstructed and compared. Let N be their product and j the integer grid
vector. The physical acquisition convention is

```math
F_h=\frac{10}{N}\sum_j\rho_j\exp(2\pi i h\cdot(j_a/72,j_b/24,j_c/96)),
\qquad I_h=|F_h|^2+\eta_h.
```

Here rho is real relative electron-density contrast, F is the complex amplitude,
I the signed relative observation, eta measurement error and i the imaginary
unit. Both rho and F use square-root relative-intensity units. Unmeasured F000
stays zero: this is a contrast map, not electron count or a strictly positive
absolute density. Do not insert the FCF model electron count.

## Source-compatible initialization and symmetry

The two reciprocal rotations are identity and diag(-1,1,-1); paired fractional
translations are zero and(0,.5,0). The integer orbit helper explicitly handles
duplicates, opposite reflections, absences and Fortran MOD signs. In particular
0k0 with odd k is absent. Its multiplicity matches actual GENHKL, rather than
the misleading suggestion that the returned count includes doubled Friedel
partners.125 independently captured native GENHKL cases test these choices.

Initialize a centered complex128 grid to zero. For each observation in original
order with .5<d<100 angstrom, set amplitude sqrt(I) for I>0 and zero otherwise.
Draw one uniform phase in[0,360) degrees using Python Random seed20261022,
including rows with negative I. Write the rotated equivalent using phase plus
360 times h dot translation, then its opposite using complex conjugation.
Preserve source row/orbit overwrite order; phases are not independently redrawn
for partners. Missing amplitudes and F000 remain zero. Seed is a method
initialization parameter, never a hidden simulated-target seed.

## Explicit double-threshold iteration

Let E be the centered amplitude-constrained grid and A its fixed initial modulus.
At each step compute a real grid r=real(FFT(fftshift(E))). NumPy FFT has negative
exponent and no forward normalization; the initial factors are centered. The
shifted FFT real-grid convention matches the source on this even-sized grid.
Compute sigma=std(r), using population standard deviation over every grid node.
Set k=.1 and K=30. The real-space modification is

```math
\widetilde r_j=
\begin{cases}
-r_j,&r_j<k\sigma\ \mathrm{or}\ r_j>K\sigma,\\
r_j,&\mathrm{otherwise}.
\end{cases}
```

Then T=ifftshift(IFFT(modified r)), with the inverse FFT normalization1/N.
Replace exactly zero complex T by1 before dividing by its modulus, and update
E=A*T/abs(T). The source is a nonlinear dual-space feasibility iteration with
custom sign modification and amplitude replacement. No generic optimizer,
learned prior, imaginary artificial iteration or domain solver substitutes for
these executable updates. The2004 precursor lets F000 vary; the selected modern
GSAS method holds its initial zero and includes an upper threshold, as described
in2013. These distinctions are intentional source fidelity.

There is no proven monotonically decreasing objective. The source diagnostic
uses only A>0 grid entries. Define a=sum(A) and t=sum(abs(T)) on that mask, then
R= min(100,100*sum(abs(A/a-abs(T)/t))) in percent. This is not the full measured
intensity residual, and it omits zero-amplitude constraints by definition.
The benchmark uses200 complete updates, stopping earlier only when R<5 percent.
An optional stop_residual corresponds to the source GUI cancellation threshold;
the standard value0 disables it. The actual source200-step budget ends near
19 percent, so neither continuous convergence nor exact physical density is
claimed. Snapshots at1,2,7,25,100,200 and the complete R history remain identified.

## Original origin heuristic

After the last amplitude projection, reproduce findOffset: complex sort,
six-decimal string amplitude dictionary, reverse string sorting, restricted
low-order equivalent reflections and their wrapped phase differences. Select
up to the original count/term limits, use endpoint-inclusive linspace meshgrid
with source default axis ordering, evaluate the wrapped phase square sum, then
choose the negative integer argmin index as the roll. These unusual numerical
conventions are preserved, not presented as an ideal or certified symmetry
origin. Output is real(FFT(fftshift(E)))/10, rolled on all axes. The factor ten
is the fixed declared scale, with no fitted reconstruction gain.

## Backend qualification and numerical validation

Original unmodified NumPy1.26 and local code in that same backend are bitwise
identical at every captured state and the final200-step map. NumPy2.4 changes
roundoff and the nonlinear long trajectory; short-time agreement alone would
not establish full parity. Independently executed author NumPy2.4 math matches
the shared NumPy2.4 local trajectory bitwise throughout. That source run requires
one private postprocessing API repair: removed reshape keyword newshape becomes
the positional shape argument. The unmodified current-backend failure and the
qualified run are distinct evidence. The modern fixture was captured after
local extraction and is not mislabeled as a pre-extraction oracle. Original
old-backend stages, baseline and pre-extraction quality criteria remain intact.

Old-state one-step tests compare sigma, modification, temporary factors, phases
and residual. Same-backend modern tests compare the entire trajectory and final
map. Independent checks cover reciprocal geometry, a known Fourier point source,
Friedel inversion, screw absences, sign-flip norm and negative-observation floor.
Stage tolerance rtol5e-7/atol1e-9 was frozen before implementation. No final
quality boundary was relaxed to hide the backend trajectory difference.

## Evaluation, ambiguity and limits

The hidden reference is the independently executed unmodified NumPy1.26 method,
not truth. NCC is uncentered full-map cosine and NRMSE is full-map RMS difference
divided by the reference dynamic range. The data do not fix origin or global
spatial inversion. Judge searches both inversions using FFT coarse correlation,
the eight highest periodic local maxima plus zero as fractional-search starts,
and bounded L-BFGS-B with the analytic correlation gradient (ftol1e-15,
gtol1e-10,maxiter80). Each start is refined within one voxel on each axis.
This finite multistart search is a documented numerical alignment procedure,
not a proof of a global optimum. A sparse-spectrum analytic test caught failure
of single-start alignment; multistart corrects it without changing score limits.
Periodic real trigonometric interpolation defines fractional translations;
no gain, global sign, selected-voxel mask, crop or nonrigid fit is permitted.

The original pre-extraction repeated seeds/budgets determine NCC>=.9 and
NRMSE<=.014489186542210463. Both must pass along with all4508 signed intensity
relative L2<=.00011263614884958044 and abs(mean)/RMS<=1e-6. The signed intensity
formula includes negative noisy measurements; zeroing those errors is forbidden.
An initial random-phase map exactly fits positive moduli yet fails density
agreement, demonstrating why data consistency alone is insufficient. Original
seed spread and source budget behavior are limitations, not independent
experimental accuracy or absolute-density uncertainty.
