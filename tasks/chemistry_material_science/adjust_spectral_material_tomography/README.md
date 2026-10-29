# Joint Spectral Material Tomography

> Reconstruct four spatial material abundance maps and their attenuation spectra from 600 real spectral fan-beam projections with strongly overlapping material signatures and an unknown specimen-specific matrix spectrum.
> Domain: chemistry & material science | Keywords: spectral tomography, joint reconstruction, material decomposition | Difficulty: Hard

## Background

X-ray attenuation depends on photon energy and chemical composition. Measuring transmitted photons in many energy bins therefore supplies more information than a conventional grayscale radiograph. Dense materials with similar attenuation at one energy can have distinguishable absorption edges or energy dependence across a spectrum. Spectral tomography combines this contrast with rotation to recover spatial composition inside an intact specimen.

This modality is useful for locating valuable or hazardous mineral grains, studying heterogeneous manufactured materials, and separating constituents whose structures overlap in ordinary tomography. Photon-counting detectors provide energy discrimination but distribute a finite photon budget across many channels. Detector response, scatter, imperfect spectral references, and the conditioning of the projection geometry make quantitative chemical interpretation difficult.

Recovering a spectrum independently in every reconstructed pixel is expensive and sensitive to noise. A small number of shared material spectra is a useful physical model, but those spectra and the material maps must be inferred consistently. A matrix constituent whose spectrum is not independently measured introduces an additional calibration challenge.

## Problem Description

The experiment measures a slice containing gold, lead, tungsten, and a quartz matrix. No material map or independently measured specimen truth is supplied. Four latent components are unordered; their row order has no intrinsic physical meaning.

Let the measured count at acquired angle index a, energy index c, and detector position d be I(a,c,d), the flat-field count I0(c,d), and the dark count D(c,d). All are photon-count units. The dimensionless optical-depth observation is

```math
Y(j,c)=\log\frac{I_0(c,d)-D(c,d)}{I(a,c,d)-D(c,d)}.
```

Here j indexes the ray formed by a and d after reversing the acquired angle sequence. The supplied angle array describes that reversed sequence's reconstruction geometry. Ray order is angle first within each detector position. Natural logarithms are used; no count clipping is needed for these observations.

For narrow energy bins, the material model is

```math
Y \simeq W A F, \qquad F=R T.
```

W is the known fan-beam line-integration operator: entry W(j,i) is the length of ray j inside image pixel i, measured in image-pixel lengths. A(i,k) is the dimensionless abundance of latent component k in pixel i. F(k,c) is optical depth per image-pixel length for that component and energy channel. R(k,p) is its nonnegative mixture weight on dictionary atom p, and T(p,c) is the atom's spectral response. Four maps on a 255 by 255 image are required. The three supplied elemental spectra, ordered tungsten, gold, lead, are scaled by the known nominal areal-mass factor 0.015 g/cm^2. The fourth, quartz-matrix response is computed from the same observations using the explicitly defined basis protocol below. It is not independently known instrument calibration. This four-material dictionary is the selected published experimental configuration.

The selected reference-release convention constrains each pixel's abundance sum to at most one. Its dictionary-mixture rows sum to one and dictionary-mixture columns sum to at most one. These are basis normalization constraints; abundance should not be interpreted as independently validated absolute mass density. The paper's weaker row-inequality convention is discussed in References and Method Hints. The spectral objective uses all 65 channels with zero-based indices 29 through 93, while all 128 channels remain available for observation-derived spectral estimation.

The inverse problem is nonconvex because both the spatial maps and spectral mixtures are unknown. Material attenuation curves overlap, the common matrix contributes broadly across the image, and real detector and spectrum mismatch prevents an exact noiseless fit. No independently validated physical ground truth is available; evaluation measures agreement with a validated standard-method result and checks the declared physical constraints.

Save **`output/reconstruction.npz`**, relative to the task workspace, with all three arrays:

| Output key | Shape | Dtype | Units and convention |
|---|---|---|---|
| `reconstruction` | `(1, 4, 255, 255)` | float64 | Real dimensionless abundance; axes batch, component, row, column; nonnegative with component sum at each pixel <= 1 |
| `spectra` | `(1, 4, 65)` | float64 | Real optical depth per image-pixel length; axes batch, component, increasing retained energy channel; common basis scale fixed above |
| `dictionary_weights` | `(1, 4, 4)` | float64 | Real dimensionless mixtures; axes batch, component, dictionary atom ordered tungsten, gold, lead, observation-derived quartz; nonnegative, row sums one, column sums <= 1 |

Use the full field of view with image-pixel center coordinates x=column-127 and y=127-row. The object-plane pixel pitch is 0.80/(883.081116/57.999001) mm. Outside the square image support the attenuation is zero. No crop, intensity rescaling, image translation, phase conversion, or spatial registration is applied during evaluation.

The evaluator resolves only the unavoidable component permutation: it minimizes total normalized spectral distance to the reference components and applies that same permutation to all three arrays. Every output quantity participates in the coupled model: the maps A and spectra F must explain all600views and65retained energy channels, and F must equal R T for the fixed observation-derived dictionary T defined below. The normalized dictionary discrepancy ||F-RT||F/||T||F must be at most1e-6. All abundance and mixture feasibility conditions above remain mandatory. Instead of comparing each component to a particular nonconvex factorization, evaluation requires the complete squared optical-depth residual to be no greater than the validated standard attainable objective times(1+1e-8). This is an optimization-quality target, not known material truth; a better feasible factorization passes, in any joint component ordering.

The independent12-view observation-consistency check also remains: ||Wv A F-Yv||F/||Yv||F<=0.17007280335223202, where Wv includes every ray at views0,50,...,550 and all65energy bins. There is no fitted gain or offset. The full objective uses the same corrected observations, reversed measurement-view convention and stated float32 ray discretization for all600views. The abundance-sum and dictionary-column violations must be at most1e-10, negativity at most1e-10, and row-sum discrepancy at most2e-6. These are numerical feasibility tolerances, not additional measurement noise.


## Data Description

The observations are actual photon-counting fan-beam measurements, not rendered material maps or synthetic projections. All 600 projection frames, all 128 energy channels, all 257 detector positions, and the measured dark/flat fields are retained. The three supplied NIST spectra define the published four-material experiment together with an actively estimated quartz response. Acquisition TIFFs were converted losslessly to NPZ; count values remain uint16. The conversion introduces no binning, spatial cropping, synthetic corruption, or downcasting of observations.

Every visible NPZ key is listed here:

| Key | Shape | Dtype | Description |
|---|---|---|---|
| `counts` | `(1, 600, 128, 257)` | uint16 | Measured photons; axes batch, acquired angle, energy channel, detector position; unit photon counts |
| `dark` | `(1, 128, 257)` | uint16 | Measured dark-field photon-count offset; axes batch, energy channel, detector position |
| `flat` | `(1, 128, 257)` | uint16 | Measured open-beam photon counts; axes batch, energy channel, detector position |
| `dictionary` | `(1, 3, 128)` | float64 | NIST-based elemental mass-attenuation samples; axes batch, atom, channel; unit cm^2/g; multiply by the declared areal-mass factor before use |
| `angles` | `(1, 600)` | float64 | Reversed-observation reconstruction angles, radians; axes batch, angle; inclusive endpoints 0 and 2*pi |
| `energies` | `(1, 128)` | float64 | Measured energy-bin midpoints, keV; axes batch, channel; each bin has width 1.10 keV |

The public dictionary rows are ordered tungsten (atomic number74), gold (79), lead (82). Their tabulation agrees with linear interpolation of the corresponding elemental NIST coefficients at the actual detector-bin midpoints, 20.90 to160.60keV. The limits20.35and161.15keV denote bin edges, not a different sampling grid.

**Output-basis definition.** The fourth atom is defined operationally from these observations, so its mixture coordinates are reproducible. For all128channels, define a fixed data-dependent basis functional X(100) by X(0)=0 and the following normalized linear recurrence for n=0,...,99:

```math
X^{n+1}=X^n+C W^{\mathsf T}L(Y-WX^n).
```

Here X is the pixel-by-energy image matrix, n is the iteration index, L is diagonal with entries equal to the reciprocal ray-weight sums of W, and C contains reciprocal pixel-weight sums. Y contains all128corrected optical-depth observations. A reciprocal is zero when the absolute corresponding weight sum is <=1e-6. Use relaxation one, float32 arithmetic, no positivity clipping, and the fan-beam line discretization defined above. The fourth spectrum is the float32 mean over zero-based image rows 129 through 139 and columns 129 through 139 in each channel. Append that spectrum to 0.015 times the three public spectra, then retain channels 29 through 93. This fixed observation-only protocol defines T for interpreting the coefficient output; it does not prescribe the subsequent joint inverse solver. The region is the authors' selection from channel reconstructions, not a claimed independently known pure-material acquisition calibration. This definition uses only the released counts, dark/flat calibrations and stated ray geometry; it does not assume the true material maps or a privately known quartz spectrum. The evaluator uses an independently reproduced value of this same observable functional, rather than adding hidden calibration input.

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Source-object distance | 57.999001 | mm | Distance from rotation center to source |
| Source-detector distance | 883.081116 | mm | Reference-release reconstruction distance |
| Detector pitch | 0.80 | mm | Physical detector position spacing |
| Detector positions | 257 | count | Complete one-dimensional detector extent |
| Acquired projections | 600 | count | No angular subsampling |
| Rotation angle endpoints | 0, 2*pi, inclusive | rad | Supplied array uses the source implementation's inclusive convention |
| Horizontal offset | 257/2 - 127.379570 | detector pixels | Positive translation along the detector basis vector |
| Image | 255 by 255 | pixels | Square support centered at the rotation origin |
| Pixel pitch | 0.80/(883.081116/57.999001) | mm | Detector pitch divided by magnification |
| Basis areal mass | 0.015 | g/cm^2 | Nominal normalization for the tabulated attenuation responses |
| Energy bins | 128, width 1.10 | keV | Lower edge 20.35 keV, upper edge 161.15 keV |
| Joint-fit channels | zero-based 29 through 93 | index | 65 channels; retain all other channels for spectral estimation |
| Acquisition ordering | reverse `counts` along angle before reconstruction | convention | Matches the recorded instrument direction |
| Source at angle theta | (SOD*sin(theta), -SOD*cos(theta)) | image pixels | SOD is source-object distance divided by object-plane pixel pitch |
| Detector direction | (cos(theta), sin(theta)) | convention | Increasing detector position |
| Image axes | x right, y up; row index increases down | convention | Centered square-pixel geometry |
| Boundary | zero outside image support | convention | Finite specimen reconstruction support |
| Noise | photon-counting observations; no supplied Gaussian sigma | counts | Do not assume simulated additive noise or an exact noiseless fit |

The source acquisition settings record slightly different source-detector and axis-position estimates. This task intentionally fixes the reconstruction values above, preserving the selected standard implementation's geometry. A geometric system matrix may be formed from these known acquisition parameters; no specimen-specific reconstructed calibration is supplied.

Data source: Sittner and colleagues, *Data for: Spectral X-ray Computed Micro Tomography: 3-dimensional chemical imaging*, version DOI [10.14278/rodare.1627](https://doi.org/10.14278/rodare.1627), archive `Spectral_CT_W_Au_Pb.zip`. The data are distributed under CC BY 4.0, with attribution to the dataset authors. The dictionary comes from the accompanying ADJUST research release and retains its published tabulation.

## Method Hints

A dictionary-constrained joint factorization can reduce spectral ambiguity compared with reconstructing and factorizing every energy channel independently. An alternating method updates spectral mixtures while holding the spatial maps fixed, then updates maps with the spectral responses fixed. Spectral projected-gradient steps with nonmonotone line search are one suitable family. A residual-lifting strategy can accelerate early progress, but the best physically feasible iterate should be distinguished from the final iterate when the trajectory is nonmonotone.

The specimen-specific matrix spectrum can be estimated from actively reconstructed spectral channels. A row/column-normalized simultaneous iterative reconstruction is useful for that preliminary tomography. Any such matrix spectrum is an estimate computed from the same observations, with corresponding limitations; it is not known truth or independent instrument calibration.

The output-basis protocol in Data Description fixes the observation-derived fourth spectrum and its coordinate normalization; the subsequent joint factorization can use that basis without treating it as independent calibration.

## References

1. M. T. Zeegers, A. Kadu, T. van Leeuwen, and K. J. Batenburg. *ADJUST: A Dictionary-Based Joint Reconstruction and Unmixing Method for Spectral Tomography*. Inverse Problems, 2022. DOI [10.1088/1361-6420/ac932e](https://doi.org/10.1088/1361-6420/ac932e). Preprint [arXiv:2112.11406](https://arxiv.org/abs/2112.11406). The selected reference-release equality variant, skip-condition correction, and matched line discretization are identified above.
2. J. Sittner and colleagues. *Spectral X-ray Computed Micro Tomography: 3-dimensional chemical imaging*. X-Ray Spectrometry. DOI [10.1002/xrs.3200](https://doi.org/10.1002/xrs.3200).
3. M. Schmidt, E. van den Berg, M. Friedlander, and K. Murphy. *Optimizing Costly Functions with Simple Constraints: A Limited-Memory Projected Quasi-Newton Algorithm*. AISTATS, 2009; associated spectral projected-gradient optimization software.
