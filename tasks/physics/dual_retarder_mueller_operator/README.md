# Dual-Retarder Mueller Operator

> Recover an optical sample's complete Mueller transfer matrix from an angle-modulated intensity scan while respecting polarization handedness, absolute throughput, and acquisition rank.
> Domain: physics | Keywords: polarimetry, optical characterization, inverse problems | Difficulty: Medium

## Background

An optical component can change intensity, rotate polarization, convert linear into circular polarization, or partially destroy polarization. A Mueller matrix describes these effects through a real linear map on the four Stokes parameters. Measuring this transfer operator supports optical-instrument calibration, material characterization, and the control of polarization aberrations in precision optical systems.

A camera or photodiode records intensity rather than all polarization components directly. Known rotating retarders and polarizers encode the unknown transfer matrix into a sequence of intensity measurements. Identifiability depends on the angular schedule and modulation geometry; collecting a nominal number of measurements alone does not guarantee that all matrix entries can be recovered.

## Problem Description

Reconstruct the full real 4-by-4 Mueller matrix $M$ of a single optical sample from the complete 24-angle **author-simulated** acquisition provided here. The task is nonspatial: the unknown is an optical transfer operator, not an image. The source uses a noiseless physical optical model. The sample matrix itself is unknown, and no particular sample family or factorization should be imposed on the inverse.

Column Stokes vectors are ordered $(I,Q,U,V)^T$. With complex electric-field components $E_x,E_y$, the convention is $I=|E_x|^2+|E_y|^2$, $Q=|E_x|^2-|E_y|^2$, $U=2\mathrm{Re}(E_xE_y^*)$, and $V=2\mathrm{Im}(E_xE_y^*)$; the star denotes complex conjugation. The sample obeys $s_{out}=Ms_{in}$.

For modulation angle $\theta_n$, the intensity measurement is

```math
p_n=e_0^TP(0)R(5\theta_n,\pi/2)\,M\,R(\theta_n,\pi/2)P(0)e_0,
\qquad e_0=(1,0,0,0)^T.
```

Here $p_n$ is the detector power relative to unit incident intensity, $P$ is the Mueller matrix of an ideal linear polarizer, and $R$ is that of a lossless linear retarder. Angles and retardances are in radians. Their calibrated optical forms are

```math
P(a)=\frac12\begin{pmatrix}
1&c&s&0\\c&c^2&cs&0\\s&cs&s^2&0\\0&0&0&0
\end{pmatrix},\qquad c=\cos(2a),\quad s=\sin(2a),
```

```math
R(a,d)=\begin{pmatrix}
1&0&0&0\\
0&c^2+Cs^2&(1-C)cs&-Ss\\
0&(1-C)cs&Cc^2+s^2&Sc\\
0&Ss&-Sc&C
\end{pmatrix},\qquad C=\cos d,\quad S=\sin d.
```

The scalar $a$ is an optic's orientation, $d$ is its phase retardance, and $c,s,C,S$ have the definitions above. The rightmost optic acts first. Both polarizer axes are zero; the generating retarder rotates with the supplied angles and the analyzing retarder rotates five times faster. These known calibrations are independent of the sample. The endpoint-inclusive angular scan has redundant endpoint states, so rank and conditioning matter.

Save **`output/reconstruction.npz`** relative to the task workspace:

| Key | Shape | Dtype | Description |
|---|---|---|---|
| `reconstruction` | `(1, 4, 4)` | float64 | Complete finite real signed Mueller matrix, dimensionless absolute throughput; axes batch, output Stokes component, input Stokes component, both ordered I,Q,U,V |

Preserve all 16 entries, their signs, and the absolute intensity scale. Do not normalize by an inferred matrix element or fit an output scale. Evaluation uses cosine NCC of flattened matrices without mean subtraction, dynamic-range-normalized RMS error, and relative L2 error of predicted versus observed detector power. A zero-norm estimate scores NCC zero. Every metric must pass; there is no sign, transpose, scale, phase or coordinate alignment.

## Data Description

`data/raw_data.npz` contains the complete angle schedule and intensity acquisition. No sample matrix, sample parameters, or target-derived calibration is present.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| `angles` | `(1, 24)` | float64 | Generator-retarder angle relative to the x axis, rad, endpoint-inclusive ascending scan |
| `power` | `(1, 24)` | float64 | Corresponding detector intensity divided by the known incident intensity, dimensionless, in the same order |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Generator angle range | 0 to pi, 24 equally spaced values including endpoints | rad | Complete bound angular schedule |
| Analyzer/generator angular ratio | 5 | dimensionless | Analyzer retarder angle divided by generator retarder angle |
| Both retarder phase delays | pi/2 | rad | Ideal quarter-wave calibration at the modeled operating wavelength |
| Both polarizer axes | 0 | rad | Fixed x-oriented polarizers |
| Modulator angular offsets | 0 | rad | Encoder origins and known optical-axis offsets |
| Incident Stokes vector | (1, 0, 0, 0) | relative intensity | Unit unpolarized input before the generating polarizer |
| Polarizer maximum/minimum transmission | 1 / 0 | dimensionless | Ideal intensity transmissions; unpolarized input transmits one half |
| Detector gain / offset | 1 / 0 | relative intensity | Known linear detector calibration |
| Added measurement noise | 0 | relative intensity | Author's noiseless benchmark; numerical roundoff remains |
| Spatial sampling | one optical sample | none | No spatial axes or pixel averaging |
| Spectral model | fixed monochromatic retardance | none | No wavelength value is assigned; phase calibration fully specifies this model |

## Method Hints

The physical modulation is linear in the sample matrix entries. Let $a_n^T$ be the first row of the analyzing optics and $g_n$ the first column of the generating optics. Then $p_n=a_n^TMg_n$. Construct a design row from their outer product in the same row-major order as the unknown matrix and solve the unweighted full-operator least-squares inverse. Singular-value analysis exposes angular aliasing and poorly measured operator directions. The selected native Moore–Penrose inverse discards singular values at or below 1e-15 times the largest singular value; it does not constrain the matrix to a target family or use a target-derived normalization.

## References

1. J. N. Ashcraft et al., “Katsu: A Python package for Mueller and Stokes simulation and polarimetry,” Journal of Open Source Software 10(107), 7375 (2025). https://doi.org/10.21105/joss.07375
2. R. M. A. Azzam, “Photopolarimetric measurement of the Mueller matrix by Fourier analysis of a single detected signal,” Optics Letters 2, 148–150 (1978). https://doi.org/10.1364/OL.2.000148
3. M. H. Smith, “Optimization of a dual-rotating-retarder Mueller matrix polarimeter,” Applied Optics 41, 2488–2493 (2002). https://doi.org/10.1364/AO.41.002488
