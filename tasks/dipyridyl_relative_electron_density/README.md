# Relative Crystal Electron Density from Indexed X-ray Diffraction

> Reconstruct a complete three-dimensional periodic electron-density contrast field from real indexed single-crystal diffraction intensities, despite missing Fourier phases, finite resolution and measurement noise.
> Domain: chemistry & material science | Keywords: crystallographic imaging, phase retrieval, diffraction inversion | Difficulty: Hard

## Background

X-rays scatter from the electrons within a material. In a periodic crystal,
interference concentrates the scattered intensity near reciprocal-lattice
points. The positions of these reflections reveal the lattice geometry, while
their intensities constrain the spatial distribution of electrons inside one
unit cell. A detector records intensity and generally loses the complex phase
of each scattering amplitude.

Recovering a real-space density from these measurements supports molecular
structure determination, identification of bonding arrangements and analysis
of functional crystalline materials. Finite angular coverage and resolution
leave spatial detail unresolved. Measurement noise can also produce negative
background-subtracted intensity estimates. A reconstructed contrast map must
therefore be interpreted at its stated resolution and relative physical scale.

## Problem Description

The observations are 4,508 indexed real X-ray reflections of dipyridyl
disulfide. Recover a real periodic density contrast on the complete unit cell.
Let the integer Miller vector be $h=(h,k,l)$ and let the density sample be
$\rho_j$, where the three integer components of $j$ index a grid with
$N_a=72$, $N_b=24$, $N_c=96$ and total sample count $N=N_aN_bN_c$.
The fractional position is
$u_j=(j_a/N_a,j_b/N_b,j_c/N_c)$. The declared relative-scale forward model is

```math
F_h=\frac{10}{N}\sum_j\rho_j\exp(2\pi i\,h\cdot u_j),\qquad
I_h=|F_h|^2+\eta_h.
```

Here $F_h$ is the complex scattering amplitude, $I_h$ is the supplied signed
relative intensity, $\eta_h$ is measurement error and $i^2=-1$. The factor ten
fixes the output scale; it is not a fitted radiometric gain. The amplitudes
have square-root relative-intensity units, as does the specified density
contrast. No absolute electron-count calibration is provided. The unmeasured
zero-frequency coefficient is represented by a zero-mean density contrast.
Negative contrast values and negative noisy observed intensities are legitimate.

The real-density assumption gives $F_{-h}=\overline{F_h}$, where the bar means
complex conjugation. The known screw-axis rotational orbit also supplies
equivalent measured moduli. Missing phases make the inverse nonlinear and
nonunique. In particular, translating the entire periodic density or replacing
it by its spatial inversion leaves all measured moduli unchanged.

Save `output/reconstruction.npz`, relative to the task workspace, with exactly
the following required array:

| Key | Shape | Dtype | Description |
|---|---|---|---|
| `reconstruction` | `(1, 72, 24, 96)` | float32 or float64 | Real zero-mean relative electron-density contrast, in square-root relative-intensity units; axes batch, fractional a, fractional b, fractional c. Node coordinates are j/N on each axis, excluding the periodic endpoint. |

The Cartesian position is $r_j=B u_j$, where the lattice vectors are the
columns of $B$. The first vector points along Cartesian x, the second along
positive y, and the third lies in the x-z plane with the stated monoclinic
angle and positive z component. Thus the grid spacings along lattice vectors
are a/72, b/24 and c/96 in angstroms. Do not return atomic coordinates, a cropped
asymmetric unit, a magnitude image or an independently rescaled map.

All declared voxels are evaluated using uncentered cosine similarity and RMS
error divided by the complete comparison map's dynamic range. The comparison
permits a global periodic translation, including its continuous trigonometric
interpolation, and either complete spatial-inversion choice; the choice
maximizes full-map cosine similarity. This represents the actual diffraction
origin/inversion ambiguity. No fitted gain, global sign change, crop or
selected-voxel mask is allowed. Independent checks also evaluate the forward
intensity residual over every signed measured intensity and the relative
zero-mean error. This is a validated numerical-method comparison, not a claim
of independently known exact crystal-density truth.

## Data Description

`data/raw_data.npz` contains existing measured relative intensities, their
uncertainties and their indexed reciprocal coordinates. The batch axis is one
complete experiment.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| `hkl` | `(1, 4508, 3)` | int32 | Integer Miller coordinates in the measured lattice basis, dimensionless; final axis h, k, l. |
| `intensity` | `(1, 4508)` | float64 | Signed integrated reflection intensity in relative intensity units; measurement order matches hkl. The complete observed RMS is one. |
| `sigma_intensity` | `(1, 4508)` | float64 | Reported positive standard uncertainty of each signed intensity, in the same relative intensity units. |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Measurement modality | Indexed single-crystal X-ray reflections | — | Existing integrated measurements, rather than detector-frame images. |
| Reflections | 4508, all retained | reflections | All indexed observations and uncertainties remain available. |
| Cell a | 15.8489 | angstrom | First lattice-vector length. |
| Cell b | 5.5008 | angstrom | Second lattice-vector length. |
| Cell c | 23.1175 | angstrom | Third lattice-vector length, retained from the indexed observation file. |
| Alpha | 90 | degrees | Angle between b and c. |
| Beta | 96.916 | degrees | Angle between a and c. |
| Gamma | 90 | degrees | Angle between a and b. |
| Crystal symmetry | P21, unique axis b | — | Author-identified noncentrosymmetric screw-axis structure. |
| Reciprocal rotational operators | I and diag(-1,1,-1) | — | Identity and twofold rotation; I is the three-dimensional identity matrix. |
| Fractional screw translations | (0,0,0) and (0,1/2,0) | lattice fraction | Translations paired with the two rotational operators. |
| Anomalous scattering | Neglected | — | Real density and conjugate Friedel partners are assumed. |
| Reported high-resolution limit | 0.7734 | angstrom | Observation-file header; exact spacings follow from hkl and the cell metric. |
| Relative intensity scale | RMS of all supplied intensities = 1 | relative intensity unit | Absolute scattering-amplitude calibration is not supplied. |
| Negative observations | Retained | relative intensity unit | Background-subtracted noise values remain in the full residual calculation. |
| Uncertainty model | Reported per-reflection standard uncertainties | relative intensity unit | No target-derived noise variance or known exact phases are supplied. |
| Spatial boundary | Periodic on all three lattice axes | — | Opposite faces of the unit cell are identified. |
| Density mean | Zero | square-root relative-intensity unit | Defines contrast when the direct-beam coefficient is unmeasured. |

The author offers these data through the crystallographic tutorial. A separate
external-data redistribution grant was not identified; this local research
package makes no broader data-license claim.

## Method Hints

A dual-space charge-flipping family alternates real-space density modification
with reciprocal-space amplitude replacement. Expand observed moduli using the
known rotational orbit and conjugate Friedel partners, and initialize unknown
phases consistently with a real map. Small-density values change sign; an upper
threshold can also suppress a spurious single-dominant-peak solution. Replacing
the temporary moduli with the observed amplitudes retains the recovered phases.

This process is a nonlinear feasibility iteration rather than guaranteed
descent of a smooth objective. Finite iteration budgets, arithmetic backends
and initial phases can affect its trajectory. Monitor spatial structure as
well as observation agreement, since arbitrary phases can reproduce measured
moduli without recovering the crystal structure. Handle negative noisy
intensities explicitly when defining amplitude constraints, while retaining
every signed observation when assessing the intensity residual.

## References

1. B. H. Toby and R. B. Von Dreele. *GSAS-II: the genesis of a modern open-source all purpose crystallography software package*. Journal of Applied Crystallography 46, 544–549 (2013). [Published article](https://doi.org/10.1107/S0021889813003531).
2. G. Oszlányi and A. Sütő. *Ab initio structure solution by charge flipping*. Acta Crystallographica A 60, 134–141 (2004). [Author manuscript](https://arxiv.org/abs/cond-mat/0308129).
