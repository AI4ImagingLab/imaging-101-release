# Three-Dimensional Bragg Coherent Crystal Strain Imaging

> Reconstruct the complex crystalline scattering density and partial-coherence kernel of a gold-coated ZnO microrod from two complete experimental three-dimensional diffraction volumes despite missing phase, finite coherence and nonconvex ambiguities.
> Domain: Mechanics | Keywords: coherent diffraction imaging, crystal strain imaging, phase retrieval | Difficulty: Hard

## Background

A coherent X-ray beam scattered near a crystalline Bragg reflection produces interference fringes that encode the shape and lattice deformation of a small illuminated crystal. Rotating the specimen through the reflection records successive detector planes through a three-dimensional reciprocal-space volume. Detectors measure photon counts rather than the complex scattered wave, so image formation requires recovering information absent from any individual detector frame.

The reconstructed complex density contains two physically distinct quantities. Its magnitude identifies the coherently scattering crystalline material. Its phase measures displacement projected onto the selected reciprocal-lattice vector. Spatial variations of that phase therefore reveal a component of the elastic distortion. A single Bragg reflection cannot determine all displacement components or the full strain tensor.

This capability is useful for investigating semiconductor interfaces, battery particles, defects and nanoscale mechanical devices. Gold-coated piezoelectric ZnO structures are particularly interesting because coating, interface charge and bending can all influence the lattice. Real experiments also have finite coherence, counting noise and finite angular sampling; separating these effects from specimen structure makes the inverse problem substantially harder than a noiseless Fourier-magnitude exercise.

## Problem Description

The observations are complete rocking scans from two spatial positions of one gold-coated ZnO microrod. Both positions must be reconstructed as three-dimensional volumes. Let `x` denote one complex scattering-density array on the oblique real-space grid and `k` a nonnegative, normalized three-dimensional partial-coherence kernel. The discrete observation model is


```math
I_{\mathrm{raw}}=Q R_{-s}\left(\left|\mathcal F_+x\right|^2 * k\right)+\eta,
\qquad
\mathcal F_+x=C\,\mathrm{ifftn}(C^{-1}x,\mathrm{norm}=\mathrm{forward}).
```


Here `Iraw` is the entire measured `(121,256,256)` photon-count array, whereas `N=(126,256,256)` is the specified reconstruction and Fourier grid. Define the embedding of measured reciprocal coordinates by two extra planes before and three after the rocking axis. `s` places the first measured maximum in this embedded grid at `(63,128,128)` and equals `(1,-19,-3)` for P1 and `(0,-49,-3)` for P2. `R` is a cyclic integer roll, and `Q` extracts rocking indices `2:123` after the inverse roll, retaining all original121 planes. Extra grid planes are a discretization convention, not additional observations. `eta` represents measurement error, `*` is zero-extended three-dimensional linear convolution cropped to its first argument's shape, and `C` is NumPy's `fftshift` on all three axes, with inverse `ifftshift`. The inverse FFT with `norm='forward'` is an unscaled positive-sign discrete Fourier transform. The convolution's starting crop index on each axis is `floor((kernel_length-1)/2)`; periodic convolution is not the declared coherence model. The inverse roll and extraction apply **after** finite linear convolution; these operations do not commute with the blur.

The amplitude and support of `x` are unknown. Here `x` is the **illuminated, probe-weighted crystalline scattering density**, rather than a deconvolved image of the entire physical rod. The nominal beam dimensions are known, but its full three-dimensional complex field is not independently calibrated; those dimensions do not specify a known probe array. Its phase is related to nominal projected displacement by `arg(x)=q dot u`, where `q` is the nominal Bragg wavevector, in radians per nanometer, and `u` is displacement in nanometers, under the assumption that additional spatially varying probe phase is negligible. Phase is defined modulo a constant and `2*pi`; absolute displacement is consequently not identifiable. Finite reciprocal-space coverage, phase loss, support uncertainty, partial coherence and nonconvex local minima make recovery difficult.

Save **one** `output/reconstruction.npz`, relative to the task workspace, containing both required arrays:

| Key | Shape | Dtype | Description |
| --- | --- | --- | --- |
| reconstruction | (1, 2, 126, 256, 256) | complex128 | Complex scattering density; axes are batch, position P1/P2, rocking-conjugate, detector-row-conjugate, detector-column-conjugate; amplitude in square-root-count units for the unscaled positive-sign DFT above; phase in radians. |
| coherence | (1, 2, 16, 16, 16) | float64 | Nonnegative unit-sum coherence kernels, dimensionless; axes are batch, position P1/P2, rocking-frequency offset, detector-row offset, detector-column offset. |

Both arrays must be finite. Preserve physical count scale: no arbitrary normalization of the submitted complex density is permitted. The evaluator checks both complex fields and coherence kernels through the independent raw forward model and the conditional compactness constraints below. The only reconstruction equivalences are a constant complex phase, an integer cyclic translation and conjugate inversion. These are Fourier-intensity ambiguities; no magnitude substitution, continuous spatial deformation, arbitrary intensity scaling or fitted linear phase ramp is allowed. Relative projected strain may be studied after subtracting a fixed support-interior mean; this removes an unidentifiable uniform strain offset rather than providing an absolute strain measurement.

The density uses the fixed measured-peak-centered reciprocal origin above. This removes an observation-determined carrier phase ramp; no additional ramp is fitted during scoring.

Observation consistency is the squared relative amplitude error


```math
E_{\mathrm{raw}}=\frac{\sum_j\left(\sqrt{\max([Q R_{-s}(|\mathcal F_+x|^2*k)]_j,0)}-a_j\right)^2}{\sum_j a_j^2},
\qquad a=\sqrt{I_{\mathrm{raw}}}.
```


The index `j` ranges over every original measured voxel, including low-count pixels. The nonnegative maximum removes negative FFT roundoff. `Q` removes only the five extra grid planes; no original measurement is masked, thresholded or cropped for this metric. Consistency uses the submitted density's physical count scale with no fitted scalar or reference-dependent phase ramp. Every required metric must pass for both positions; intensity consistency does not establish a unique physical object. A solver may use noise suppression internally, but a thresholded internal objective is distinct from this raw-observation metric.

The nominal reciprocal basis `B` has laboratory rows `(outboard, up, downstream)` and columns `(rocking, detector row, detector column)`. With wavelength `lambda`, detector angles `delta` and `gamma`, angular step `dtheta`, pixel pitch `p`, distance `L`, and `c=2*pi/lambda`, define


```math
B=c\begin{pmatrix}
d\theta(1-\cos\delta\cos\gamma)&(p/L)\sin\delta\sin\gamma&(p/L)\cos\delta\\
0&-(p/L)\cos\gamma&0\\
d\theta\sin\delta\cos\gamma&(p/L)\cos\delta\sin\gamma&-(p/L)\sin\delta
\end{pmatrix},
\quad
q=c\begin{pmatrix}\cos\gamma\sin\delta\\\sin\gamma\\\cos\delta\cos\gamma-1\end{pmatrix}.
```


The oblique real-space voxel basis is `D=2*pi*inverse(B).T*diag(1/N)`, in nanometers per array index. These are nominal first-order acquisition coordinates. The nominal wavevector magnitude differs by about 0.56% from the article's reported absolute reflection position; use the specified nominal geometry consistently. This task concerns the relative field in that geometry and does not require reproducing the article's absolute strain, refraction correction or electric-field inference.

The conditional compactness prior uses only the submitted field. Peak-center its amplitude by an integer cyclic shift, smooth amplitude with a Gaussian of sigma1voxel, half-sample reflected boundaries and4sigma truncation, and define support where the smooth amplitude is at least0.1of its maximum. Its support volume fraction must be at most1/8. These scale/threshold conventions come from the selected shrinkwrap method and its initial half-box; the volume cap is an explicit v2 conditional prior, not independently measured true crystal geometry. The squared-amplitude fraction outside this self-derived support may exceed the independently attained standard fraction by at most1e-8. For this finite-grid support calculation choose the better of the field's own direct and conjugate-inversion peak-centered representatives. This fixes boundary/gauge handling without a hidden target mask.

The original all-voxel raw amplitude residual limits are0.08332872900431508(P1) and0.09608879570479088(P2); kernel negative mass must be zero and unit-sum error at most1e-12. All conditions must pass for both positions. The evaluator does not require agreement to a particular complex field, partial-coherence kernel or strain image. It establishes intensity/compact-prior consistency at the declared scale; joint object/PSF ambiguity remains, and neither true absolute strain nor an independently measured PSF is certified.

Strain is a derived diagnostic rather than an additional submitted array. For any supported complex field z, the central index-phase gradient is g(j,n)=Arg[z(n+e(j))*conj(z(n-e(j)))]/2, with e(j) the unit index vector and Arg the principal argument. The nominal projected strain is q^T inverse(D)^T g/(q^Tq), dimensionless, using the physical D and q above. Diagnostic evaluation uses the submitted amplitude's exp(-1)peak support eroded by a3by3by3cube and subtracts its own interior strain mean; no additional spatial phase ramp is fitted. The convention assumes two-step phase differences remain in the principal interval and does not unwrap phase. Its finite range is reported; it is not compared with an unknown true strain.

## Data Description

The observations are real photon-count measurements, not simulated objects. The public archive contains the entire detector and angular extents at both positions; no two-dimensional slice or synthetic replacement is used.

| Key | Shape | Dtype | Description |
| --- | --- | --- | --- |
| photon_counts | (1, 2, 121, 256, 256) | int32 | Raw detector photon counts; axes batch, position P1/P2, increasing rocking angle, detector row downward, detector column. Unit: count. |
| theta | (1, 2, 121) | float64 | Recorded per-frame sample rocking motor angles. Unit: degree. |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
| --- | --- | --- | --- |
| Facility and beamline | APS 34-ID-C | none | Synchrotron coherent Bragg diffraction acquisition. |
| Specimen | Gold-coated ZnO microrod | none | Two spatial measurement positions, in stored P1/P2 order. |
| Photon energy | 9 | keV | Nominal incident X-ray photon energy. |
| Nominal focused beam dimensions | 2000 by 450, horizontal by vertical | nm | Experimental beam size after KB focusing and a slit; the full complex illumination field and its exact spatial profile are not calibrated inputs. |
| Wavelength convention | 1.2398419843320026 / 9 | nm | Photon energy-to-wavelength conversion used for the nominal geometry. |
| Detector | Timepix, 256 by 256 | pixel | Full recorded detector plane. |
| Pixel pitch | 55 | micrometer | Square detector pixels. |
| Sample-detector distance | 2.8 | m | Physical distance verified against the experimental description. |
| Exposure | 0.02 | second per frame | Per-frame integration time. |
| Nominal rocking increment | 0.002 | degree | Uniform-grid first-order model; recorded motor angles are also supplied. |
| Scan interval | 23.685004 to 23.925 | degree | Recorded first and last rocking angles. |
| Detector in-plane angle delta | 28.488 | degree | Nominal detector motor angle. |
| Detector out-of-plane angle gamma | 0.0478 | degree | Nominal detector motor angle. |
| Sample chi | 95.4 | degree | Recorded sample-circle position; above the rocking theta circle. |
| Sample phi | 3 | degree | Recorded sample-circle position; above theta. |
| Laboratory frame | outboard, up, downstream | none | Cartesian rows of the nominal reciprocal and real-space bases. |
| Boundary | zero extension for coherence convolution | none | No periodic wrap of coherence convolution. Fourier transform itself is discrete and periodic. |
| Measured peak index, P1 / P2 | (60,147,131) / (61,177,131) | array index | First maximum in stored rocking/row/column order, zero based. |
| Noise | measured counting data; no added synthetic noise | count | No independently calibrated read-noise or background variance is supplied. |

The data originate from the two NeXus files `P1.nxs` and `P2.nxs` in the dataset cited below. Counts and recorded theta arrays retain their original values and dtypes. The source files mistakenly label the distance string `2800` with meter units; the independently documented physical distance is 2.8 m. This correction applies to acquisition metadata, not to measured counts. Only relative strain in the specified geometry is supported by the task.

## Method Hints

Alternating Fourier-amplitude and real-space support constraints provides a useful starting point. Error-reduction iterations enforce support directly; hybrid input-output iterations use feedback outside it to reduce stagnation. Updating support from a thresholded Gaussian-smoothed amplitude can adapt the crystal boundary as spatial resolution increases.

A partial-coherence model convolves coherent diffraction intensity with a nonnegative kernel. Its kernel update must fit **measured intensity as coherent intensity convolved with the kernel**, preserving the direction of the physical observation equation. Multiple independent starts, objective-based selection and correlation-aligned complex averaging help assess and reduce nonconvex reconstruction variability. A single favorable initialization does not establish stability.

## References

- P. Jordt et al., *Visualizing Intrinsic 3D-Strain Distribution in Gold Coated ZnO Microstructures by Bragg Coherent X-Ray Diffraction Imaging and Transmission Electron Microscopy with Respect to Piezotronic Applications*, Advanced Electronic Materials (2021), DOI: https://doi.org/10.1002/aelm.202100546.
- P. Jordt, experimental gold-coated ZnO rocking-scan data, DOI: https://doi.org/10.5281/zenodo.7553757. Data license: Creative Commons Attribution 4.0.
- J. R. Fienup, *Phase retrieval algorithms: a comparison*, Applied Optics 21 (1982), DOI: https://doi.org/10.1364/AO.21.002758.
- S. Marchesini et al., *X-ray image reconstruction from a diffraction pattern alone*, Physical Review B 68 (2003), DOI: https://doi.org/10.1103/PhysRevB.68.140101.
- J. N. Clark et al., *High-resolution three-dimensional partially coherent diffraction imaging*, Nature Communications 3 (2012), DOI: https://doi.org/10.1038/ncomms1994.
