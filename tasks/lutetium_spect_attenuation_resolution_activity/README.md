# Lutetium SPECT Activity with Attenuation and Resolution Modelling

> Reconstruct a three-dimensional lutetium activity volume from real single-photon projections while accounting for photon attenuation, scatter and depth-dependent detector blur.
> Domain: medicine | Keywords: emission tomography, photon transport, statistical reconstruction | Difficulty: Hard

## Background

Single-photon emission computed tomography detects gamma photons emitted by an administered radionuclide. A physical collimator selects photon directions, allowing rotating cameras to observe different line projections of the radioactive distribution. Unlike coincidence detection, each accepted photon supplies a collimator-defined projection measurement.

Quantitative emission imaging supports radionuclide therapy planning and assessment. Photons can be absorbed or scattered before reaching the camera, and collimator resolution decreases with distance. These effects mix spatial information and bias activity estimates unless the acquisition model accounts for them. Independently acquired CT information supplies attenuation coefficients, while neighbouring energy windows estimate scattered counts.

## Problem Description

The observations are a complete real physical-phantom acquisition containing hot spheres in an emitting background. Reconstruct the nonnegative activity field on the supplied CT-calibration lattice. Its scale is the native projection-count contribution per reconstruction voxel, rather than an absolute radionuclide concentration.

```math
g_{j r k}\sim\mathrm{Poisson}\bigl((H a)_{j r k}+s_{j r k}\bigr),\qquad
(H a)_{j r k}=\sum_i\left[K_j\left(p_j R_j a\right)\right]_{i r k}.
```

Here $a$ is the activity volume, $g$ is the photopeak count array, $j$ indexes camera angle, $r$ indexes detector radial position and $k$ indexes axial position. The operator $R$ rotates the zero-extended volume into the detector frame, $p$ is the photon survival probability, $K$ blurs each detector-parallel plane, and $s$ is the additive scatter estimate. The integration is a voxel sum: no additional voxel-size multiplier belongs in the count projection.

For detector-facing coordinate $i$ increasing towards the camera, the path begins at the centre of the emitting voxel:

```math
p_{j i r k}=\exp\left[-d\left(\tfrac12(R_j\mu)_{i r k}
+\sum_{h>i}(R_j\mu)_{h r k}\right)\right].
```

The known attenuation coefficient is $\mu$ in inverse centimetres and $d$ is voxel spacing in centimetres. The supplied independent CT calibration must retain its physical scale. Neighbouring windows determine scatter using their widths $w$:

```math
s=w_{\mathrm{peak}}\left(\frac{g_{\mathrm{lower}}}{2w_{\mathrm{lower}}}
+\frac{g_{\mathrm{upper}}}{2w_{\mathrm{upper}}}\right).
```

Save **`output/reconstruction.npz`** with key **`reconstruction`**, shape **`(1,128,128,128)`**, real **float32**, axis order **batch,x,y,z**, in the native count-contribution scale. All voxels are evaluated, including the background; submit finite, nonnegative values everywhere. No cropping, fitted scaling, alignment or support selection is allowed. Evaluation uses cosine NCC, RMS error divided by reference dynamic range, maximum absolute error divided by that range, and full photopeak forward relative L2 error, and requires zero negative voxels. This is a precisely specified standard-estimator agreement task; such agreement does not establish voxelwise physical truth. Required boundaries are respectively 0.9999999, 0.000001, 0.000005 and 0.22, with NCC bounded below and errors above.

## Data Description

| Key | Shape | Dtype | Description |
|---|---|---|---|
| energy_windows | (1,3,96,128,128) | uint16 | Original photon counts, ordered photopeak/lower/upper, then angle/radial/axial; unit counts. |
| attenuation | (1,128,128,128) | float32 | Independently acquired CT-derived attenuation at 208 keV, aligned x/y/z lattice; unit cm⁻¹. |
| angles | (1,96) | float32 | Sorted native camera angles, unit degrees. |
| radii | (1,96) | float64 | Corresponding camera-to-centre distances, unit cm. |

Acquisition metadata:

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Scanner/collimator | Siemens Symbia T / medium energy | — | Dual-head parallel-hole single-photon acquisition. |
| Radionuclide / peak | lutetium-177 / 208 | keV | Selected emission energy; other emission peak is outside this task. |
| Photopeak bounds | 187.19999694824, 228.80000305176 | keV | Exact archived DICOM energy limits. |
| Lower bounds | 166.39999389648, 187.19999694824 | keV | Lower scatter window. |
| Upper bounds | 228.80000305176, 249.60000610352 | keV | Upper scatter window. |
| Sampling | 96 views, nominal 3.75 degree increment, 15 seconds/view | degree, s | Use supplied native angles, including initial offset 3.7144622802734375 degrees. |
| Voxel/detector spacing | 0.47951998710631993 | cm | Identical x/y/z and radial/axial spacing; archived value retained. |
| Rotation convention | detector azimuth = 270 minus camera angle | degree | Forward rotation is the negative detector azimuth about z. |
| Interpolation | bilinear, centre (N−1)/2, corner-aligned coordinates | — | Zero extension; pad x and y by 27 voxels before rotating, yielding 182×182×128. |
| Coordinate precision | float32 pi and pixel-normalisation constants | — | Homogeneous rotation retains these constants; inverse normalized transform samples the source lattice. |
| Object lattice origin | −304.49519181251316, −462.5951918125131, 110.4048081874937 | mm | Native DICOM affine translation; diagonal spacing 4.7951998710632 mm, x/y/z axes increasing. |
| Projection axes | angle, radial, axial | — | Original detector row order reversed on axial axis before radial/axial transpose; supplied arrays already use this convention. |
| PSF width coefficients | A=0.03161982752665455, B=0.1248503046423388, C=0.16137114205472364 | A dimensionless, B/C cm | Gaussian standard deviation at distance D is sqrt((A D+B)²+C²), in cm. |
| Detector distance | max(radius+((181/2)−i)d,0) | cm | Distance for padded depth index i=0,…,181. |
| PSF boundary/normalization | separable y/z, zero extension, each kernel sums to one | — | Each odd kernel spans at least three maximum standard deviations for its view. |
| CT conversion provenance | 130 kVp, bilinear table at 208 keV | kVp, keV | Input attenuation already converted and registered; no target activity is used. |
| Noise | photon counting; scatter estimated from independent windows | counts | No extra noise model or smoothing is applied to the windows. |

## Method Hints

Use the Poisson ordered-subsets expectation-maximization family. The conditional estimator is four sweeps of eight interleaved angle subsets from an all-ones volume. For subset $m$, let $B$ denote the inverse-angle continuum backprojector, $n=B\mathbf{1}$ the sensitivity, and $\delta=10^{-11}$ a numerical safeguard. The update is

```math
a\leftarrow\max\left(0,\ a+
a\frac{1}{n+\delta}\left[B\left(\frac{g}{Ha+s+\delta}\right)-n\right]\right).
```

Apply the same attenuation and symmetric detector blur in reverse order during backprojection, then rotate back. Inverse-angle bilinear rotation approximates the continuous adjoint; it is not the exact transpose of discrete interpolation. Preserve this convention for the specified estimator rather than substituting a different discrete inverse. Each depth-dependent Gaussian is sampled at integer offsets, divided by its discrete sum, and uses the same safeguard in its variance denominator. Physical calibration and the scatter term remain fixed throughout reconstruction.

## References

- Polson and colleagues, *PyTomography: A Python library for medical image reconstruction*, SoftwareX (2024), [doi:10.1016/j.softx.2024.102020](https://doi.org/10.1016/j.softx.2024.102020); [author manuscript, version 6](https://arxiv.org/abs/2309.01977v6), Appendix A.2 and Figure A.6.
- Author-offered phantom acquisition, [Zenodo dataset](https://doi.org/10.5281/zenodo.15314460), CC BY 4.0.
