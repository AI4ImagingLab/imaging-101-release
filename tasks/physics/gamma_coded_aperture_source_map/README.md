# Coded Aperture Gamma Source Mapping

> Reconstruct a planar gamma-emission image from a real masked semiconductor-detector exposure, where overlapping coded shadows and defective pixels obscure the source.
> Domain: physics | Keywords: nuclear imaging, coded aperture, inverse imaging | Difficulty: Medium

## Background

Gamma rays penetrate matter and can reveal radioactive material without direct optical access. A semiconductor detector records the energy deposited by incident radiation. At these photon energies ordinary glass lenses cannot form an image, so spatial information must be encoded before photons reach the detector. A perforated absorbing mask creates a position-dependent shadow. Several apertures admit photons simultaneously, improving collection efficiency relative to a single pinhole while superposing the contributions of different source positions.

Computational decoding separates those overlapping contributions into an image of the emitting region. Compact coded cameras support radiation surveys, detector characterization and localization of radiotracers. The finite aperture dimensions, mask penetration, detector defects and counting statistics limit spatial resolution and quantitative interpretation. A decoded response map retains the camera's point-spread response; it does not independently establish an absolute radioactive activity in becquerels.

## Problem Description

The input is one full 256 by 256 experimental Minipix/Timepix3 exposure of an Am-241 source, acquired with a rank-31 no-two-holes-touching tungsten mask. Recover the spatial emission response on the specified plane 50.0 mm in front of the mask. The plane distance is known; lateral source position and the image are unknown. Retain the entire detector observation as supplied.

Let f(u,v) denote the nonnegative planar source strength, a(p,q) the binary aperture transmission, b the detector-to-mask separation and z the source-plane distance. A source displacement u projects its mask shadow by -b*u/z on the detector, with magnification m=1+b/z. A discrete idealized shadow model is

```math
y_{ij}=\sum_{k,l} h_{i-k,j-l}(z)f_{kl}+\beta_{ij}+\epsilon_{ij},\qquad m=1+b/z.
```

Here y is accumulated detector energy in keV, h is the geometrically magnified aperture response, beta is unmodulated background/penetrating radiation, and epsilon contains measurement fluctuations and detector errors. Indices i,j label detector row and column; k,l label source-induced shadow displacements. The summation is the finite sampled convolution. A central repeated-mask period permits a circular decoding approximation. The acquisition is not assumed to be a perfectly noiseless binary-shadow simulation. The missing direction information at each detector pixel and overlapping shadows make the inverse ambiguous without the known aperture geometry.

Save `output/reconstruction.npz`, relative to the task workspace, with exactly the required quantity below:

| Key | Shape | Dtype | Description |
|---|---|---|---|
| reconstruction | (1, 126, 126) | float32 | Real signed, unnormalized MURA source-response image in summed keV units; axes batch, row, column. |

This is the planar decoded image at z=50.0 mm, with no 180-degree output rotation. Its nominal source-plane sample pitch is 0.1375 mm. Column j corresponds to x=-(j-63)*0.1375 mm and row i to y=-(i-63)*0.1375 mm in the unrotated reconstruction convention. The output includes the finite-resolution response and signed decoding sidelobes. Retain its physical energy scale: use a discrete sum, not an average, in decoding; do not normalize its maximum, clip negatives, rotate, translate or crop the declared output.

The decoded energy response is not absolute activity. No independent Bq calibration is provided. Quality uses uncentered cosine similarity NCC=(x dot r)/(norm(x) norm(r)) and dynamic-range NRMSE=sqrt(mean((x-r)^2))/(max(r)-min(r)), where x and r are flattened response images on the declared grid. A zero estimate has NCC zero. Both metrics must pass, with no alignment or fitted gain. Agreement measures reproducibility of the standard decoding convention, rather than independent physical truth.

## Data Description

Load `data/raw_data.npz` using `allow_pickle=False`.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| detector_energy | (1, 256, 256) | uint32 | Full measured integrated deposited energy, keV; batch, detector row, detector column. |
| mask_pattern | (1, 124, 124) | float32 | Binary dimensionless aperture transmission; 1 is an opening, 0 tungsten; batch, mask row, mask column. |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Detector | MinipixEDU, Timepix3, silicon sensor | - | Semiconductor gamma camera |
| Detector side | 14.08 | mm | Square active sensor width |
| Detector sampling | 256 by 256 | pixel | Native detector grid, row down and column right |
| Detector pitch | 0.055 | mm | 14.08/256, no observation downsampling |
| Sensor thickness | 0.5 | mm | Silicon conversion layer |
| Mask side | 9.92 | mm | Full repeated 2 by 2 mask |
| Mask representation | 124 by 124 | pixel | Binary no-two-holes-touching grid |
| Mask rank | 31 | dimensionless | Four repeated central coded periods |
| Aperture diameter | 0.08 | mm | Circular holes |
| Tungsten thickness | 0.11 | mm | Mask attenuation thickness |
| Mask-detector distance b | 20.0 | mm | Nominal camera geometry; assembly errors remain possible |
| Source plane z | 50.0 | mm | Known mask-to-plane separation |
| Magnification | 1.4 | dimensionless | 1+b/z |
| Source-plane sample pitch | 0.1375 | mm | Detector pitch times z/b |
| Acquisition time | 15 | minute | Integrated experimental exposure |
| Dominant emission energy | 59.5 | keV | Am-241 gamma line; no narrow energy window imposed |
| Energy threshold | 5 | keV | Lower detector threshold |
| Coordinate origin | Detector centre; reconstruction index (63,63) | pixel | Unrotated author reconstruction convention |
| Finite field | Square full detector; no external detector samples | - | Repeated-mask central-period decoding approximation |
| Noise | Real detector/counting errors; independent variance unavailable | - | No fabricated noise-free image or variance map |

The NPZ conversion preserves every native uint32 observation and every float32 mask sample. The mask is instrument calibration independently available before reconstruction. Position-dependent camera mismatch and finite source size remain limitations of the nominal geometric model.

## Method Hints

A modified uniformly redundant array supports direct correlation-style decoding. Construct the signed central decoding period from the supplied mask, project that period according to the known geometric magnification, and combine it with the corresponding detector region using circular convolution. Define d as the signed projected decoder, c as that detector region and F as the unnormalized forward discrete Fourier transform. The decoded image is

```math
r=\mathrm{Re}\left[F^{-1}\{F(c)F(d)\}\right].
```

Here the inverse transform includes its usual inverse-size normalization, so r equals the discrete convolution sum. Camera defects warrant robust neighbourhood preprocessing before decoding. Nearest-neighbour versus sparse/delta sampling of the decoder gives different discretizations; compare their localization and sidelobes without treating their raw peak amplitudes as the same calibrated activity. The core inverse is direct and does not require artificial iterations.

## References

- Meißner et al. (2024). *3D-localization of single point-like gamma sources with a coded aperture camera*. Physics in Medicine and Biology 69, 165004. [DOI](https://doi.org/10.1088/1361-6560/ad6370). The selected benchmark is its planar MURA decoding stage at a known source plane, rather than its subsequent three-dimensional point-localization fit.
- Meißner et al. (2024). *Assessment of the axial resolution of a compact gamma camera with coded aperture collimator*. EJNMMI Physics 11, 30. [DOI](https://doi.org/10.1186/s40658-024-00631-5). Camera physics and preprocessing description.
- Accorsi (2001). *Design of Near-Field Coded Aperture Cameras for High-Resolution Medical and Industrial Gamma-Ray Imaging*. [MIT dissertation](https://hdl.handle.net/1721.1/8684).
