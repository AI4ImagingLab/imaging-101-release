# Nickel Crystal Orientation from Electron Diffraction

> Recover a spatial nickel crystal-orientation map from real electron backscatter diffraction patterns despite noisy bands, crystal symmetry and a large nonlinear orientation search.
> Domain: chemistry & material science | Keywords: electron diffraction, orientation imaging, subspace reconstruction | Difficulty: Hard

## Background

Electrons scattered from a crystalline material undergo diffraction that produces angle-dependent intensity bands. An electron backscatter detector records these bands as a two-dimensional pattern. Their arrangement depends on the lattice orientation relative to the electron beam and detector, allowing a spatial raster scan to reveal the orientations of grains within a polycrystalline material.

Orientation maps help connect microstructure to mechanical behavior, texture and processing history. Individual patterns may have weak contrast, background variation and detector noise. Crystallographic symmetry creates multiple equivalent descriptions of the same physical orientation. Accurate interpretation therefore requires both a calibrated detector geometry and a model of the angular diffraction intensity.

## Problem Description

Recover the nickel crystal orientation at each of2000 native scan locations in a40-row by50-column rectangular region of a real0dB camera-gain acquisition. Each location has a native60 by60 detector pattern. The observations are indirect diffraction intensities, not supplied orientation labels. A separately simulated20keV nickel master pattern is legitimate calibration describing the angular diffraction response.

Let i,j be detector row and column, L the detector size in pixels, and (px,py,pz) the dimensionless Bruker projection center. Define a=i+1/2-L*py, b=j+1/2-L*px, D=L*pz, and phi the detector-minus-sample tilt. The sample-frame ray is

```math
\mathbf r(i,j)=(-\sin(\phi)a+\cos(\phi)D,\ b,\ -\cos(\phi)a-\sin(\phi)D).
```

For a unit scalar-first quaternion q=(w,x,y,z), its rotation R(q) maps a sample-frame ray into crystal coordinates. The ideal detector intensity is proportional to M(R(q)r/||r||), where M is the known spherical master intensity. The measured pattern also contains spatially varying background and noise. M is supplied as square-Lambert projections of the northern and southern hemispheres. Intensities have arbitrary detector/calibration scales, not calibrated photon counts. The nonlinear dependence on q, weak band contrast and cubic symmetry make the inverse ambiguous.

For completeness the square-Lambert coordinates can be computed as follows. For crystal unit vector (X,Y,Z), let f=sqrt(2(1-|Z|)); let B be the coordinate of larger absolute magnitude among X,Y, S the other coordinate, and sigma=-1 when B<0 and+1 otherwise. Put A=sigma*f/sqrt(2) and C=sigma*f*atan2(sigma*S,|B|)*2sqrt(2)/pi. Take (aL,bL)=(A,C) if |Y|<=|X| and(C,A) otherwise. Coordinates on the concatenated north-left/south-right array are u=aL/2-1/2+1[Z<0], v=bL, both normalized for interpolation; 1[condition] is1 if true and0 otherwise. Hemisphere index0 is north. The supplied array is not a stereographic projection and must not be vertically flipped.

Save `output/reconstruction.npz` with key `reconstruction`, numeric dtype **float32**, shape **(1,40,50,4)**, axes **(batch,scan_row,scan_column,quaternion_component)**, quaternion components **(w,x,y,z)**. Quaternions are dimensionless and must have unit norm within1e-5. R(q) is the active sample-ray-to-crystal rotation q*(0,r)*conjugate(q). Both q and-q are accepted. Physical crystal symmetry acts as left multiplication by any of the24 proper cubic rotation quaternions; such equivalent orientations are accepted. No spatial registration, permutation or scan crop is allowed.

Evaluation independently renders the calibrated diffraction contrast for every
submitted quaternion and uses all2000 observed patterns. Average each nonoverlapping
2by2 detector block to30by30 in real arithmetic; subtract the per-detector mean
across the2000 scan points, then subtract each resulting pattern's pixel mean.
Render the raw concatenated master (north then south) at the30by30 detector ray
centers with bilinear interpolation, aligned normalized endpoints and border
extension; subtract each rendered pattern's pixel mean. Its cosine with the
observed contrast is the dot product divided by both L2 norms. Zero norm scores0.
Take the best cosine among all24 left cubic representatives, so physical symmetry
and q versus-q are accepted despite finite master interpolation differences.

At every scan point, contrast cosine may fall at most0.001 below an independently
attained standard fit. The full-field mean must also be at least0.3447338940900441.
This predeclared dimensionless contrast allowance is not an angular-accuracy or
measurement-noise statement. Any better orientation fit is accepted; PCA bases,
dictionary nodes and refinement iterations are unrestricted. No spatial
permutation, additional intensity correction or fitted detector geometry is
allowed in scoring. The master is independent acquisition calibration; the
attained contrast is not measured orientation truth.

## Data Description

These are existing **real EBSD measurements**, paired with an independently simulated nickel diffraction calibration. The selected scan region is original rows50:90 and columns60:110, zero-based, of a149-row by200-column acquisition containing29800 patterns. This is a native40 by50 crop of scan locations, not the entire original scan and not a resized scan. Each supplied detector pattern remains native60 by60. Only the public2000 patterns may contribute to a scan-derived background estimate; no outside-ROI pattern or hidden label is provided.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| patterns | (1,40,50,60,60) | uint8 | Real detector ADU, axes batch,scan_row,scan_column,detector_row,detector_column. |
| master_pattern | (1,2,401,401) | uint8 | Independently simulated20keV nickel angular intensity in arbitrary units; axes batch,hemisphere,Lambert_row,Lambert_column; hemisphere0 north,1 south. |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Material | nickel, cubic high symmetry | phase | Single phase;24 proper rotations. |
| Accelerating voltage | 20 | kV | Matches the master-pattern calibration. |
| Sample tilt | 70 | degree | Source geometry relative to detector tilt0. |
| Detector tilt | 0 | degree | phi=-70degrees in the stated ray model. |
| Detector raw size | 60 by60 | pixels | L=60 and native center indices0..59. |
| Projection center(px,py,pz) | (0.4221,0.2179,0.4954) | dimensionless | Selected source-provided Bruker fractions; distance is normalized by detector height. |
| Detector coordinate convention | row/column centers at integer indices | pixels | The +1/2 terms in the ray equation encode the pixel-center/PC convention. |
| Master projection | square Lambert, no vertical flip | convention | Northern and southern matrices correspond to Z>=0 and Z<0. |
| Master grid | 401 by401 per hemisphere | pixels | Concatenate north then south along columns if using normalized(u,v) above. |
| Master lattice parameter | 0.35236 | nm | Independently supplied nickel simulation parameter. |
| Native scan size | 149 rows by200 columns | samples | Original acquisition, not all released here. |
| Selected native scan rows | 50..89 inclusive | index | Zero-based crop; all selected points evaluated. |
| Selected native scan columns | 60..109 inclusive | index | Zero-based crop. |
| Scan pitch | 1.5 | micrometer | Both scan directions; output retains native spacing. |
| Selected origin | x=90,y=75 | micrometer | x=original_column*1.5,y=original_row*1.5 relative original scan origin. |
| Camera gain | 0 | dB | Selected real acquisition. |
| Exposure | 3500 | microsecond | Acquisition metadata. |
| Detector noise variance | not independently calibrated | ADU squared | No known Poisson count scale or hidden clean-pattern input. |
| Spatial boundary | finite selected rectangle | convention | No periodic scan continuation or spatial alignment. |

## Method Hints

Dictionary indexing projects calibrated angular intensities for many possible orientations and matches each experimental pattern to those candidates. A principal-component basis learned from simulated dictionary patterns can reduce the matching dimension while retaining informative band variation. An incremental covariance update avoids storing the complete image dictionary. Local cubochoric orientation search can refine the initial discrete match. Keep preprocessing, quaternion action, detector pixel centers and interpolation conventions consistent.

## References

Varley,Z.T., Rohrer,G.S. and DeGraef,M. (2026). *Accelerating dictionary indexing of electron backscatter diffraction patterns with PCA and quantization*, Scientific Reports16,4382. [Published article](https://doi.org/10.1038/s41598-025-34391-8). The reference implementation uses a compressed dictionary and local refinement; this benchmark evaluates calibrated pattern contrast independently of that implementation.

Ånes,H.W., Hjelen,J., vanHelvoort,A.T.J. and Marthinsen,K. *Electron backscatter patterns from Nickel acquired with varying camera gain*, [real dataset](https://doi.org/10.5281/zenodo.7498632), CC BY4.0. Independent nickel master-pattern calibration, CC BY4.0, simulated with EMsoft. Only numerical calibration arrays are used; no reconstruction library is required.
