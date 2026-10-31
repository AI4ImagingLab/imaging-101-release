# Spinal Cord IVIM Microperfusion
> Recover tissue diffusion, vascular pseudodiffusion and signal fractions from measured spinal-cord diffusion MRI with overlapping slow and fast signal decays.
> Domain: medicine | Keywords: diffusion imaging, perfusion imaging, parameter inference | Difficulty: Hard

## Background

Diffusion-weighted MRI attenuates signals from moving spins. Tissue water produces a relatively slow decay with diffusion weighting, while blood moving through small vessels can contribute a faster incoherent-motion component. Separating these effects can characterize tissue and microvascular signal without injecting contrast agents.

The spinal cord presents a challenging setting because its small dimensions, physiological motion and low signal make the fast component difficult to distinguish. The vascular fraction is a signal fraction, and the additional fast decay coefficient is a pseudodiffusion coefficient, not a measured blood speed or an absolute perfusion rate.

## Problem Description

Infer four parameter fields from one complete measured right-left diffusion-encoding acquisition of human spinal cord at 7 T. For diffusion weighting b in s/mm², the measured magnitude signal is

```math
S(b)=S_0\exp(-bD)\,[f\exp(-bD^*)+1-f]+\epsilon(b).
```

Here S0 is extrapolated zero-weighting signal in acquisition intensity units, D is tissue diffusion in mm²/s, f is dimensionless vascular signal fraction, D* is additional vascular pseudodiffusion in mm²/s, and epsilon denotes measurement/model discrepancy. The total fast decay coefficient is D+D*. There is no measured b=0 image. At f=0, D* is unidentifiable; near D*=0, f is weakly identified. No independent parameter truth is supplied.

Save `output/reconstruction.npz` with all four required real float32 or float64 arrays, each shape `(1,128,134,6)` and axes batch,x,y,z: `S0` (positive intensity units), `diffusion` (D, mm²/s), `fraction` (f, fraction between0and1), `pseudodiffusion` (D*, mm²/s). Set all fields exactly zero outside `observation_mask`. Preserve native voxel locations; no registration, cropping or rescaling is allowed.

Evaluation requires all quantities and reprojection agreement. S0 and D use absolute RMS differences against a standard-method reference. D* uses weighted RMS `norm(w*(estimate-reference))/norm(w)` with fixed reference f as w; f uses the same definition with w=1-exp(-800*reference D*). A zero weight contributes zero; a zero total weight is an evaluator error. These are evaluation weights, not available calibration. The product fD* also uses unweighted RMS, so vascular strength must agree. All11 diffusion-weighted observations in all2523 supported voxels contribute to the forward relative L2 discrepancy `norm(predicted-measured)/norm(measured)`. Reference agreement does not establish physical accuracy in weakly identified regions.

## Data Description

| Key | Shape | Dtype | Description |
|---|---|---|---|
| measurements | (1,128,134,6,11) | float64 | Measured magnitude signal; acquisition intensity units; native x,y,z,b order |
| observation_mask | (1,128,134,6) | bool | Author-provided dilated cord support; dimensionless |
| bvalues | (1,11) | float64 | Measured diffusion weightings in s/mm² |
| affine | (1,4,4) | float64 | Native NIfTI voxel-to-world homogeneous affine; spatial outputs in mm |

### Acquisition metadata

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Magnetic field | 7 | T | Human spinal-cord acquisition |
| Encoding direction | right-left | anatomical direction | Single measured direction, not rotationally averaged |
| Matrix | 128 x 134 x 6 | voxels | Complete supplied spatial acquisition |
| Nominal voxel spacing | 0.7835821,0.7835821,5.2500005 | mm | Native header spacing; use affine for exact coordinates |
| Diffusion weights | 5,10,20,30,50,75,150,250,600,700,800 | s/mm² | Native acquisition order |
| Support | 2523 | voxels | Complete author dilated cord mask |
| Intensity convention | native magnitude, unnormalized | acquisition units | No target-derived calibration |
| Noise variance | not independently calibrated | acquisition units squared | No known Gaussian/Rician variance is assumed |
| Spatial boundary | voxelwise model | none | No padding or periodic extension |

## Method Hints

A high-weighting log-linear estimate can initialize tissue diffusion. Bounded nonlinear least squares then separates the two decay components; population-based optimization can help with competing local minima. Compare a joint fit against a segmented variant that fixes the high-weighting diffusion estimate. Explicitly handle vanishing vascular contributions and report the coupling between f and D*, rather than interpreting every fitted coefficient as equally identifiable.

## References

Lévy et al., spinal-cord intravoxel incoherent motion MRI, Magnetic Resonance in Medicine (2020), https://doi.org/10.1002/mrm.28195.
