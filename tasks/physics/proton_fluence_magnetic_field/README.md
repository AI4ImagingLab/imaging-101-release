# Proton Fluence Magnetic Field

> Reconstruct both transverse path-integrated magnetic-field components from a proton-count image whose nonlinear geometric distortion couples fluence and particle deflection.
> Domain: physics | Keywords: charged-particle imaging, nonlinear transport, magnetic-field tomography | Difficulty: Hard

## Background

Fast protons crossing a magnetized plasma receive a transverse Lorentz impulse. A downstream detector records their positions. The resulting bright and dark structures encode spatial changes in magnetic deflection, making proton imaging useful for diagnosing magnetic structures in laboratory plasmas, fusion experiments and laboratory astrophysics.

A detector measures particle density rather than individual trajectories. Recovering the magnetic field therefore requires inferring a spatial transport map. In the regime where trajectories do not cross, particle conservation and a potential-flow approximation constrain this map. Once caustics form, a single fluence image generally cannot identify the underlying magnetic field uniquely.

## Problem Description

This task uses an existing author-supplied simulated proton radiograph. Recover a two-component spatial field over the complete observed region. In a right-handed laboratory frame, X points right, Y points up and Z points out of the detector toward the observer. The proton beam travels **into the detector, along negative Z**. Array coordinates x and y are the laboratory X and Y coordinates. The measured image is expressed in demagnified detector coordinates on the same grid. Let T be the map from object coordinates to those demagnified coordinates, w the transverse velocity impulse, M the geometric magnification, v the positive proton speed and rs the object-to-detector distance:

```math
\mathbf T(\mathbf x)=\mathbf x+\frac{r_s}{Mv}\mathbf w(\mathbf x),\qquad
M=\frac{r_s+r_i+l_i/2}{r_i+l_i/2}.
```

The path-integrated field components, denoted b(x,y), obey the selected small-angle cgs model. In vector form, the Lorentz impulse is w = (q/mpc) k cross b, where the beam unit vector k points along negative laboratory Z. In the stated laboratory X/Y components this gives

```math
b_x=-\frac{m_pc}{q}w_y,\qquad b_y=\frac{m_pc}{q}w_x.
```

Here mp is the proton mass, c the speed of light and q its positive charge. Each field component is an integral along the beam path, in gauss centimetres (G cm). It is not a three-dimensional local magnetic field. Incident illumination f0 is uniform and equals the mean measured counts per pixel. In the injective potential-map regime, T equals the gradient of a convex mapping potential Phi and particle conservation gives

```math
f(\nabla\Phi(\mathbf x))\det\nabla^2\Phi(\mathbf x)=f_0.
```

The measured image f is evaluated by bilinear interpolation. Exterior coordinates use the mean observed fluence. Zero normal deflection is imposed at the boundary. The spatially varying Jacobian makes the inverse nonlinear, and a weak-contrast approximation can introduce a systematic field-amplitude error.

Save **`output/reconstruction.npz`** relative to the task workspace. It must contain `reconstruction`, a real float32 or float64 array of shape **(1, 2, 200, 200)**. The axes are batch, component, x index, y index. Components 0 and 1 are the laboratory X and Y field components, respectively, in **G cm**, with the signs fixed by the into-detector beam convention above. Use the complete supplied pixel grid and absolute physical amplitude. No translation, rotation, scale fitting, sign adjustment or cropping is applied during evaluation.

Each component is evaluated separately using cosine NCC and dynamic-range NRMSE. NCC is the flattened inner product divided by the two vector norms; a zero estimate has NCC zero. NRMSE is the root-mean-square error divided by the reference component's maximum minus minimum. Both components must meet their respective accuracy requirements.

## Data Description

All observations and coordinate arrays are in `data/raw_data.npz`. The stored float64 count array preserves the supplied integer-valued measurements without changing precision.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| x | (1, 200, 200) | float64 | Object-plane x coordinate; cm; varies on the first spatial axis |
| y | (1, 200, 200) | float64 | Object-plane y coordinate; cm; varies on the second spatial axis |
| fluence | (1, 200, 200) | float64 | Simulated detected proton counts per demagnified pixel; counts/pixel |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Coordinate convention | x first/right, y second/up; laboratory Z out of detector | — | Right-handed laboratory axes |
| Beam direction | Negative laboratory Z, into detector | — | The beam direction and display x/y axes do not themselves form a right-handed basis |
| Object-grid spacing | 0.001 on both axes | cm | Half-pixel-centred 200 by 200 grid |
| First/last coordinate | -0.0995 / 0.0995 | cm | Pixel-centre endpoints on both axes |
| Proton energy | 14.7 | MeV | Author acquisition label |
| Proton speed v | 5.24e9 | cm/s | Supplied rounded speed used directly in this model |
| Source-to-volume entrance ri | 1.3 | cm | Geometric source distance |
| Volume extent li | 0.2 | cm | Geometric beam-direction extent |
| Volume-to-detector distance rs | 20 | cm | Drift distance used in the mapping |
| Proton mass mp | 1.67e-24 | g | Supplied cgs conversion constant |
| Proton charge q | 4.803e-10 | statC | Supplied cgs conversion constant |
| Light speed c | 3.0e10 | cm/s | Supplied cgs conversion constant |
| Incident illumination | Spatially uniform, equal to mean(fluence) | counts/pixel | Equal total incident and measured count normalization |
| Boundary condition | Zero normal velocity impulse | cm/s | Identity normal component of the transport map on each edge |
| Exterior fluence | mean(fluence) | counts/pixel | Bilinear-interpolation fill value outside the supplied coordinates |
| Noise | Finite-particle count fluctuations in existing simulation | counts | No additional Gaussian-noise variance is supplied |
| Field model | Small-angle magnetic deflection, injective potential map | — | No electric-field contribution or loss compensation is inferred |

## Method Hints

A nonlinear Monge–Ampère transport reconstruction can estimate the convex mapping potential. A log-parabolic relaxation uses the discrepancy between transported observed fluence and uniform incident fluence as its update direction, with Neumann boundaries representing zero normal deflection. The inferred transport displacement gives both magnetic components through the Lorentz rotation. A linear Poisson inversion provides a useful weak-contrast comparison, while positivity of the mapping Jacobian and count conservation provide physical checks.

## References

- A. F. A. Bott, C. Graziani, P. Tzeferacos, T. G. White, D. Q. Lamb, G. Gregori and A. A. Schekochihin, *Proton imaging of stochastic magnetic fields*, Journal of Plasma Physics 83, 905830614 (2017). [Published article](https://doi.org/10.1017/S0022377817000939), [author manuscript](https://arxiv.org/abs/1708.01738).
