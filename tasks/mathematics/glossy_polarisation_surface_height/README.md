# Glossy Polarisation Surface Height

> Reconstruct a glossy object's surface height from images recorded through a rotating linear polariser despite ambiguous surface orientation and mixed diffuse and specular reflection.
> Domain: mathematics | Keywords: inverse geometry, polarimetric imaging, surface reconstruction | Difficulty: Hard

## Background

Reflection from a dielectric surface partially polarises initially unpolarised light. A linear polariser in front of a camera transmits different intensities as its transmission axis rotates. The resulting intensity modulation contains information about the direction in which the surface faces the camera. This gives a passive geometric cue for objects whose texture is unsuitable for feature matching.

Polarimetric geometry is useful for inspecting glossy manufactured objects, documenting cultural artefacts and recovering shape under weak texture. Its principal difficulties are physical ambiguities: the same polarisation can correspond to multiple surface orientations, and diffuse and specular reflection obey different Fresnel laws. Spatial consistency and calibrated illumination help constrain these ambiguities.

## Problem Description

The supplied orthographic image stack obeys a per-pixel polarimetric modulation model:

```math
I(r,c,\alpha)=U(r,c)\left[1+\rho(r,c)\cos\bigl(2\alpha-2\phi(r,c)\bigr)\right]+\epsilon(r,c,\alpha).
```

Here $r,c$ are image row and column, $\alpha$ is the analyser angle in radians, $I$ is the observed normalised intensity, $U$ is the unpolarised mean intensity, $\rho$ is the degree of linear polarisation, $\phi$ is its phase modulo $\pi$, and $\epsilon$ includes measurement noise and model mismatch. No noise variance is supplied. The foreground and specular-region masks accompany the acquisition.

Let $z$ denote the unknown height in pixel units, $x$ the coordinate increasing to image-right and $y$ the coordinate increasing toward image-up. Define slopes $p=\partial z/\partial x$ and $q=\partial z/\partial y$. The unit normal is $\boldsymbol\nu=(-p,-q,1)/\sqrt{1+p^2+q^2}$. Its zenith angle $\theta$ satisfies $\cos\theta=\nu_z$. For diffuse pixels the index of refraction $\eta$ determines the Fresnel relation

```math
\rho_d(\theta)=\frac{(\eta-\eta^{-1})^2\sin^2\theta}{2+2\eta^2-(\eta+\eta^{-1})^2\sin^2\theta+4\cos\theta\sqrt{\eta^2-\sin^2\theta}}.
```

For specular pixels the relation is

```math
\rho_s(\theta)=\frac{2\sin^2\theta\cos\theta\sqrt{\eta^2-\sin^2\theta}}{\eta^2-\sin^2\theta-\eta^2\sin^2\theta+2\sin^4\theta}.
```

Diffuse phase determines the azimuth only modulo $\pi$; specular phase is offset by $\pi/2$. The diffuse shading model is $U=\boldsymbol\ell\cdot\boldsymbol\nu$, where $\boldsymbol\ell$ combines calibrated directional illumination with unknown uniform albedo and light intensity. The selected published approximation constrains specular normals toward $\boldsymbol h=(\boldsymbol\ell+\boldsymbol v)/\lVert\boldsymbol\ell+\boldsymbol v\rVert$, where $\boldsymbol v=(0,0,1)$ is the camera direction. This convention uses the intensity-scaled light vector directly and is scale-dependent; it differs from a halfway construction using a unit light direction. Preserve this explicit model convention when comparing standard-method reconstructions. These are approximate single-reflection models; the data contain departures from them. Height also has an additive gauge ambiguity.

Save **`output/reconstruction.npz`**, relative to the task workspace, with exactly one key: `reconstruction`. Its shape is **`(1, 457, 235)`**, its dtype must be real **float32 or float64**, and its axis order is **batch, image row, image column**. Values represent height in **pixel units** under orthographic unit pixel spacing; no conversion to millimetres is available. All values must be finite. Set pixels outside the public foreground mask to zero. Fix the first foreground pixel in row-major order to height zero. Positive height follows the normal/slopes convention above; do not negate or independently scale the surface.

Evaluation measures cosine NCC and dynamic-range NRMSE over exactly the public foreground mask. For flattened masked prediction $a$ and reference $b$, NCC is $a\cdot b/(\lVert a\rVert\lVert b\rVert)$ and NRMSE is $\sqrt{\operatorname{mean}((a-b)^2)}/(\max b-\min b)$. There is no translation, reflection, scale, crop or phase alignment. Reference agreement measures reproduction of a standard geometric inference; it does not establish independently measured physical accuracy.

## Data Description

All observations and masks are in `data/raw_data.npz`.

| Key | Shape | Dtype | Description |
| --- | --- | --- | --- |
| images | (1, 457, 235, 36) | float64 | Captured normalised analyser-image intensities, dimensionless; axes batch, row, column, angle. |
| angles | (1, 36) | float64 | Analyser angles in radians, in the same order as the image-stack angle axis. |
| mask | (1, 457, 235) | bool | Foreground region available with the acquisition; dimensionless indicator. |
| specular_mask | (1, 457, 235) | bool | Specular region used for the reflection model; dimensionless indicator, contained in the foreground mask. |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
| --- | --- | --- | --- |
| Projection | Orthographic | dimensionless | No perspective depth scaling. |
| Pixel spacing | 1 | pixel | Common spacing for image x, image y and reconstructed height. |
| Refractive index | 1.5 | dimensionless | Assumed dielectric refractive index. |
| Light direction | (2, 0, 7) | dimensionless | Known unnormalised vector in right, up, camera-facing coordinates; intensity/albedo scale is unknown. |
| Camera direction | (0, 0, 1) | dimensionless | Direction from surface toward camera. |
| Analyser convention | Clockwise from image-up, viewed into camera lens | radian | Zero and sign convention for angles. |
| Image axes | Rows increase downward; columns increase rightward | pixel | Geometric y points opposite the row direction. |
| Radiometric scale | Normalised author observations, unchanged float64 values | dimensionless | Do not apply display gamma to the stack. |
| Boundary condition | Masked domain; no height information outside the foreground | pixel | Foreground is not a periodic domain. |
| Height gauge | First row-major foreground pixel equals zero | pixel | Removes additive integration ambiguity. |
| Noise | Not independently calibrated | intensity unit | Avoid interpreting a fitting residual as calibrated chi-square. |

## Method Hints

The conditional linear height estimate uses unnormalized equal-weight squared residuals of the phase and shading equations, plus 0.01 times the sum of squared five-point height Laplacians wherever all four axial foreground neighbors exist. It fixes the first row-major foreground height to zero. For diffuse shading use min(1,U)/cos(theta)-ell_z on the right side of the ratio equation; clipping is applied here only, after illumination estimation. Each specular pixel supplies p=-h_x/h_z and q=-h_y/h_z in addition to its corrected phase equation.

The discrete x derivative, at a pixel with all eight neighbors, is right-column weights (1,4,1)/12 minus the same left-column weights; the y derivative is upper-row weights (1,4,1)/12 minus lower-row weights. If only axial neighbors are complete use centered axial differences. Otherwise use a centered derivative when both opposite neighbors exist, or a one-sided derivative toward the existing neighbor. Omit a pixel's photometric equations only when either axis lacks all neighbors; do not wrap across the foreground boundary. These stencils define the evaluated discrete target rather than an arbitrary surface-integration convention.

For the known lighting direction s, its unknown scalar intensity is fitted using diffuse pixels. Let b=[sin(phi)sin(theta),cos(phi)sin(theta),cos(theta)] and T=diag(-1,-1,1). The conditional lighting objective is the sum over diffuse pixels of min((b dot ell-U)²,(b T dot ell-U)²), restricted to ell along s. This expresses the phase ambiguity in the fit without prescribing alternating-update counts. For the specular Fresnel inverse use the increasing branch, tabulated at theta=0,0.01,... radians below pi/2 and linearly interpolated to measured degree of polarization; this is the selected finite angular representation.

Fit a sinusoid at each foreground pixel to recover mean intensity, degree of polarisation and phase. The Fresnel model supplies a zenith angle; use the appropriate diffuse or specular branch. A common light direction permits alternating assignment of the diffuse azimuth ambiguity and estimation of the scalar illumination/albedo factor.

Direct height inference can combine phase and shading constraints in a sparse linear system. For diffuse pixels, the geometric relations include

```math
\cos\phi\,p-\sin\phi\,q=0,
\qquad -\ell_x p-\ell_y q=U/\cos\theta-\ell_z.
```

Here $\ell_x,\ell_y,\ell_z$ are the components of $\boldsymbol\ell$. Apply the specular phase correction before using its phase constraint. Smoothness of the height surface and a fixed integration gauge help make the spatial inverse problem well posed; boundary stencils must respect the foreground mask. A direct height formulation avoids choosing every pixel's ambiguous normal independently.

The numerical standard uses the following conventions so that this regularized inverse has a defined target. Fit the sinusoid by ordinary least squares. Invert the diffuse Fresnel law on its real physical branch; invert the increasing specular branch by linear interpolation of samples spaced 0.01 rad, ending at its maximum. On diffuse pixels, candidate normals are $(\sin\phi\sin\theta,\cos\phi\sin\theta,\cos\theta)$ and the alternative with its first two signs reversed. Starting with the supplied light vector, alternate the lower shading-residual ambiguity and the least-squares scalar light factor until convergence. Use unmodified mean intensity for this scalar fit and $\min(U,1)$ for the height equations.

Measured diffuse polarisation can exceed the exact Fresnel model's range. For a defined numerical convention, use the analytic diffuse inverse below, including its stated real-part rule, rather than replacing such pixels by a different clipped or fitted model. Define $R=\sqrt{(1-\rho)(1+\rho)}$ using complex intermediate arithmetic when needed, and

```math
A=2\rho+2\eta^2\rho-2\eta^2+\eta^4+\rho^2+4\eta^2\rho^2-\eta^4\rho^2-4\eta^3\rho R+1,
```
```math
B=\eta^4\rho^2+2\eta^4\rho+\eta^4+6\eta^2\rho^2+4\eta^2\rho-2\eta^2+\rho^2+2\rho+1,\qquad\theta=\arccos\!\left(\min\!\left(\operatorname{Re}\sqrt{A/B},1\right)\right).
```

This extension defines the published numerical approximation beyond ideal-model consistency; it is not evidence that every measured diffuse pixel follows an exact Fresnel law.

For specular pixels, replace $\phi$ by $(\phi+\pi/2)\bmod\pi$ and include both slope constraints $p=-h_x/h_z$, $q=-h_y/h_z$ along with the phase constraint, where $h_x,h_y,h_z$ are the components of the approximate halfway vector already defined. Minimize the sum of squared phase, shading and specular-slope residuals, plus $\sum(0.1\Delta z)^2$ and the squared zero-height gauge residual. Here $\Delta z$ is the unnormalised five-point Laplacian on pixels with four foreground axial neighbours. The factor 0.1 multiplies the Laplacian residual, so its squared objective weight is 0.01.

On complete 3-by-3 foreground neighbourhoods, the derivative kernels for $p$ and $q$ are respectively

```math
D_x=\frac1{12}\begin{pmatrix}-1&0&1\\-4&0&4\\-1&0&1\end{pmatrix},\qquad
D_y=\frac1{12}\begin{pmatrix}1&4&1\\0&0&0\\-1&-4&-1\end{pmatrix}.
```

At other pixels use centred axial differences when both neighbours exist and first-order one-sided differences otherwise. A data equation requires at least one foreground neighbour along each axis. The gauge anchors the first row-major foreground pixel; no boundary height is imposed. These fixed discretization and weighting conventions identify the selected standard inference without assuming access to a particular solver implementation.

## References

William A. P. Smith, Ravi Ramamoorthi and Silvia Tozza. *Linear depth estimation from an uncalibrated, monocular polarisation image*. European Conference on Computer Vision, 2016. Smith, Ramamoorthi and Tozza, ECCV 2016 paper.
