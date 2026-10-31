# Pendant Drop Capillary Shape

> Recover the Bond number, pixel-scale capillary length and equilibrium silhouette of a photographed pendant water drop from its measured boundary despite pixel localization error and geometric uncertainty.
> Domain: mechanics | Keywords: capillary mechanics, inverse geometry, fluid interfaces | Difficulty: Hard

## Background

A liquid drop hanging from a needle deforms under gravity while surface tension resists curvature changes. The equilibrium interface balances hydrostatic pressure and capillary pressure. Its silhouette encodes how strong gravity is relative to the restoring surface force, making optical drop-shape measurements useful for studying liquid interfaces, emulsions and wetting.

Near-spherical drops are difficult to distinguish because their gravitational distortion is small. Image localization errors, camera tilt and incomplete visibility near the supporting needle further affect the inference. Absolute surface tension also requires a physical length calibration and density contrast; a photograph measured only in pixels cannot provide those quantities by itself.

## Problem Description

The acquisition is a real water-in-air pendant-drop camera image with an observed free-drop boundary extracted by image processing. The needle is excluded from the boundary. Infer the axisymmetric Young–Laplace equilibrium shape and the dimensionless gravitational-to-capillary ratio. **Absolute surface tension in N/m is not identifiable from the supplied calibration and is not requested.**

Let $R$ be apex curvature radius in pixels. Let $u=r/R$, $v=z/R$ and $s$ be dimensionless radial coordinate, upward height from the apex and signed arclength divided by $R$. Let $\phi$ be tangent angle and $B$ the Bond number. The equilibrium equations are

```math
\frac{du}{ds}=\cos\phi,\qquad
\frac{dv}{ds}=\sin\phi,\qquad
\frac{d\phi}{ds}=2-Bv-\frac{\sin\phi}{u}.
```

The apex has $u=v=\phi=0$, interpreted by the regular axisymmetric limit. Reflect the positive branch to obtain the other side. For apex image coordinates $(x_0,y_0)$ and image angle $\theta$, the predicted pixel boundary is

```math
x=x_0+R(u\cos\theta-v\sin\theta),\qquad
y=y_0-R(u\sin\theta+v\cos\theta).
```

Image rows increase downwards. The Bond number is $B=\Delta\rho gR_{\mathrm{physical}}^2/\gamma$, where $\Delta\rho$ is the liquid-minus-air density, $g$ gravitational acceleration, $\gamma$ surface tension and $R_{\mathrm{physical}}$ metric radius. Only $B$ and pixel radius are recoverable here. The capillary length in pixels is $L=R/\sqrt B$. An unknown pixel-to-length conversion leaves $\gamma$ undetermined even if density were supplied.

Save `output/reconstruction.npz` relative to the task workspace with:

| Key | Shape | Dtype | Description |
|---|---|---|---|
| parameters | (1, 5) | float32 or float64 | Batch then $(x_0,y_0,R,B,\theta)$; first three in pixels, Bond dimensionless, angle in radians. Radius and Bond must be positive. |
| capillary_length | (1, 1) | float32 or float64 | Positive $L=R/\sqrt B$ in pixels. |
| reconstruction | (1, 401, 2) | float32 or float64 | Batch, ordered silhouette sample, $(x,y)$ pixel coordinates in the original image; real values with no shift or scaling. |

The silhouette traverses from the left needle-side end through the bottom apex to the right end. Use 401 equally spaced signed dimensionless arclengths from $-s_{\max}$ to $s_{\max}$, where $s_{\max}$ is the greatest absolute closest-point arclength of the observed boundary for the fitted model. All293observed points participate. Coordinate order and sampling are part of the output contract.

Evaluation checks pixel geometry, Bond number, angle, capillary length, the complete silhouette, consistency with the equilibrium equation and proximity to the observed contour. There is no fitted alignment, scale adjustment or arbitrary point reordering. Numerical method agreement does not imply independently known surface tension or exact shape truth.

## Data Description

`data/raw_data.npz` preserves the complete grayscale camera frame and its processed measured contour. The contour is an observation, not a fitted shape. It was extracted before any Young–Laplace fitting; repeated coordinates from the native contour routine are retained. The image supplies acquisition context and permits independent inspection of boundary localization.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| image | (1, 1024, 1280) | uint8 | Batch, image row, image column; grayscale sensor intensity in digital counts0–255. |
| contour | (1, 293, 2) | float64 | Batch, observed boundary point, $(x,y)$ image coordinates in pixels; original extraction order. |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| System | Pendant water drop in air | — | Released author camera example; gravitationally deformed free interface. |
| Image dimensions | 1280 columns,1024rows | pixel | Full frame retained without resizing. |
| Coordinates | x rightward, y downward, origin upper-left | pixel | Native image-coordinate convention. |
| Free-drop region | x250:760,y190:720, half-open | pixel | Geometric selection excludes the rigid needle ending near row180 and contains the visible free droplet. |
| Observation extraction | Grayscale Gaussian3×3; Canny30/60; longest contour and native ordering | count/pixel | Author image-processing convention; numerical contour supplied directly. |
| Optical setup | Backlit CMOS camera, diffuse illumination | — | Published experimental setup; exact camera variant for this frame is not identified. |
| Symmetry | Axisymmetric equilibrium surface | — | Solid needle itself is not described by Young–Laplace free-surface equations. |
| Gravity direction | Downward, with fitted small in-plane tilt | — | Upward model height is opposite gravity. |
| Physical pixel size | Unavailable | m/pixel | No absolute surface tension can be inferred. |
| Density contrast and exact frame temperature | Not supplied | kg/m³ and degrees Celsius | Do not infer SI values from a nominal liquid or a fitted result. |
| Noise standard deviation | Unavailable | pixel | Pixel-edge localization is imperfect; no calibrated stochastic noise model. |
| Apex regularity | Zero radius/height/tangent angle in axisymmetric limit | — | Positive-branch solution reflected across symmetry axis. |

## Method Hints

Integrate the coupled shape and Bond-sensitivity equations, locate the closest arclength for each measured point, and adjust apex position, radius, Bond number and camera rotation using damped nonlinear least squares. A circle near the apex provides a geometric initialization. The signed normal-distance residual avoids assuming the boundary is a single-valued vertical graph. Check capillary-shape identifiability by comparing different visible height ranges and the spherical limit.

## References

1. J. D. Berry et al., *Measurement of surface and interfacial tension using pendant drop tensiometry*, Journal of Colloid and Interface Science454,226–237(2015). https://doi.org/10.1016/j.jcis.2015.05.012
2. E. Huang et al., *OpenDrop: Open-source software for pendant drop tensiometry & contact angle measurements*, Journal of Open Source Software6,2604(2021). https://doi.org/10.21105/joss.02604
