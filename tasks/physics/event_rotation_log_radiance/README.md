# Event Rotation Log Radiance

> Reconstruct a relative log-radiance image from a real event-camera stream and independent rotational motion, despite directional-gradient ambiguity and sparse asynchronous measurements.
> Domain: physics | Keywords: event imaging, variational inversion, inertial sensing | Difficulty: Hard

## Background

An event camera reports the time, position and sign of a local brightness change instead of recording complete intensity frames. Its pixels respond asynchronously, preserving fast changes with fine temporal resolution while avoiding much of the redundancy of conventional video. These sensors support robotics, autonomous navigation and imaging in scenes with large illumination contrasts.

Events encode changes in log radiance rather than radiance itself. A static edge can produce many events during camera movement, whereas a nearly uniform region can produce almost none. Camera motion therefore determines which spatial intensity gradients are observed. Recovering an image requires combining these incomplete, direction-dependent constraints with spatial regularity and a stated brightness convention.

## Problem Description

The observations depict a static indoor scene viewed during camera rotation. Reconstruct the log-radiance image at the last event time. An event at position x and time t has polarity p=+1 for increasing log radiance and p=−1 for decreasing log radiance. For local log radiance L, the sensor model is


```math
L(\boldsymbol{x},t)-L(\boldsymbol{x},t^-)=pC,
```


where t− is the previous event time at that pixel and C is the positive contrast threshold. The absolute contrast C and additive brightness level are not independently calibrated. The image is consequently expressed in the normalized relative-log convention specified below.

Let ω=(ωx,ωy,ωz) be the independently measured angular velocity in the camera frame. At pixel (x,y), write u=x−cx and v=y−cy, where fx,fy are focal lengths and cx,cy are principal-point coordinates. The image velocity is


```math
\begin{aligned}
\dot{x}&=(uv/f_y)\omega_x-(u^2/f_x+f_x)\omega_y+(f_xv/f_y)\omega_z,\\
\dot{y}&=(v^2/f_y+f_y)\omega_x-(uv/f_x)\omega_y-(f_yu/f_x)\omega_z.
\end{aligned}
```


Velocities are in pixel/s and ω is in rad/s. The short-window approximation treats this field as constant in time. If Δt is the first-to-last event duration, d=(ẋ,ẏ)Δt is displacement in pixels. The normalized event time is τ=(t−tfirst)/Δt. An event is transported to the final image location as x′=x+(1−τ)d, sampling the displacement at that event's two-dimensional pixel location. Let W be unit-area bilinear deposition onto the image grid. The signed image of warped events is


```math
H(\boldsymbol{x})=\sum_k p_k W(\boldsymbol{x}-\boldsymbol{x}'_k),
\qquad
b(\boldsymbol{x})=H(\boldsymbol{x})/\lVert\boldsymbol{d}(\boldsymbol{x})\rVert.
```


Here k indexes events, H is in signed event counts per pixel and b is in event-count units per unit pixel displacement. With unit contrast, the directional constraint is b≈Dℓ, where ℓ is the grid log image, û=d/|d| is unit motion and


```math
(D\ell)(\boldsymbol{x})=\ell(\boldsymbol{x})-
\ell(\boldsymbol{x}+\hat{\boldsymbol{u}}(\boldsymbol{x})).
```


The displaced value is bilinearly sampled. The outermost sensor row and column have zero unit-flow and zero b, so no observation constraint is imposed there. Regions with few events and gradients perpendicular to motion are weakly constrained. Sensor noise, a finite acquisition interval and the constant-motion approximation prevent exact consistency.

Save `output/reconstruction.npz`, relative to the task workspace, with required key `reconstruction`: real float64, shape `(1, 180, 240)`, axes batch/downward-y/rightward-x, dimensionless relative log radiance. The output gauge is: for a reconstructed log image ℓ, let a and z be its1st and99th image percentiles; output clip((ℓ−a)/(z−a),0,1). This specifies a relative image, not calibrated linear radiance. All180×240pixels are evaluated; no crop, spatial alignment, fitted gain or reference-dependent normalization is applied to image comparison.

Evaluation version 2 uses the public-observation residual and a spatial-regularity class instead of identity with a particular iterative result. Write A for the bilinear directional operator D above and y for the submitted image. The observation residual is ||Ay-cb||_2/||Ay||_2, with c=max(0,(Ay)^T b/(b^T b)); it may not exceed 0.1496703024441773. A constant image has residual 1. The centered anisotropic total variation is

```math
T(y)=\tfrac12\sum_{i=0}^{177}\sum_j|y_{i+2,j}-y_{i,j}|+\tfrac12\sum_i\sum_{j=0}^{237}|y_{i,j+2}-y_{i,j}|.
```

Require T(y)<=2090.431945645451. This is the independently executed feasible standard regularity value with relative 1e-8 numerical allowance, not a pixelwise target. The output must lie in [0,1], with at least431 exact zero and431 exact unit samples; this is the declared percentile gauge. No intensity-reference comparison or fitted spatial alignment is performed. These criteria define a noise-compatible, TV-regular image class; they do not claim unique calibrated radiance.

## Data Description

These are real DAVIS240C measurements from an indoor boxes scene, with acquisition-only camera rectification. Native event timing and polarity are preserved; floating positions result from independent lens calibration. No complete intensity image or optical flow inferred from an intensity target is supplied.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| events | (1, 27643, 4) | float64 | Rows are [time in s, x in pixel, y in pixel, polarity0/1 dimensionless]; chronological order;0 maps to−1 and1 to+1. |
| angular_velocity | (1, 3) | float64 | Independent IMU gyro [ωx,ωy,ωz], rad/s, in the camera frame. |
| intrinsics | (1, 3, 3) | float64 | Rectified pinhole matrix K; focal lengths and principal-point entries in pixels, homogeneous last row dimensionless. |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Sensor height, width | 180,240 | pixel | Camera array, y downward and x rightward; coordinate origin is upper left. |
| Event start | 19.500000000 | s | Time in the source recording. |
| Last event minus first event | 0.016518998999998757 | s | Actual stored acquisition duration; compute it from the event timestamps. |
| Raw acquisition selection | First30000events at or after19.5s | events | Rectification and calibrated sensor-interior selection retain27643events. |
| Rectified focal lengths | 169.50404059859204,179.58096603591352 | pixel | fx,fy, matching the supplied rectified event positions. |
| Rectified principal point | 136.0468660434908,114.28206073946285 | pixel | cx,cy; also stored in K. |
| Gyro sampling convention | First IMU record at or after the median retained event time | dimensionless convention | Provides the independent short-window constant angular velocity. |
| Camera-frame gyro axes | x right,y down,z forward | dimensionless orientation | Right-handed camera coordinates and the velocity signs specified above. |
| Grid precision convention | float32 grid, focal lengths and principal point; float64 gyro/time | dimensionless convention | Preserves the original camera-flow arithmetic while retaining stored input precision. |
| Event flow lookup | floor y and floor x separately, then row-major flatten | dimensionless convention | Two-dimensional lower-neighbor sampling; warped positions remain fractional. |
| Warp boundary | Discard interpolation weights outside the sensor | dimensionless convention | Bilinear weights inside the sensor are retained without renormalization. |
| Directional-model boundary | Unit-flow and b zero on the outermost row/column | dimensionless convention | No edge observation equations; image comparison still includes every pixel. |
| Event contrast | Unknown positive, common scale | relative log-radiance unit | Removed by the declared output convention; no absolute brightness claim. |
| Data license | CC BY-NC-SA3.0 | license identifier | Attributed noncommercial research data, with acquisition-only derivative changes described in NOTICE. |

## Method Hints

A useful family of methods first forms a motion-compensated signed event image, then reconstructs a log image from sparse directional difference constraints. An anisotropic total-variation penalty can propagate edge information through regions with few events while retaining sharp boundaries. Split Bregman separates the nonsmooth gradient penalty from a linear least-squares subproblem. A Laplacian quadratic penalty is a useful smoother comparison. Respect the distinction between image velocity for a seconds-based trajectory and window displacement for the normalized event constraint; both must use the same clock convention.

## References

1. Z.Zhang,A.Yezzi,G.Gallego. *Formulating Event-based Image Reconstruction as a Linear Inverse Problem with Deep Regularization using Optical Flow*. IEEE TPAMI45(7),8372–8389(2023). https://doi.org/10.1109/TPAMI.2022.3230727. Selected method: its classical variational reconstruction family; the learned-denoiser variant is outside this task's selected method.
2. E.Mueggler,H.Rebecq,G.Gallego,T.Delbruck,D.Scaramuzza. *The Event-Camera Dataset and Simulator*. IJRR36(2),142–149(2017). https://doi.org/10.1177/0278364917691115. Acquisition data: https://rpg.ifi.uzh.ch/davis_data.html.
