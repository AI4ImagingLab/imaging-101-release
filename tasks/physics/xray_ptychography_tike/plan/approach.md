# Solution Approach: X-ray Ptychography Reconstruction

## Problem Statement

Given a set of 516 far-field diffraction patterns collected by raster-scanning a focused X-ray probe across a Siemens star test sample, reconstruct the complex-valued transmission function of the object. The detector records only intensity (squared magnitude of the wavefield), so all phase information is lost. The overlapping illumination regions between adjacent scan positions provide the redundancy needed to recover the missing phase.

## Forward Model

The forward model for ptychography at scan position $j$ is:

1. **Extract patch**: $O_j = \psi[\mathbf{r}_j : \mathbf{r}_j + W]$ -- extract a $W \times H$ patch from the object at position $\mathbf{r}_j$
2. **Apply probe**: $\phi_j = P \cdot O_j$ -- element-wise multiplication of probe and object patch (exit wave)
3. **Propagate**: $\Phi_j = \mathcal{F}[\phi_j]$ -- 2D FFT for far-field (Fraunhofer) propagation
4. **Detect**: $I_j = |\Phi_j|^2$ -- squared magnitude (intensity measurement)

The inverse problem minimizes:

$$\mathcal{L}(\psi, P) = \sum_{j=1}^{N} \| \sqrt{I_j} - |\mathcal{F}[P \cdot \psi_j]| \|^2$$

## Iterative Ptychographic Engine

The reconstruction uses an iterative gradient-based engine. At each iteration:

1. **Batch selection**: Divide the 516 scan positions into 7 mini-batches using the `wobbly_center` clustering method. This provides stochastic acceleration and reduces memory per step.

2. **For each batch**:
   - Compute the exit wave: $\phi_j = P \cdot O_j$
   - Propagate to detector plane: $\Phi_j = \mathcal{F}[\phi_j]$
   - Compute amplitude mismatch: replace $|\Phi_j|$ with $\sqrt{I_j^{\text{meas}}}$ while keeping the phase
   - Back-propagate the corrected wavefield
   - Update the object $\psi$ and probe $P$ using gradient-based least-squares steps

3. **Rescaling**: Every 10 epochs, rescale the object so that the mean of its absolute value is approximately 1.0. This stabilizes the joint object-probe optimization.

## Algorithm: lstsq_grad

The `lstsq_grad` algorithm (least-squares gradient) computes analytic gradients of the intensity-matching objective with respect to both the object and the probe. It uses an automatic step-size selection based on the local curvature (Hessian diagonal approximation), which avoids manual tuning of learning rates.

Key advantages over simpler methods (e.g., ePIE, rPIE):
- Uses all positions in a batch simultaneously for the update (not sequential position-by-position)
- Automatic step size via least-squares fitting
- Compatible with adaptive moment estimation for the object update

## Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `num_iter` | 64 | Sufficient for convergence on this dataset |
| `num_batch` | 7 | ~74 positions per batch; balances memory and stochastic acceleration |
| `n_modes` | 1 | Single coherent mode (sufficient for this dataset) |
| `psi_init` | 0.5+0j | Uniform complex initialization; amplitude ~0.5 avoids scaling instabilities |
| `scan_offset` | 20 px | Buffer so no scan position maps to edge of psi array |
| `object_options.use_adaptive_moment` | True | Adam-like momentum for object updates |

## Expected Results

- **Cost convergence**: The objective function decreases from approximately 2.0 at iteration 1 to approximately 0.33 after 64 iterations
- **Object**: The Siemens star pattern should be clearly resolved in the phase image, with spoke features visible down to the resolution limit
- **Probe**: The refined probe should show a focused Airy-like pattern, matching the Fresnel zone plate illumination used at the Velociprobe
