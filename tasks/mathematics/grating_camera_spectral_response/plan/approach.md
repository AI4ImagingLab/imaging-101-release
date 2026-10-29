# Implemented spectral calibration

The selected method is the author's real-data LED plus fluorescent estimator, run independently for all four released cameras. All calculations use float64 after loading float32 image observations. Pixel medians produce profiles: the direct ROI is reduced across both spatial axes; diffracted and fluorescent images are reduced across rows. Direct and diffracted LED profiles share division by the largest direct RGB value. Fluorescent profiles retain their native relative scale.

For each RGB channel, normalize each of the 44 population spectra by its channel maximum. Form the uncentred 31-by-31 Gram matrix and retain the seven descending-eigenvalue eigenvectors. There is no centring. The population channel mean predicts fluorescent peak locations. Match the two strongest measured and predicted samples in each channel (six pairs total); solve a quadratic pixel-to-spectral-bin least-squares fit. This is peak matching, not peak detection with a local-maximum criterion.

Let P be the 321-by-31 interpolation matrix, Bc the seven-mode channel basis, L the known 321-vector LED spectrum, K the pixel-by-321 linear-splat matrix, and E the 321-by-7 reciprocal-attenuation basis. E columns are constant, sine harmonics 1--3, cosine harmonics 1--3 on the inclusive phase interval 0--2pi. Define ac as the pseudoinverse of K diag(L) applied to the measured dispersed channel. With coefficient vectors h for reciprocal attenuation and bc for sensitivity,

```math
A_c=[\operatorname{diag}(a_c)E,\ -PB_c],\quad s_c=B_cb_c,\quad a=(Eh+10^{-12})^{-1}.
```

The second block occupies the channel-specific location in a stacked block-diagonal matrix. Each residual row is divided by its Euclidean norm plus epsilon=1e-12. Direct constraints are L-transpose P Bc bc=dc, each divided by the norm of its augmented coefficient/right-hand-side row plus epsilon. There are three constraints, one per channel. Solve min one-half squared norm of A x subject to Cx=d using

```math
\begin{bmatrix}A^T A&C^T\\C&0\end{bmatrix}
\begin{bmatrix}x\\\mu\end{bmatrix}
=\begin{bmatrix}0\\d\end{bmatrix}.
```

x concatenates h and the three bc vectors; mu contains three Lagrange multipliers. Preserve the original final row normalization of the augmented KKT matrix and explicit matrix inverse multiplied by its right-hand side. This closed-form estimator has no artificial iterations, regularization multiplier or convergence threshold. Normalize the resulting 31-by-3 sensitivity by its global maximum. No clipping, channel-wise scaling, target fitting or use of independent measurements occurs in reconstruction.

The optional LED-only comparison initializes reciprocal attenuation as a least-squares constant and uses 500 fixed point-to-plane ICP updates. Each RGB curve is peak normalized; nearest neighbours are found in (spectral-bin, relative-signal) coordinates. Tangents use consecutive target differences, duplicating the final tangent; normals are their 90-degree rotations. A quadratic mapping starts with slope 321 divided by profile length and zero offset. Every update solves the stacked linear point-to-plane equations; no early stopping is added. Other comparisons change sensitivity basis size to 5 or 9, or attenuation basis size to 3 or 5, preserving the selected estimator.

Forward diagnostics compute L-transpose P s and K diag(L*a) P s from unnormalized sensitivity. The unconstrained attenuation can be negative, notably for Sony; it is a nuisance fit, not a validated physical efficiency. Evaluation uses independent measured response samples exactly at 400--700 nm, normalized by their joint RGB maximum. The paper's pseudoinverse-projected extended-grid relative-error reporting is a different convention and must not be called dynamic-range NRMSE. Thresholds were frozen from independent original runs before extraction: per-camera NCC at least 0.9 times original NCC, NRMSE at most 1.1 times original NRMSE; peak deviation at most 1e-6.
