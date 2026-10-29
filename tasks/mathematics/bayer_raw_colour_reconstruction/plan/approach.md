# Implemented mathematics

## Unknown, observations and normalization

The spatial target is native-grid, linear camera RGB x∈R^(2128×2840×3). The observed detector values d are uint16 containers carrying a10bit sensor signal. Native rows8:2136 and columns44:2884 are the camera's declared active area, not a content-dependent crop. The original LibRaw decoder gives RGGB at this active origin, black descriptors RGBG=(31,31,31,31)ADU and white w=1023ADU. The public array is exactly this full active sensor exposure; original raw DNG and metadata remain private. No new observations or synthetic benchmark are generated.

For pixel colour c use y=(d-bc)/(w-bc), with float64 arithmetic and no clipping. The second green black value equals the first in this instrument; preprocessing rejects unequal greens rather than silently changing the acquired normalization. Measurement x-to-y is the one-hot sampling operator Sx=Σc Mc xc, where three binary masks Mc partition every pixel. Its adjoint inserts each scalar observation at its known colour and zero elsewhere. The source row/column phase and sensor orientation are fixed. x is in normalized detector units, not absolute radiance, reflectance, gamma-encoded sRGB or a deblurred pre-optics object.

## Selected published reconstruction

The selected method is the direct5×5 gradient-corrected linear interpolation of Malvar, He and Cutler, as implemented in the remotely pinned BSD Colour Developers reference. This is a modern reference implementation, not a claim to possess the2004 authors' original executable. There is no optimized scalar objective, regularizer coefficient, initialization, step size, line search, update loop or stopping rule: each missing sample is evaluated once by its prescribed finite linear filter. Inventing convergence iterations would change the method's nature.

Let convolution of the scalar mosaic y with kernel K be K*y, centered at the current pixel. The four kernels are:

```math
K_G=\frac18\begin{pmatrix}0&0&-1&0&0\\0&0&2&0&0\\-1&2&4&2&-1\\0&0&2&0&0\\0&0&-1&0&0\end{pmatrix},
\quad
K_H=\frac18\begin{pmatrix}0&0&1/2&0&0\\0&-1&0&-1&0\\-1&4&5&4&-1\\0&-1&0&-1&0\\0&0&1/2&0&0\end{pmatrix},
```

```math
K_V=K_H^{\mathsf T},\qquad
K_D=\frac18\begin{pmatrix}0&0&-3/2&0&0\\0&2&0&2&0\\-3/2&0&6&0&-3/2\\0&2&0&2&0\\0&0&-3/2&0&0\end{pmatrix}.
```

G denotes green, H horizontal interpolation at the appropriate green site, V its transpose and D opposite red/blue colour. At measured green locations preserve y; elsewhere green is KG*y. At a red location preserve red and estimate blue with KD*y; at a blue location preserve blue and estimate red with KD*y. At green locations on a red-containing row, use KH*y for red and KV*y for blue. At green locations on a blue-containing row, reverse these choices. All convolutions use SciPy's mode='reflect', half-sample symmetric edge extension. The local mask-based implementation preserves original even/odd phase, float64 constants and native output pixel alignment.

The output can overshoot below0 or above1; no positivity projection, clipping, colour matrix or noise suppression is added. Defective pixels can spread into neighbouring reconstructed channels through the signed gradient filters. The constant/affine-colour reproduction property holds in the interior; reflected boundaries can violate smooth affine extrapolation. Tests independently check both sampling adjointness and affine spatial-field recovery away from the2pixel boundary.

## Actual comparison and scientific limits

The implemented original bilinear comparator convolves red/blue masked mosaics with [[1,2,1],[2,4,2],[1,2,1]]/4 and green with [[0,1,0],[1,4,1],[0,1,0]]/4, using the same reflective extension. Its boundary behavior is retained exactly. This comparator is actively computed in the notebook and independently compared to original outputs. It is not substituted as the benchmark method. No missing literature method or learned model is filled with zeros. The new notebook covers full-frame source/fresh comparisons, measured colour profiles, two native detail regions, signed method differences, kernel and impulse-response diagnostics. These are direct-method diagnostics, not fabricated iterative curves.

White-balance metadata is used only for review rendering: display gains are the first3values divided by the recorded green gain; one shared percentile-based exposure scale and gamma1/2.2 are applied to all displayed colour images. Native numeric reconstructions and Judge metrics always remain unclipped and unbalanced. Display subsampling is explicitly labeled and never changes the benchmark observations or reconstructed output.

## Evaluation

There is no independent physical RGB truth. The hidden reference is the successfully executed original Malvar result; a separate original rerun was bit-identical. NCC is the uncentered cosine of all flattened float64 channels. Dynamic-range NRMSE is sqrt(mean((x-v)^2))/(max(v)-min(v)), with v the reference. A zero prediction norm scores NCC0; a zero/constant reference or overflowing/nonfinite metric is rejected. Relative scaling, phase adjustment, cropping and magnitude conversion are prohibited. Every declared metric must pass.

Pre-implementation acceptance: NCC≥1−1e−10, NRMSE≤1e−10 and maximum absolute measured-channel discrepancy≤1e−12 normalized detector units. These strictly positive numerical tolerances accommodate floating reductions while the independent exact original rerun justifies a stringent boundary. The sampling criterion supplements method agreement but does not measure the missing-channel error. No physical accuracy claim is made from self-reference.
