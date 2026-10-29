# Approach: KLIP + Angular Differential Imaging (ADI)

## Problem Summary

We want to detect and characterise faint companions (planets, brown dwarfs)
around nearby stars from ground-based coronagraphic or non-coronagraphic
high-contrast imaging.  The dominant noise source is quasi-static speckle
noise — residual wavefront errors that produce PSF patterns resembling
point-like companions at $10^{-4}$–$10^{-6}$ contrast relative to the star.

The key observational strategy is **Angular Differential Imaging (ADI)**:
the telescope pupil is kept fixed while the sky (and any companions) rotates
as the Earth turns.  Quasi-static speckles stay fixed in detector coordinates
while companions sweep through an arc of $\Delta\theta = \theta_{\max} -
\theta_{\min}$ degrees over the observation.

## Algorithm: KLIP (Karhunen-Loève Image Processing)

**Reference**: Soummer, Pueyo & Larkin (2012), ApJL 755 L28.

### Step 1 – Reference library

In full-frame ADI KLIP (as used here), the entire image cube serves as its
own reference library.  Each frame is a potential reference for every other
frame.  (An angular exclusion criterion can be applied to exclude frames
where a companion at the target separation would have rotated less than
$\sim 0.5$ FWHM; we omit this for simplicity.)

### Step 2 – KL basis construction

Given the $N \times n_{pix}$ matrix $R$ of mean-subtracted, flattened
reference frames:

$$Z^{KL} = \text{top-}K\text{ right singular vectors of }R$$

Equivalently, diagonalise the covariance $RR^T$ and project back to pixel
space (the original KLIP formulation):

$$\Lambda, C = \text{eigh}(R R^T), \qquad Z^{KL}_k = \frac{R^T c_k}{\sqrt{\lambda_k}}$$

We use SVD by default (numerically equivalent, faster for tall-and-thin $R$).

### Step 3 – PSF subtraction

For each frame $T_k$ (mean-subtracted, NaN-replaced):

$$\hat{I}_{\psi,k} = \sum_{j=1}^{K} \langle T_k, Z^{KL}_j \rangle Z^{KL}_j$$

$$A_k = T_k - \hat{I}_{\psi,k}$$

The truncation to $K$ modes is the regularisation: large $K$ removes more
speckle noise but also attenuates the companion signal (self-subtraction).

### Step 4 – Derotation

Rotate each residual frame $A_k$ by $-\theta_k$ (parallactic angle) to align
companions to a common sky orientation.  Implemented as a batched bilinear
grid_sample over all frames simultaneously.

### Step 5 – Temporal combination

Average (or median) the derotated cube:

$$\hat{A} = \frac{1}{N} \sum_k \text{rot}(A_k, -\theta_k)$$

The resulting image is the final detection map.  Companions appear as
positive point sources; the background is dominated by residual speckle noise
and read noise, approximately Gaussian after combination.

### Step 6 – SNR

The companion SNR is measured using the Mawet et al. (2014) two-sample t-test:

$$\text{SNR} = \frac{s - \bar{n}}{\sigma_n \sqrt{1 + 1/N_n}}$$

where $s$ is the median within an aperture of diameter FWHM centred on the
companion, $\bar{n}$ and $\sigma_n$ are the mean and standard deviation of
$N_n$ noise apertures placed at the same separation.

## Key hyperparameters

| Parameter   | Value(s) | Role |
|-------------|----------|------|
| K_klip      | 10, 20   | KL truncation; higher = more speckle removal but more self-subtraction |
| method      | svd      | KL basis algorithm (svd / pca / eigh, all equivalent) |
| statistic   | mean     | Temporal combination (mean or median) |
| iwa         | 4 px     | Inner working angle (coronagraph edge) |

## Expected results

Beta Pictoris b is a well-known directly imaged planet at ~0.44" (16 px)
separation.  With $K = 10$–$20$ modes, SNR $\gtrsim 5$ is expected in the
VLT/NACO L'-band data used here.
