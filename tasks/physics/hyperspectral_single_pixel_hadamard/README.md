# Hyperspectral Single-Pixel Hadamard Imaging

> Reconstruct a full spatial–spectral tomato-slice image cube from complementary bucket-detector spectra, resolving spatial detail despite multiplexing, detector background and measurement noise.
> Domain: physics | Keywords: spectral imaging, coded imaging, linear inverse problems | Difficulty: Medium

## Background

A single-pixel spectral camera measures light collected from an entire illuminated field rather than from an array of image-forming detectors. A spatial light modulator displays known binary patterns, while a spectrometer records the integrated light at many wavelengths for each pattern. Spatial structure is encoded in changes between successive spectra. This arrangement makes it possible to collect a densely sampled spectrum at every reconstructed spatial location with relatively simple detection hardware.

The approach is useful when detector arrays are expensive or inefficient, and when collecting light from many spatial locations improves sensitivity. Applications include material discrimination, food inspection, biological tissue imaging and measurements outside the visible spectrum. Spatial multiplexing couples many unknown pixels into each measurement; detector offset, finite collection efficiency and noise must therefore be handled consistently to recover the spatial and spectral information.

## Problem Description

This acquisition contains a complete real tomato-slice field with 64 by 64 spatial cells and 2048 measured spectral channels. Recover the signed, detector-equivalent intensity cube at its native resolution. Its values retain the illumination spectrum and instrument response: the result is not an absolutely calibrated transmission, reflectance or fluorophore concentration map.

Let $F_{p\ell}$ be the unknown intensity at flattened spatial pixel $p$ and spectral channel $\ell$, in detector counts per spatial pixel. There are $N=4096$ spatial pixels, indexed in row-major order: $p=64y+x$, where $x,y$ are column and row indices from 0 to 63. Let $W$ be the unnormalized 64 by 64 Walsh matrix, obtained by ordering the rows of the Sylvester Hadamard matrix by increasing number of sign changes along each row. The two-dimensional sensing matrix is $H=W\otimes W$, where $\otimes$ is the Kronecker product. Its entries are positive or negative one, and $H^\mathsf{T}H=NI$; $I$ is the spatial identity matrix.

For Walsh coefficient index $j$, the positive and negative physical patterns are $P^+_{jp}=(1+H_{jp})/2$ and $P^-_{jp}=(1-H_{jp})/2$. Measured pattern identifiers $2j$ and $2j+1$ denote those patterns, respectively. If $a$ indexes the order of acquired complementary pairs, the expectation model is


```math
\mathbb{E}[Y^+_{a\ell}]=\sum_p P^+_{j(a)p} F_{p\ell}+b_\ell,
\qquad
\mathbb{E}[Y^-_{a\ell}]=\sum_p P^-_{j(a)p} F_{p\ell}+b_\ell,
\qquad
\mathbb{E}[Y^+_{a\ell}-Y^-_{a\ell}]=\sum_p H_{j(a)p}F_{p\ell}.
```


Here $Y^+,Y^-$ are observed detector counts, $j(a)$ is specified by the acquired pattern identifiers, and $b_\ell$ is a detector offset assumed stable within a complementary pair. Differences contain acquisition noise and any within-pair drift. Independent electron gain and noise-variance calibration are unavailable; no absolute photon-noise likelihood is prescribed. All spatial coefficients are acquired, so the ideal linear operator is invertible. Practical difficulties include the acquisition permutation, two-dimensional Walsh convention, correct normalization and preservation of noise-sensitive spectral detail.

Save **`output/reconstruction.npz`**, relative to the task workspace, containing:

| Key | Shape | Dtype | Description |
| --- | --- | --- | --- |
| reconstruction | (1, 64, 64, 2048) | float64 | Real signed intensity, in detector counts per spatial pixel; axes batch, spatial row y, spatial column x, spectral channel. Spectral order matches wavelength_nm. Nominal spatial pixel width is 182.4 micrometres in both directions; pixel centres are (x+0.5, y+0.5) times this width, with x rightward and y downward from the upper-left field boundary. |

Preserve the measured physical amplitude convention, including small negative reconstructed intensities caused by signed measurement differences. Do not apply clipping, fitted rescaling, spectral normalization, absolute values or spatial alignment. Evaluation uses cosine NCC without mean subtraction, NRMSE equal to RMS error divided by the comparison cube's dynamic range, and relative L2 consistency with the measured complementary differences under the declared sensing operator. A zero-norm prediction receives NCC0 and fails. No crop or spectral selection is performed. These checks evaluate numerical reconstruction and measurement consistency, rather than independently known material truth.

## Data Description

The complete native observations and legitimate acquisition identifiers are in `data/raw_data.npz`:

| Key | Shape | Dtype | Description |
| --- | --- | --- | --- |
| measurements | (1, 8192, 2048) | float64 | Original detector readings in counts; axes batch, acquired binary pattern, measured spectral channel. Stored values are unchanged parsed source counts. |
| pattern_id | (1, 8192) | int64 | Dimensionless physical pattern identifiers in acquisition order. Adjacent rows contain positive even identifier 2j followed by negative identifier 2j+1; all identifiers from 0 to 8191 occur once. |
| wavelength_nm | (1, 2048) | float64 | Original increasing wavelength calibration in nm, paired with the last measurement axis. Use supplied values rather than a uniformly generated grid. |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
| --- | --- | --- | --- |
| Acquisition identifier | tomato_slice_2_zoomx1 | dimensionless | Complete author-provided real field, acquired 2021-12-08. |
| Spatial shape | 64 by 64 | pixels | Native reconstruction grid; no spatial resizing or crop. |
| Complementary pair count | 4096 | pairs | Every spatial Walsh coefficient measured once. |
| Spectral channel count | 2048 | channels | Complete native spectrometer axis. |
| Nominal object-plane pixel width | 182.4 | micrometres | Published nominal scale for the x1 configuration; not a newly measured calibration. |
| Integration time | 1.0 | ms | Actual stored spectrometer integration time per binary pattern. |
| DMD illumination time | 1356 | microseconds | Stored mirror illumination interval. |
| DMD dark phase | 44 | microseconds | Stored mirror switching interval. |
| Pattern period | 1400 | microseconds | Stored period per binary pattern. |
| Mean spectrometer time per pattern | 1.400009742966066 | ms | Source-recorded measured timing. |
| Total measured acquisition time | 11.4674765625 | s | Source-recorded complete spectrometer acquisition duration. |
| DMD display dimensions | 1024 by 768 | micromirrors | Hardware display size; the x1 pattern uses its largest square region. |
| Light source | white LED with diffuser | dimensionless | Illumination spectrum remains in reconstructed intensity. |
| Instrument configuration | v1.2, telecentric illumination, 50 mm collection lens, 20x objective | dimensionless | Source setup record; no new optical correction is required. |
| Detector saturation detected | false | dimensionless | Source acquisition flag. |
| Detector dark correction | disabled | dimensionless | Stable additive offsets cancel through complementary differencing. |
| Independent electron gain and noise variance | unavailable | counts/electron; counts squared | Do not infer a calibrated shot-noise uncertainty from the unknown image. |
| Spatial boundary convention | finite square orthogonal transform | dimensionless | No extrapolated or periodic physical specimen is assumed. |

The experimental data are redistributed under CC BY 4.0, with attribution in the task's NOTICE. They contain no independently measured true intensity cube.

## Method Hints

Complementary differencing removes a stable detector offset and produces signed Walsh coefficients. Undo the acquisition permutation before applying the separable two-dimensional inverse transform. Orthogonality makes the direct least-squares inverse appropriate for the fully measured case; its normalization is the total number of spatial pixels, not the one-dimensional side length.

For an optional retrospective sampling experiment, the minimum-norm reconstruction places missing Walsh coefficients at zero before applying the same inverse. This changes spatial information and noise behaviour without creating new measurements. Keep the measured spectral calibration and signed amplitudes throughout.

## References

Beneti Martins, G., Mahieu-Williame, L., Baudier, T. and Ducros, N. (2023). *OpenSpyrit: an ecosystem for open single-pixel hyperspectral imaging*. Optics Express 31(10), 15599–15614. [doi:10.1364/OE.483937](https://doi.org/10.1364/OE.483937). This task selects its explicit direct linear inverse, including its full native spectral dimension.

SPIHIM collection. Complete real acquisition `tomato_slice_2_zoomx1`, configuration v1.2. [Persistent acquisition record](https://pilot-warehouse.creatis.insa-lyon.fr/#collection/6140ba6929e3fc10d47dbe3e/folder/61b07ee23f7ce79f5e565c2c). CC BY 4.0.
