# UCLA BOLD Hemodynamic and Neural Input Deconvolution

> Recover regional hemodynamic response kernels and relative neural inputs from real resting-state fMRI BOLD time courses whose vascular blurring confounds temporal interpretation.
> Domain: medicine | Keywords: functional imaging, blind deconvolution, hemodynamics | Difficulty: Hard

## Background

Functional MRI detects blood-oxygenation-dependent changes in magnetic-resonance signal. Neural activity changes local blood flow and oxygenation through neurovascular coupling, so the recorded response is temporally blurred and delayed relative to the underlying activity. The vascular response varies across brain regions and subjects.

Resting-state measurements are useful for studying functional organization without requiring an explicit stimulus sequence. Estimating their hemodynamic response can help separate vascular timing differences from relative changes in latent neural input. This separation is intrinsically ambiguous: physiological fluctuations and motion can also contribute to BOLD, and fMRI sampling cannot resolve arbitrary fast firing dynamics.

## Problem Description

**Scientific contract status: not certified for benchmark release.** The blind HRF/input target depends on a finite event/lag-selection protocol whose independent physiological estimand is unresolved. The physical model, observations and output layout below remain documented; historical method-comparison criteria are not a certified public challenge.

The observations are the complete152retained time samples from one real author-distributed UCLA resting-state acquisition, aggregated using400fixed anatomical cortical parcels. Recover one hemodynamic response function(HRF)and one relative latent input per parcel. The input aggregation is an arithmetic average of independently defined atlas vertices, before any HRF inference; it does not depend on a reconstructed response or target.

Let y(r,t) denote the standardized and conventionally filtered BOLD signal in parcel r at scan t. Let x(r,t) be its model-conditioned relative input and h(r,j) its discrete HRF at scan-rate lag j. The finite-record observation model is

```math
y(r,t)=\sum_{j=0}^{12}h(r,j)x(r,(t-j)\bmod152)+\epsilon(r,t).
```

Epsilon denotes measurement and model discrepancy. The periodic convention describes the finite FFT inverse; it is not a claim that a living brain is periodic. A finer HRF representation has37samples at lags0,2/3,...,24seconds. Scan-rate kernels use downsampling by3with zero exterior values and the standard Kaiser-windowed polyphase antialiasing convention. There is no integration-step multiplier in the discrete sum.

Blind scaling and timing are not uniquely determined by BOLD alone. The reference convention uses unit-amplitude point events inferred from local BOLD peaks, a causal HRF basis with onset-delay search and an intercept, followed by regularized input recovery. Unit events define the HRF amplitude scale; the latent series has no firing-rate units and is not independent neuronal ground truth. The source-derived first-level event design uses zero initial prehistory, distinct from the finite periodic inverse above. All400regions are included without fit-quality or outcome-based selection.

Save `output/reconstruction.npz` relative to your task workspace, containing exactly these real float64 arrays:

| Output key | Shape | Units and convention |
|---|---|---|
| hrf | (1,400,37) | Standardized BOLD amplitude per unit discrete input; axes batch,parcel,HRF lag; lag grid0to24swith2/3sstep |
| neural_drive | (1,400,152) | Relative dimensionless discrete input; axes batch,parcel,retained scan; no positivity constraint or firing-rate conversion |

Parcel order is left cortical atlas labels1–200then right labels1–200. Scan0is the first retained sample, with relative time0; its original scanner-start offset is not supplied. Amplitude standardization subtracts each observed parcel mean and divides by sample standard deviation(ddof1). The actual selected source flow applies a reflected-record0.01–0.08Hzfilter to both HRF estimation and input recovery; it adds back the original mean. No per-output scale alignment, translation, amplitude fit or hidden mask is allowed.

Evaluation compares both complete arrays using cosine NCC and RMS error divided by each reference array’s dynamic range. It also checks worst-parcel dynamic-range NRMSEseparately for each quantity and the maximum over parcels of forward-model residual RMS divided by the RMS of filtered BOLD. Every criterion must pass. These are standard-estimator reproduction and observation-consistency criteria, not proof of uniquely recovered neuronal activity.

## Data Description

| Key | Shape | Dtype | Description |
|---|---|---|---|
| bold | (1,400,152) | float64 | Real author-preprocessed parcel-average BOLD intensity, arbitrary scanner units; batch,parcel,retained scan |
| vertex_counts | (1,400) | int64 | Number of independently defined atlas vertices in each arithmetic mean, dimensionless count; batch,parcel |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Subject/run | UCLA CNP sub-10171,task-rest | none | Author demo acquisition; no diagnosis is inferred |
| Scanner | Siemens TrioTim | none | Native author sidecar |
| Magnetic field | 3 | tesla | Field strength |
| Repetition time | 2 | s | Separation between retained scans |
| Echo time | 0.03 | s | BOLD echo timing |
| Native processed surface | fsaverage5,10242vertices per hemisphere | vertex count | Author fMRIprep-projected cortical data |
| Atlas | Schaefer2018,400parcels,7networks | none | Fixed independent anatomical aggregation |
| Retained observations | 152 | scan count | All scans actually present in the supplied author surface files |
| Nominal sidecar scan time | 312 | s | Acquisition metadata; not used to invent additional missing frames or a known removal history |
| Coordinate order | LH1–200thenRH1–200; temporal order unchanged | none | Each output row matches its input row |
| Boundaries | Reflected filtering; causal event design; periodic finite inverse | none | Explicit numerical boundary conventions, not physiological assumptions |
| Noise parameters | No calibrated per-parcel noise variance supplied | none | Correlated physiological/model effects remain possible |

## Method Hints

Consider a blind hemodynamic-deconvolution family that detects candidate BOLD events, fits a canonical response basis with timing/dispersion flexibility under autoregressive noise, and regularizes the latent input inverse. The Fourier-domain input update for a fixed kernel is

```math
\widehat X=\frac{\overline H\,Y}{|H|^2+\lambda},
```

where X,YandHare the discrete Fourier transforms of input,BOLDandzero-padded HRF, the overbar is complex conjugation and lambda is a positive ridge parameter. Preserve a stated amplitude convention and model the finite sampling/boundary conditions explicitly. A reconstructed latent input remains conditional on the event and HRF assumptions.


The actual reflected filter appends the exact reversal of its152 input values, giving304 values with both join endpoints duplicated: (y0,...,y151,y151,...,y0). On that304-point Fourier transform, bin k has nonnegative folded frequency min(k,304-k)/(304 times2s). Retain bins with frequency at least0.01Hz and strictly below0.08Hz, set the others to zero, take the real inverse transform and first152 values, and add back the mean of the input to this filter call. Before this bandpass, the standardized record undergoes the same extension/inverse/mean-return operation with all bins retained. The returned mean is a standardized-input mean, not the original scanner-intensity mean.

Define G(t;a,b)=t^(a-1)exp(-t/b)/(Gamma(a)b^a) for t>0 and zero otherwise, where a is shape, b is scale in seconds and Gamma is the gamma function. For sampled times t=0,2/3,...,24s, form b(d,o) by normalizing the sampled sum of G(t-o;6/d,d)-G(t-o;16,1)/6 to one. The response dispersion d is in seconds and onset o in seconds. Use the three columns b(1,0), b(1,0)-b(1,1), and [b(1,0)-b(1.01,0)]/0.01, orthogonalized in this order by Gram-Schmidt without unit-length normalization. The gamma arguments use the corresponding numerical second units.

For the already specified zero-padded scan-rate HRF spectrum H and filtered BOLD spectrum Y, the actual input inverse is

```math
\widehat X(\omega)=
\frac{\overline{H(\omega)}Y(\omega)}
{|H(\omega)|^2+\lambda},\qquad
\lambda=0.1\,\frac{1}{152}\sum_{k=0}^{151}|H(\omega_k)|^2.
```

Here omega labels the152 discrete Fourier bins. Take the real inverse transform without an additional scale factor. This lambda is a fixed relative spectral ridge, not an independently measured neuronal noise variance. The37-to13 sample conversion is a61-tap, unit-DC low-pass filter with cutoff1/3 of the microtime Nyquist frequency, Kaiser shape5, constant-zero exterior values and zero-phase alignment to microtime indices0,3,...,36. More explicitly, for integer l with absolute value at most30, let

```math
K_l=c\,\operatorname{sinc}(l/3)
\frac{I_0\!\left(5\sqrt{1-(l/30)^2}\right)}{I_0(5)},
\qquad \sum_{l=-30}^{30}K_l=1,\qquad
h^{\mathrm{scan}}_j=\sum_{m=0}^{36}K_{3j-m}h_m,\quad j=0,\ldots,12.
```

Here sinc(v)=sin(pi v)/(pi v), with sinc(0)=1, I0 is the modified Bessel function of order zero, c enforces the stated unit sum, and K is zero outside that integer support. The finer-grid kernel is h, and its scan-rate version is h-scan. All finite event/lag/AR/basis conventions above are part of the conditional standard-estimator definition rather than physiological ground truth.

## References

1. Wu G.-R. et al.,“rsHRF:A toolbox for resting-state HRF estimation and deconvolution,”NeuroImage244,118591(2021),[doi:10.1016/j.neuroimage.2021.118591](https://doi.org/10.1016/j.neuroimage.2021.118591).
2. Wu G.-R. et al.,“A blind deconvolution approach to recover effective connectivity brain networks from resting state fMRI data,”Medical Image Analysis17,365–374(2013),[doi:10.1016/j.media.2013.01.003](https://doi.org/10.1016/j.media.2013.01.003).
3. Poldrack R.A. et al.,“A phenome-wide examination of neural and cognitive function,”Scientific Data3,160110(2016),[doi:10.1038/sdata.2016.110](https://doi.org/10.1038/sdata.2016.110).
4. Schaefer A. et al.,“Local-Global Parcellation of the Human Cerebral Cortex from Intrinsic Functional Connectivity MRI,”Cerebral Cortex28,3095–3114(2018),[doi:10.1093/cercor/bhx179](https://doi.org/10.1093/cercor/bhx179).
