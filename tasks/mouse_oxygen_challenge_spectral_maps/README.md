# Mouse Oxygen-Challenge Spectral Maps

> Recover spatial haemoglobin spectral coefficients and apparent blood-oxygenation dynamics from a real multiwavelength optoacoustic image sequence despite spectral overlap and unknown tissue fluence.
> Domain: biology | Keywords: molecular imaging, spectral unmixing, oxygenation imaging | Difficulty: Medium

## Background

Oxy- and deoxyhaemoglobin absorb near-infrared light differently. Short optical pulses generate acoustic waves through thermoelastic expansion where light is absorbed. Reconstructing those waves into wavelength-dependent images makes blood contrast accessible beneath tissue surfaces, with applications in vascular physiology, tumour perfusion and hypoxia research.

Changing an animal's breathing gas can alter blood oxygenation over time. A spatial map can reveal differences among tissue regions that would be lost in one whole-subject average. Such measurements provide conditional imaging biomarkers rather than direct counts of oxygen molecules.

Light transport complicates this interpretation. Optical fluence varies with depth and wavelength, while acoustic reconstruction contributes its own noise and response. Spectral decomposition can therefore produce an apparent oxygenation estimate even when its mathematical fit is reproducible. Absolute haemoglobin concentration is not identifiable without additional fluence and instrument calibration.

## Problem Description

Recover both haemoglobin coefficient fields and apparent saturation through all76 measured wavelength cycles of a real mouse oxygen-challenge acquisition. Each cycle contains ten wavelength images. The images are supplied after the author's reference acoustic reconstruction and standard spatial averaging/resampling; acoustic initial pressure is not the unknown in this task. Three author-drawn anatomical regions define fixed evaluation support, without fit-based pixel rejection.

At pixel r, cycle t and wavelength lambda, the conditional spectral model is

```math
I(t,\mathbf r,\lambda)\simeq\sum_{c=1}^{2}a_c(t,\mathbf r)\,\epsilon_c(\lambda),
\qquad s(t,\mathbf r)=\frac{a_2(t,\mathbf r)}{a_1(t,\mathbf r)+a_2(t,\mathbf r)}.
```

I is the supplied arbitrary-unit optoacoustic image. The calibrated spectral rows epsilon1 and epsilon2 are decadic molar extinction coefficients of deoxyhaemoglobin(Hb) and oxyhaemoglobin(HbO2), in inverse centimetres per molar. The unknown a1 and a2 are effective spectral coefficients, in image units divided by those extinction units. They absorb unknown common fluence, acoustic gain and the decadic-to-natural absorption factor; they are not absolute molar concentrations. The derived s is dimensionless apparent oxygen saturation under the wavelength-independent common-fluence approximation. Wavelength-dependent fluence, other absorbers, motion within a cycle and image noise violate this model.

Save `output/reconstruction.npz` relative to the task workspace with exactly these arrays:

| Output key | Shape | Dtype | Units and convention |
|---|---|---|---|
| reconstruction | (1,76,2,1,111,111) | float32 | Effective spectral coefficients in image units/(inverse centimetres per molar); axes batch,cycle,chromophore,z,y,x; chromophore order Hb,HbO2 |
| saturation | (1,76,1,111,111) | float32 | Dimensionless apparent oxygen saturation; axes batch,cycle,z,y,x; fraction, not percent |

The singleton z axis is the acquired image plane. Both outputs must be real, finite and exactly zero where roi_labels is zero, with no spatial registration or fitted scaling. The conditional inverse is unconstrained: do not silently clip negative coefficients or fractions outside [0,1]. A zero total coefficient makes the ratio undefined; the supplied fixed support has no such case for the selected estimate. No support location is removed based on its inferred oxygenation.

Evaluation compares Hb, HbO2 and saturation separately over every supported pixel and all76 cycles, using cosine NCC and RMS error divided by each reference field's dynamic range. All three fields must pass. A separate consistency metric is the L2 norm of the difference between supplied images and the two-spectrum prediction, divided by the L2 norm of supplied images, using all ten wavelengths, all cycles and the fixed support. No phase, offset, magnitude, temporal or spatial alignment is applied. Reference agreement does not establish true physiological oxygenation.

## Data Description

| Key | Shape | Dtype | Description |
|---|---|---|---|
| images | (1,76,10,1,111,111) | float32 | Real author-standard spatially averaged optoacoustic images, arbitrary image units; axes batch,cycle,wavelength,z,y,x |
| spectra | (1,2,10) | float64 | Independent tabulated Hb,HbO2 extinction spectra sampled at the measured wavelengths, in inverse centimetres per molar; axes batch,chromophore,wavelength |
| wavelength | (1,10) | float64 | Optical wavelengths in nm |
| acquisition_time | (1,76,10) | float64 | Per-wavelength acquisition timestamps relative to the first acquisition, in s |
| roi_labels | (1,1,111,111) | uint8 | Author-drawn anatomical regions, dimensionless:0exterior,1reference region,2left tumour,3right tumour; axes batch,z,y,x |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Acquisition | 201001_05_GC_SS,2020-10-01 | none | Real mouse oxygen-enhanced acquisition |
| System | MSOT inVision256,cylindrical array | none | Author preclinical optoacoustic system |
| Optical wavelengths | 700,730,750,760,770,800,820,840,850,880 | nm | Original wavelength order, all retained |
| Wavelength cycles | 76 | cycles | Complete temporal sequence; no cycle averaging or selection |
| Nominal image field of view | 0.024975 by0.024975 | m | Author image x/y field; no additional crop |
| Native reconstructed lattice | 333by333 | pixels | Supplied author acoustic reconstruction before standard spatial rescaling |
| Bound lattice | 111by111 | pixels | Author's factor3 spatial rescaling of the complete image field |
| Spatial rescaling | uniform3by3convolution,weight1/9,zero extension,followed by indices0,3,...330 | native pixel indices | Original rescaling operator; boundary apertures include zero extension |
| Bound image coordinates | linspace(-0.0124875,+0.0124875,111) in both x and y | m | Author retained-FOV coordinate convention after rescaling; row and column indices increase along y and x |
| Coordinate approximation | reduced grid retains nominal full FOV | none | Source metadata convention; strided samples need not coincide exactly with these relabeled endpoints |
| Native motor position | 59.65 | mm | Author acquisition plane position; not an additional image axis |
| Acoustic reconstruction | author Reference Backprojection,5kHz–7MHz bandpass,IRF correction,temporal interpolation3,detector interpolation2,Hilbert transform,no envelope detection | Hz,dimensionless | Processing history supplied with wavelength images; this task starts from those images |
| Sound speed convention | 1469 | m/s | Author upstream acoustic reconstruction parameter |
| Image normalization | as supplied by author reconstruction | arbitrary units | No additional per-wavelength pulse-energy,water-loss or tissue-fluence correction is applied by this inverse |
| Spectral calibration | independent tabulated haemoglobin spectra | inverse centimetres per molar | Physical basis is supplied directly; no target-derived pure spectrum |
| Timestamp convention | source100ns clock converted to seconds then referenced to first acquisition | s | Sequential wavelengths within each cycle have distinct timestamps |
| Noise | measured experimental noise; variance unavailable | arbitrary units | No known shot-noise or reconstruction-noise variance is assumed |
| Support | fixed union of three author-drawn regions | dimensionless | Geometry selected from images, without coefficient/saturation-dependent quality masking |

Float32 storage of the averaged images was validated against the original higher-precision averaging; the bound inverse uses these exact stored observations. The nominal coordinate convention and finite averaging boundaries follow the author implementation. No absolute fluence, true haemoglobin concentration or true oxygen saturation is supplied.

## Method Hints

Use linear spectral unmixing with the two supplied spectral rows. The conditional target minimizes the unweighted squared residual over all ten wavelengths independently at each pixel and cycle; evaluate the Moore–Penrose pseudoinverse without positivity projection or wavelength renormalization. The spectral matrix has shape chromophore-by-wavelength. Preserve the selected float32 coefficient output before forming the float32 HbO2/(Hb+HbO2)ratio, with no clipping. A zero denominator is undefined rather than silently replaced by a physiological value.

The shared-fluence approximation lets a common intensity scale cancel in the ratio, while wavelength-dependent spectral colouring remains a source of bias. Wavelength-subset and spectral-colouring comparisons can expose that conditionality, and spatial averaging trades resolution for statistical stability.

## References

1. Else,T.R., Gröhl,J., Hacker,L., and Bohndiek,S.E. **PATATO: a Python photoacoustic tomography analysis toolkit.** Journal of Open Source Software9(93),5686(2024). [Paper](https://doi.org/10.21105/joss.05686).
2. Else,T. and Bohndiek,S. **Dataset for: PATATO: A Python PhotoAcoustic Analysis Toolkit.** [Author data](https://doi.org/10.17863/CAM.93181), Creative Commons Attribution4.0.
