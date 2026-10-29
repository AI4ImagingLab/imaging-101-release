# California Seismic Source Tensor

> Reconstruct the complete earthquake moment tensor and source depth from regional three-component displacement waveforms despite imperfect Earth models and overlapping source radiation patterns.
> Domain: earth science | Keywords: seismic source inversion, elastic wave imaging, waveform tomography | Difficulty: Medium

## Background

An earthquake launches elastic waves whose directional amplitudes encode the forces released at its source. At regional wavelengths, a compact rupture can be represented by a symmetric moment tensor: the diagonal components describe directed expansion or contraction, while the off-diagonal components describe shear couples.

Recovering this tensor helps distinguish faulting styles, quantify earthquake strength, and understand volcanic or induced seismicity. The signal recorded at a seismometer also depends on wave propagation through the crust. Multiple stations and all three displacement components help separate source radiation from these propagation effects, although errors in Earth structure can still bias the answer.

## Problem Description

Recover a symmetric source tensor $M$ in north-east-down coordinates and a depth $h$ from a finite candidate set. For station (s), component (c), and time sample (t), the observation satisfies

```math
d_{sct}=\sum_{j=1}^{6}G_{sctj}(h)\,a_j+\epsilon_{sct},\qquad
a=\frac{(M_{NN},M_{NE},M_{ND},M_{EE},M_{ED},M_{DD})}{10^{13}\,\mathrm{Nm}}.
```

Here $d$ is calibrated displacement in cm, $G$ is the calibrated elastic response per $10^{20}$ dyne cm, $a$ is dimensionless, and $\epsilon$ includes observational noise and propagation-model discrepancy. The acquisition fixes the horizontal source position. The response is supplied as ten fundamental-fault Green traces, rather than a preassembled tensor operator.

The following expressions specify the complete acquisition model. Write $\alpha$ for station azimuth clockwise from north; $N,E,D$ denote the tensor axes. Define

```math
A=\frac{1}{2}(a_1-a_4)\cos(2\alpha)+a_2\sin(2\alpha),\quad
B=a_3\cos\alpha+a_5\sin\alpha,\quad
C=-0.166667(a_1+a_4)+0.33333a_6,\quad
I=0.33333(a_1+a_4+a_6).
```

For vertical or radial component $q$, use $u_q=AqSS+BqDS+CqDD+IqEX$. For transverse displacement, use

```math
u_T=[\frac{1}{2}(a_1-a_4)\sin(2\alpha)-a_2\cos(2\alpha)]TSS
    +[a_3\sin\alpha-a_5\cos\alpha]TDS.
```

The symbols $qSS,qDS,qDD,qEX,TSS,TDS$ are the provided fundamental Green traces at the selected depth. Retain the decimal constants shown above: they fix the supplied calibration convention. Vertical points up, radial points away from the event, and transverse follows the supplied right-handed great-circle rotation. Radiation patterns and source depth trade off with one another, and the measured waveforms contain contributions not captured by the fixed Earth response.

Save `output/reconstruction.npz` containing `reconstruction`, a real symmetric float64 array of shape `(1,3,3)` in Nm, with rows and columns ordered north, east, down, and `source_depth`, a real float64 array of shape `(1,1)` in km containing one candidate depth. Keep the physical amplitude and signed tensor components. No scale, sign, rotation, or translation alignment is applied. Evaluation requires relative Frobenius tensor agreement, correct depth, and distance-weighted waveform variance reduction. The latter is $100[1-\sum w(d-u)^2/\sum wd^2]$, with $w_s=r_s/\min r_s$, where $r_s$ is station distance. Every required quantity is scored.

## Data Description

| Key | Shape | Dtype | Description |
|---|---|---|---|
| waveforms | (1,8,3,231) | float32 | Measured instrument-corrected and bandpass-filtered Z,R,T displacement, cm; complete supplied time records. |
| fundamental_greens | (1,3,8,10,150) | float32 | Independently computed propagation calibration, cm per10^20 dyne cm; depth, station, fundamental fault, time axes. |
| station_distance | (1,8) | float64 | Horizontal source-to-station distances, km. |
| station_azimuth | (1,8) | float64 | Source-to-station azimuth clockwise from north, degrees. |
| station_coordinates | (1,8,2) | float64 | Station longitude and latitude, degrees east and north. |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Event origin | 2019-07-16T20:11:01.470000Z | UTC | Fixed origin used for waveform timing. |
| Source longitude; latitude | -121.7568;37.8187 | degree east;north | Fixed horizontal source position. |
| Candidate depths, in array order | 10,12,20 | km | Propagation calibration hypotheses. |
| Sampling interval | 1 | s | Common waveform and Green sampling. |
| Record interval | -30 to200 | s relative to origin | Complete measured records have231samples. |
| Window start indices, station order | 32,32,31,31,31,31,31,32 | sample | Supplied per-station time registration. |
| Window length | 150 | sample | Samples used by each Green response and inverse; no automatic recorrelation. |
| Bandpass | 0.02 to0.05 | Hz | Common3-corner zero-phase Butterworth filter in calibrated data. |
| Waveform taper | 0.05 | fraction | Supplied trace edge taper. |
| Stations, in array order | BK.QRDG.00,BK.RUSS.00,BK.CVS.00,BK.OAKV.00,BK.FARB.00,BK.SAO.00,BK.CMB.00,BK.MNRC.00 | identifier | All eight stations in this acquisition. |
| Fundamental-fault order | ZSS,ZDS,ZDD,ZEX,RSS,RDS,RDD,REX,TSS,TDS | identifier | Strike-slip,dip-slip,dip-dip,explosion response components. |
| Green model | gil7, layered isotropic elastic Earth | identifier | Source-independent supplied crustal propagation model; free surface and outward propagation are included in calibration. |
| Moment scale | 10^20 dyne cm=10^13 Nm | moment | Coefficients multiplying the supplied Green traces. |
| Noise model | Unspecified additive noise and modelling error | cm | No calibrated statistical noise variance is supplied. |

Station-dependent windows start at the listed indices of the measured records, while each Green trace starts at its first sample. All input sample values retain the acquisition float32 precision; use float64 arithmetic for the inverse. The calibrated Green responses are available independently of the unknown source tensor.

## Method Hints

The canonical estimate minimizes sum over every station, component and registered sample of w_s(d_sct-(G(h)a)_sct)², using w_s=r_s/min(r) as the objective weight itself, not its square. Use all six symmetric-tensor coefficients, without a trace constraint or other penalty. Select the candidate depth with the smallest weighted residual energy; exact ties choose the first depth in the listed order. The response and observation windows above fully specify this finite linear inverse.

Time-domain moment-tensor inversion combines elementary radiation patterns into a small linear least-squares problem at each trial depth. Distance weighting can balance unequal station amplitudes. Compare the full symmetric tensor with a deviatoric constraint to diagnose how much apparent volume change the fixed propagation model permits. A tensor fit is a source characterization, with uncertainty arising from limited station coverage and Earth-model mismatch.

## References

Minson,S.E.and Dreger,D.S.(2008). Stable inversions for complete moment tensors. Geophysical Journal International174,585–592. https://doi.org/10.1111/j.1365-246X.2008.03797.x

Chiang,A.(2020). MTtime: Time Domain Moment Tensor Inversion in Python. LLNL-CODE-814839. 

Berkeley Digital Seismic Network, Northern California Earthquake Data Center. https://ncedc.org/
