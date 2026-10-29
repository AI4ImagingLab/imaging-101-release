# Tidal Current Ellipses

> Infer two-component tidal harmonic currents and constituent ellipses from an irregular, gappy coastal Doppler-current record containing substantial non-tidal variability.
> Domain: earth science | Keywords: ocean current inference, harmonic inversion, vector reconstruction | Difficulty: Medium

## Background

Lunar and solar gravitational forcing drives periodic ocean motion that coastal geometry transforms into locally varying horizontal currents. The tip of each harmonic velocity vector can trace a narrow reversing line or a rotating ellipse. These constituent responses describe transport direction, rotation and flood/ebb variability.

Current measurements support navigation, coastal engineering and ocean-transport studies. Storm-driven flow, river discharge and local variability coexist with the tidal response, while telemetry gaps and closely spaced tidal frequencies complicate separation. A fitted harmonic response describes a stationary tidal component and does not explain every measured fluctuation.

## Problem Description

Recover the constant mean current and eight vector harmonics from the complete available first-quarter2025 observations at NOAA stationcb1401, Newport News Shipbuilding. In a fixed east,north coordinate system, model

```math
\mathbf v(t)=\mathbf m+\sum_{k=1}^{8}\left[\mathbf c_k\cos(2\pi f_k\tau)+\mathbf s_k\sin(2\pi f_k\tau)\right]+\boldsymbol\epsilon(t),
\qquad \tau=(t-t_r)/3600.
```

Here v is measured horizontal current incm/s, m is its constant two-component fitted mean, ck andsk are cosine and sine vector coefficients incm/s, fk is the supplied astronomical frequency incycles/hour, t andtr areseconds, τ is elapsedhours and ε contains measurement error and non-tidal dynamics. Reference time tr is the midpoint of the first and last supplied observation. The selected model has no linear trend, nodal/satellite modulation or conversion to Greenwich astronomical phase. It estimates stationary effective coefficients over the bound90-day interval; it is not a multi-year astronomical tide prediction model.

Measured current direction d is in degrees clockwise from true north, **toward** flow. East velocity is speed*sin(d), north velocity is speed*cos(d). Do not use the meteorological direction-from convention. Samples are irregular and gaps remain absent; do not fill them with zero.

Save `output/reconstruction.npz` relative to the task workspace, with real float32 orfloat64 arrays:

| Output key | Batch-first shape | Meaning and units |
|---|---|---|
| cosine | (1,2,8) | ck incm/s; component order east,north and constituent order below |
| sine | (1,2,8) | sk incm/s with the positive-sine sign in the stated equation |
| mean_current | (1,2) | Constant east,north mean incm/s |
| reconstruction | (1,2,16822) | Mean plus all eight harmonics at every suppliedtime,cm/s; same time/component order |

All values retain absolute gain,physical units and coordinate signs. Every component of every array is evaluated. Cosine,sine and reconstructed east/north series are independently compared by uncentered cosine similarity and RMS error divided by comparison-array dynamic range; mean current is assessed by maximum absolute error incm/s. There is no rescaling, temporal shift or rotation alignment.

The full Cartesian coefficients uniquely specify each constituent velocity ellipse and avoid angle ambiguities for weak or circular components. For a constituent define complex C=cEast+i*cNorth,S=sEast+i*sNorth, rotary coefficients p=(C−iS)/2 andq=(C+iS)/2. Its semi-major axis is|p|+|q| and signed semi-minor axis|p|−|q|, positive for counterclockwise rotation. Display inclination θ=[(argp+argq)/2]mod180degrees and raw phase g=(θ−argp)mod360degrees use a fixed gauge. Their separate angles become nonidentifiable for circular/zero-strength limits; the required Cartesian output remains well defined.

## Data Description

The data retain all16822 finite polar-current pairs actually returned by the three NOAA monthly observation queries. No grid resampling, gap imputation, response fitting or value clipping was applied. The API records contain timestamp,speed,direction andbin; no per-record QCflag is supplied, so these observations are not represented as individually verified measurements.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| time_seconds | (1,16822) | float64 | Actual UTC acquisition times,seconds since2025-01-01T00:00:00UTC |
| speed | (1,16822) | float64 | Measured horizontal current speed incm/s |
| direction_deg | (1,16822) | float64 | Measured current direction,degree clockwise from true north,towardflow |
| frequency_cph | (1,8) | float64 | Known astronomical constituent frequencies incycles/hour,not estimated from unknown current coefficients |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Station | cb1401, Newport News Shipbuilding | — | NOAA CO-OPS, Chesapeake Bay South PORTS |
| Latitude / longitude | 36.9838 / −76.4436 | degree | North-positive latitude,east-positive longitude |
| Instrument | Sontek ADP/Nortek AWAC | — | Deployed side-looking Doppler current profiler |
| Deployment start | 2021-02-24T10:00:00 | station metadata time | Deployment covering this record |
| Sensor depth | 6.7 | m | Instrument depth,not a vertical depth assigned to eachbin |
| Sampling bin | 30 | — | Predefined operational realtimebin |
| Bin30 horizontal range | 121.01 | m | Side-looking sample distance from instrument,not water depth |
| Nominal interval | 360 | s | Actual timestamps vary; retain them |
| First / last observation | 2025-01-01T00:05 /2025-03-31T23:58 | UTC | Full available quarter record |
| Largest data gap | 182100 | s | Missing telemetry is left absent |
| Constituent order | M2,S2,N2,K1,O1,M4,MS4,MN4 | — | Fixed order for frequency/coefficient axes |
| Phase reference | Midpoint of first andlasttime | UTC | Raw-phase gauge,sameforbothcomponents |
| Boundary assumptions | Stationary local harmonics andmean | — | No trend,nodal modulation or imposed spatialboundary |
| Noise | No supplied variance model | — | Measured noise plus unresolved environmental flow |

Source: NOAA CO-OPS observed currents, January–March2025; public-domain NOAA data. Credit NOAA CO-OPS. The supplied frequencies are independently known astronomical calibration evaluated at the record midpoint; they are not fitted from these currents.

## Method Hints

Represent the east/north signal as a complex current and fit positive- and negative-frequency exponentials jointly with a mean. Ordinary least squares handles the actual irregular timestamps directly. Converting rotary coefficients back to Cartesian sine/cosine coefficients preserves both flow components; constituent ellipses then follow algebraically. Inspect harmonic identifiability, residual structure and sensitivity to record length rather than assuming all flow is tidal.

## References

1. Codiga,D.L.(2011). Unified Tidal Analysis and Prediction Using the UTide Matlab Functions. **GSO Technical Report2011-01**, University of Rhode Island; a technicalreport,not a journalarticle.
2. NOAA CO-OPS Data API documentation. https://api.tidesandcurrents.noaa.gov/api/prod/
3. NOAA station metadata. https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations/cb1401.json
