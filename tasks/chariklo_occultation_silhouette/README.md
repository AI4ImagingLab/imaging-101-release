# Chariklo Occultation Silhouette

> Recover the projected elliptical silhouette of a distant minor planet from five timed stellar-occultation chords and a station that observed no disappearance.
> Domain: astronomy | Keywords: occultation imaging, geometric inversion | Difficulty: Medium

## Background

A distant star acts as a compact light source behind an intervening body. As the body's shadow sweeps over Earth, different observing stations record when the star disappears and reappears. Each positive station samples a chord through the projected outline. A station where the star remains visible constrains where the outline cannot extend.

Occultations measure angular sizes and shapes that conventional telescopes may not resolve. Their interpretation depends on timing precision, the distribution of observing sites and the accuracy of the known star and shadow trajectory. Sparse chords constrain a geometric model of the projected limb, rather than an arbitrary surface image or a complete three-dimensional body.

## Problem Description

Recover Chariklo's two-dimensional projected ellipse during the real 2017-06-22 event in Namibia. Five positive instrument records from four sites provide ten contacts; Hakos provides a negative record. The public limb points are known sky-plane positions of each observing ray at the measured contact time, derived from independent stellar/ephemeris calibration. Their coordinates are indirect timing constraints, not independent shape measurements.

Let f,g be coordinates in the sky plane, f0,g0 the unknown center in km, a the semi-major axis in km, o the dimensionless oblateness, b=a(1-o), and phi the pole position angle in degrees modulo 180. The pole is the minor-axis direction, measured clockwise from positive north. The polar limb around the center is

```math
r(\theta)=\frac{ab}{\sqrt{a^2\sin^2(\theta+\phi\pi/180)+b^2\cos^2(\theta+\phi\pi/180)}},
\quad (f,g)=(f_0,g_0)+r(\theta)(\cos\theta,\sin\theta).
```

Here theta is the counterclockwise polar angle from positive f. A positive contact must lie on this boundary within its timing uncertainty, while every sampled ray in the negative record must remain outside. The supplied local ray velocity converts timing uncertainty into distance uncertainty. Limited chord orientations, shared-station observations and uncertainty in the timings make the five geometry parameters correlated and nonunique.

Save `output/reconstruction.npz` with both required finite real float64 arrays:

| Key | Shape | Dtype | Description |
|---|---|---|---|
| parameters | (1, 5) | float64 | Ordered f0[km],g0[km],a[km],o[dimensionless],phi[degree modulo180] |
| reconstruction | (1, 360, 2) | float64 | Sky-plane boundary points[km], axes batch,angle,(f,g), at exactly contour_angles |

Require a>0 and 0<=o<1. The 360 output samples describe the five-parameter ellipse; they do not represent 720 independent unknowns. Evaluation checks every geometry parameter, wrapping the position-angle difference to [-90,90)degrees, and checks the complete boundary with uncentered cosine similarity and dynamic-range-normalized RMS error. It also independently verifies parameter/boundary consistency, all positive ray/ellipse intersection times normalized by their timing uncertainties, zero missed positive trajectories, and zero penetration into the negative path. No fitted translation, scale, angle alignment, or shape matching is applied.

## Data Description

All keys in `data/raw_data.npz` are batch first.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| limb_points | (1, 10, 2) | float64 | Observing-ray coordinates at measured contacts, km; axes contact,(f,g) |
| velocity | (1, 10, 2) | float64 | Known local ray velocity at each contact, km/s; axes contact,(f,g) |
| timing_sigma | (1, 10) | float64 | Author-example contact uncertainty weights, s |
| times_jd | (1, 10) | float64 | Contact times, Julian day UTC |
| negative_path | (1, 2394, 2) | float64 | Full Hakos non-occultation trajectory sampled every0.5s, km; axes sample,(f,g) |
| contour_angles | (1, 360) | float64 | Fixed output polar angles 0inclusive to2pi exclusive, rad |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Event | 2017-06-22 | UTC date | Real multi-station stellar occultation |
| Contacts | Outeniqua I/E,Onduruquea I/E,Tivoli I/E,Windhoek C14 I/E,Windhoek D16 I/E | ordered records | Immersion then emersion for every instrument |
| Sky axes | f positive east, g positive north | km | Plane perpendicular to star direction; origin is supplied ephemeris/star reference, not the unknown body center |
| Known star direction | 18h55m15.65210s,-31deg31arcmin21.6676arcsec at J2016.0 | ICRS | Author-example Gaia astrometry |
| Star motion | pmRA3.556,pmDec-2.050; parallax0.2121; radial velocity-40.49 | mas/year; mas; km/s | Known calibration used to compute public ray geometry |
| Outeniqua | longitude16:49:17.710E,latitude21:17:58.170S,height1416 | degree notation; m | Positive station |
| Onduruquea | 15:59:33.750E,21:36:26.040S,1220 | degree notation; m | Positive station |
| Tivoli | 18:01:01.240E,23:27:40.190S,1344 | degree notation; m | Positive station |
| Windhoek | 17:06:31.900E,22:41:55.160S,1902 | degree notation; m | Two separate positive instruments at one location |
| Hakos | 16:21:41.320E,23:14:11.040S,1843 | degree notation; m | Negative station |
| Hakos sampled window | 21:10:19.960588 to21:30:16.954860 on event date | UTC | Author cadence file's actual first/last sample after loading; sampled every0.5s, excluding final endpoint |
| Contact uncertainty | supplied timing_sigma; intentionally increased in author teaching example | s | These are author-example fitting weights, not a claim of original published measurement precision |
| Radial uncertainty | norm(velocity)*timing_sigma | km | Source contact-weight convention |
| Timing forward check | local straight ray at each supplied contact coordinate and velocity | km,s | Solve its intersections with the ellipse; nearest time shift compared with timing_sigma |
| Geometry calibration | bundled Chariklo ephemeris and de438_small; Chariklo kernel ID20010199; fixed geocenter; bundled Earth orientation | - | Public ray arrays already include this calibration; no ephemeris package is required |

There are exactly five free geometric parameters. Stellar astrometry, station positions and trajectory calibration are fixed inputs. The author's Hakos example flux column is identically one; it supplies example cadence and a declared negative status, rather than raw measured photometry. No independent body-surface truth is supplied. Elliptical shape, the continuous negative-window assumption, inflated example weights and linearized contact trajectories limit physical interpretation.

## Method Hints

A Monte Carlo chi-square search can explore correlated ellipse geometries using the radial distance mismatch at each contact. Keep the exact pole-angle convention and convert timing errors with local ray speed. Reject geometries that cross a negative station's trajectory. Confidence regions in the accepted parameter ensemble reveal degeneracy among size, oblateness, orientation and center.

## References

- Gomes-Junior et al. (2022), *SORA: Stellar Occultation Reduction and Analysis*, MNRAS511,1167-1181. [Author manuscript](https://arxiv.org/abs/2201.01799).
- [Author SORA occultation documentation](https://sora.readthedocs.io/en/latest/).
