# Photoelastic Granular Stress Mapping

> Infer four contact forces, their directions and an integrated principal-stress-difference map from a measured polarized-light image of a loaded transparent disk.
> Domain: mechanics | Keywords: photoelastic imaging, inverse elasticity, granular mechanics | Difficulty: Hard

## Background

Transparent stress-sensitive materials become optically birefringent when loaded. Light components along the principal stress directions accumulate different phases, creating characteristic intensity patterns between polarizers. These patterns encode the difference between principal stresses rather than the hydrostatic pressure.

Photoelastic disks provide a way to inspect how contact loads are transmitted through granular materials. Quantifying their patterns helps study force chains, friction, packing stability and mechanical failure. The periodic optical response, finite contact size, uneven illumination and uncertain material calibration can produce ambiguous inverse solutions.

## Problem Description

This task binds a full-disk region from a real RGB image of a photoelastic granular packing. Use a two-dimensional isotropic elastic disk with four specified point contacts. Each unknown force has nonnegative magnitude F_j and angular departure alpha_j from its radial contact direction. Static equilibrium requires the sum of both Cartesian force components and the sum of contact torques to vanish; no body force is assumed. The contact position is beta_j, measured from increasing row toward increasing column. The effective thickness-integrated stress tensor is the sum of the analytic disk contact responses.

For a contact, construct its chord direction d through the disk; for a pixel-to-contact vector v define r=norm(v) and q=(v dot d)/r. In the local chord coordinates the radial integrated stress is S=-2F_j*q/(pi*r). Rotate this tensor into the declared image axes and sum over contacts. If T is this total integrated tensor, the dark-field stress-optic law is

```math
\Delta T=\sqrt{(T_{xx}-T_{yy})^2+4T_{xy}^2},\qquad
I=\sin^2\!\left(\pi\Delta T/f_\sigma\right)+n.
```

Here T and Delta T have units N/m, representing stress integrated through thickness; f_sigma is the effective stress-optic coefficient in N/m, I is normalized optical intensity and n includes noise and optical/model mismatch. Disk geometry and the complete angular convention are specified below. This is a point-contact model of the recorded disk, not an independently force-calibrated experiment. The acquisition's nominal optical coefficient is supplied as an assumption; its uncertainty limits absolute inferred forces.

Save `output/reconstruction.npz` with all three real finite float64 arrays:

| Output key | Batch-first shape | Units and convention |
| --- | --- | --- |
| reconstruction | (1, 181, 181) | N/m; nonnegative integrated principal-stress difference, axes batch,row,column, exactly zero outside the disk |
| forces | (1, 4) | N; magnitudes in the supplied contact order |
| contact_directions | (1, 4) | rad; signed alpha departures, each in[-pi/2,pi/2] |

Pixel row/column coordinates are integer sample indices. The disk centre is(row93.26971254206686,column90.04002272424479), radius72.92296540262036pixels. Increasing row is downward and increasing column rightward. Contact beta=0 lies below the centre; beta=pi/2 lies to its right. In the source force convention, the Cartesian components are `(F_row,F_column)=(F*cos(beta-alpha), -F*sin(beta-alpha))`; these are signed load components, not image-coordinate displacement vectors. Evaluate the model using the effective sampling stated below. No hidden thickness value is needed or inferred.

Every output quantity is evaluated using cosine NCC and RMSE divided by its reference dynamic range; all required metrics must pass. The net-force magnitude divided by total contact-force magnitude and absolute torque divided by disk radius times total force must each be at most1e-6. No phase, sign, translation, scale fitting or cropping is applied. Reference-method agreement does not establish actual force or stress accuracy, and fitted parameters must not be treated as ground truth.

## Data Description

| Key | Shape | Dtype | Description |
| --- | --- | --- | --- |
| rgb | (1, 181, 181, 3) | uint8 | Measured RGB detector values, count on0–255 scale; axes batch,row,column,red-green-blue |
| contact_angles | (1, 4) | float64 | Fixed contact position beta, radians, order bottom,right,top,left |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
| --- | --- | --- | --- |
| Nominal spatial calibration | 0.007/160 | m/pixel | Author-supplied nominal scale, approximately4.375e-5m/pixel |
| Effective model sampling | 160/0.007 | pixel/m | Nominal calibration preserved without integer truncation |
| Nominal effective optical coefficient | 140 | N/m | Author-supplied assumption; not an independent measurement for this disk |
| Disk centre | 93.26971254206686,90.04002272424479 | pixel | Observation-only robust red-rim circle fit, row,column order |
| Disk radius | 72.92296540262036 | pixel | Observation-only fitted circular outline; no force target used |
| Fixed beta | 0,pi/2,pi,-pi/2 | rad | Four coarse geometric contact positions; no hidden force information |
| Polariscope | dark field | 1 | Sin-squared optical law |
| Image normalization | clip((green-0.5*red)/(255*0.65),0,1) | 1 | Red-channel leakage subtraction and recorded optical contrast convention |
| Thickness | absorbed into integrated quantities | 1 | No conversion from N/m to Pa is justified |
| Boundary model | circular disk, concentrated contacts | 1 | No stresses outside disk; finite-size contact effects are unmodelled |
| Noise distribution | uncalibrated real image | 1 | Detector/illumination/colour crosstalk not independently characterized |

The measured image is a crop of the PeGS2 example dataset. **Image data are CC BY-NC-SA4.0: noncommercial and sharealike conditions apply.** Attribution and software's separate MIT terms are preserved in NOTICE. No unrestricted commercial reuse or relicensing is asserted.

The standard implementation is an explicit physically constrained correction of the inverse-force method: it preserves the original elastic response and pixel mismatch, restores static force/torque balance and retains measured floating-point calibration. It does not claim unchanged reproduction of an erroneous optional upstream balance penalty.

## Method Hints

Use the published photoelastic inverse-contact approach: explicit elastic stress superposition followed by the nonlinear stress-optic law and pixelwise image mismatch minimization. The periodic optical law creates local minima, so initialization and parameter bounds matter. The published static force and torque constraints should be enforced explicitly. A constrained general optimizer can fit normal/tangential force components while preserving compressive normal loads and bounded force magnitudes. Excluding the outer4% of disk radius from the optical mismatch avoids concentrated-contact model singularities; the full stress map is still reconstructed.

Keep the contact positions fixed and fit both force magnitudes and angular departures. Compare unconstrained image fitting, multiple initial states and perturbations of geometry and optical coefficient to expose physical balance, calibration dependence and local minima. Do not infer an absolute hydrostatic stress from photoelastic intensity alone.

## References

Daniels, Kollmer and Puckett, *Photoelastic Force Measurements in Granular Materials*, Review of Scientific Instruments88,051808(2017), [doi:10.1063/1.4983049](https://doi.org/10.1063/1.4983049).

Abed Zadeh et al., *Enlightening Force Chains: A Review of Photoelasticimetry in Granular Matter*, Granular Matter21,83(2019), [doi:10.1007/s10035-019-0942-2](https://doi.org/10.1007/s10035-019-0942-2).

Lee et al., *Photoelastic Grain Solver v2.0: An updated tool for analysis of force measurements in granular materials*(2025), [arXiv:2507.01933](https://arxiv.org/abs/2507.01933). The published0.96-radius optical-fit convention is retained.
