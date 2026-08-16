# OTF details

generate_otf(n=512, na=0.9, wavelength=488, pixel_size=65).

The cone is a linear taper from 1 at DC to 0 at the cutoff radius `fc_pixels`, clipped to [0,1], then divided by its max.

Downstream radius cutoffs applied to copies of this OTF (all scaled by n/512):
- parameter estimation support: fc = ceil(220 * n/512)
- modulation phase support (wide): fc_ang = ceil(120 * n/512)
- modulation depth support (narrow): fc_con = ceil(105 * n/512) (or ceil(80*n/512) when wavelength==647)
- Wiener support: fc = ceil(200 * n/512)

Each cutoff produces `H` (masked magnitude OTF) and a binary support `H1 = (H!=0)`.
