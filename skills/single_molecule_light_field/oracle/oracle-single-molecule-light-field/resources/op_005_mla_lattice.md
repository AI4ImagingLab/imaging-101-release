# Hexagonal lattice reference

With optic_size=10000, pitch=2390: width=4.184, marks = arange(-2, 4) = [-2,-1,0,1,2,3] (6 values), 36 raw centres generated. Only 7 fall inside the BFP after the aperture filter.

Hexagonal placement (lattice-spacing units):
  x = yy * sqrt(3)/2
  y = xx + dx.T + 0.5
where dx is a [length, ceil(length/2)] tile of [0.5, 0] truncated to [length, length], giving the half-row stagger. Flatten with order 'F'.

Rotation matrix (theta = 30.8 deg) applied about centre [0,0]:
  [cos -sin; sin cos].

The 7 in-aperture lens centres in (u,v) units (centres*mla_to_uv_scale) are approximately:
  ( 0.0000,  0.0000)
  ( 0.6452,  0.0090)
  (-0.6452, -0.0090)
  ( 0.3304, -0.5542)
  (-0.3304,  0.5542)
  ( 0.3148,  0.5632)
  (-0.3148, -0.5632)
These are the 7 hexagonal views (1 central + 6 surrounding).
