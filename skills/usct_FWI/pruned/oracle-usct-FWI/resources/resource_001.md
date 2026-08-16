# CBS domain setup details

Frequency convention: the solver receives frequency in 100 kHz units. Convert MHz via `freq_100k = freq_MHz * 10` before calling.

## Geometry / grids
- `pixel_size = lamb / ppw`  (lamb=1, ppw=8 -> 0.125)
- `bw = boundary_widths * ppw`  (20*8 = 160 cells each padded side, subject to FFT-friendly growth)
- `scale = dh / pixel_size`
- `speed_mean = mean(velocity)`
- `lamb_val = speed_mean / (freq_100k * scale)`
- `k0 = 2*pi / lamb_val`
- `refractive_index = speed_mean / velocity`; `e_r = refractive_index**2`
- `e_r_center = (max(e_r) + min(e_r)) / 2`; `k0c = sqrt(e_r_center) * k0`

## FFT-friendly padded size
Grow each padded axis size upward until `max(prime_factors) <= 11` AND (`<=2 distinct factors` OR `second_largest_factor <= 5`). ROI crop `[Bl, Br]` centers the original field: `Bl=ceil((all-orig)/2)`, `Br=floor((all-orig)/2)`. Pad the potential map by replicating edge rows/columns (left/bottom then right/top).

## Spatial and frequency grids
- `x=pixel_size*arange(Nx)`, similarly y,z reshaped for broadcasting.
- `dfx=2*pi/(pixel_size*Nx)` (and y,z); `fx = dfx * fftshift(arange(Nx) - floor(Nx/2))` reshaped; similarly fy,fz.

## Potential and preconditioner
- `V_tot = padded_e_r*k0^2 - k0c^2`
- `Vabs_max = max|V_tot|`; if real part of the argmax entry `< 0.05*Vabs_max`: `Vabs_max *= 1.05`
- `epsilon = max(Vabs_max, 3)`
- `V = padded_e_r*k0^2 - k0c^2 - 1j*epsilon`; then apply absorbing edge filters.
- `gamma = (1j/epsilon) * V`; expand dims for the singleton z axis.
- `mix_weight1 = 1 - gamma`; `mix_weight2 = 1j*gamma**2`

## Edge (absorbing boundary) filter
For each active dim with boundary `L=boundary_widths[dim]`: for 2D domains use window `w(B) = (arange(1,B+1) - 0.21)/(B + 0.66)`; build `[zeros(bl-L), smoothstep, ones(roi), flip(smoothstep), zeros(br-L)]` and multiply along that axis. When `br==bl`, `smoothstep=cat([0], w(L-1))` else `w(L)`.

## Wiggles
- `wiggle_set`: 4 permutations of +/-1 quarter-pixel shifts across the two active dims.
- `k02e = k0c^2/epsilon + 1j`
- Per wiggle `wig`: `pxe=(fx - dfx*wig0/4)/sqrt(epsilon)` (and y,z); `scaler_step1 = pxe^2+pye^2+pze^2 - k02e`.
- `gx = exp(1j*wig0*x * (pi/2)/(pixel_size*Nx))` (and y,z); `filter_step0 = gx*gy*gz`; `filter_step2 = conj(gx)*conj(gy)*conj(gz)`.

## Iteration budget
- `iterations_per_cycle = lamb_val / (2*k0c/epsilon)`; `max_iterations = ceil(born_max * iterations_per_cycle)`.
- Source injection amplitude `A = 1j/epsilon/n_wiggle`.
