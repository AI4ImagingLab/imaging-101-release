Illumination convention

The output of the forward model must be stacked in the same order as the measurement array:

1. Ring 0: NA=0.130, 8 azimuths, BF
2. Ring 1: NA=0.251, 16 azimuths, BF
3. Ring 2: NA=0.420, 16 azimuths, DF
4. Ring 3: NA=0.560, 16 azimuths, DF

For each ring use local angle index `m`, not the global index. Azimuth is uniformly sampled as `2*pi*m/n_ring`. The tilt is intentionally truncated with Python `int`, which truncates toward zero, before making the phase ramp. This removes nonperiodic boundary mismatch and is a task-critical convention.