# Whitaker-Hayes despiking (per spectrum, applied along last axis)

Parameters: kernel_size = 3, threshold = 8.

```
modified_z_score(values):
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    if mad == 0: return zeros_like(values)
    return 0.6745 * (values - median) / mad

despike_spectrum(spectrum, kernel_size, threshold):
    out = spectrum.copy()
    z = |modified_z_score(np.diff(out))|      # length B'-1
    spikes = z > threshold
    while any(spikes):
        changed = False
        for i in range(len(spikes)):
            if not spikes[i]: continue
            lo = max(0, i - kernel_size)
            hi = min(len(out) - 1, i + 1 + kernel_size)
            neighbours = arange(lo, hi)
            good = neighbours[ ~spikes[clip(neighbours, 0, len(spikes)-1)] ]
            if len(good) == 0: continue
            out[i] = mean(out[good])
            spikes[i] = False
            changed = True
        if not changed: break
    return out
```

Apply with an along-last-axis map over the (40,40,10,B') volume. The spikes array is indexed on diff space (length B'-1); neighbour indices are clipped into that range before lookup.