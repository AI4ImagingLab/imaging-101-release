# Plane interpretation

- `reconstruction[0]`: first axial plane in the provided PSF stack.
- `reconstruction[1]`: second axial plane in the provided PSF stack.
- The second plane is not an error term; it represents out-of-focus/background fluorophore distribution.
- The two planes can have different total intensities. Do not force equal plane sums after reconstruction.

Diagnostic idea: compare `reconstruction.sum(axis=0)` to `measurements.sum(axis=-1)` only qualitatively; the deconvolved reconstruction is sharper and not on the same per-pixel scale as a raw channel-sum image.
