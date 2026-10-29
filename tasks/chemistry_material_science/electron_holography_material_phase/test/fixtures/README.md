# Frozen numerical oracle

`upstream.npz` is consumed by `test/comparing_test/test_parity.py` and the
maintainer notebook. It was captured before local implementation by unmodified
HoloSpy v0.3 `holospy/reconstruct.py` on the exact bundled object and independently
acquired vacuum observations. Stage tolerances were fixed before extraction:
frequency absolute 1e-15 nm^-1, radius 1e-12 pixel, complex wave relative 1e-11 /
absolute 1e-10 detector units, corrected complex output relative 1e-10 /
absolute 1e-11. These account only for NumPy versus SciPy FFT roundoff.

Keys: `position` FFT-bin carrier indices; `radius`/`smoothness` aperture parameters;
`frequencies` radial frequencies in nm^-1; `object_wave`/`vacuum_wave` complex
demodulated intensities; their quotient supplies the full-pipeline oracle. They are not
ground truth. The full capture environment, hashes, source and archive are
indexed in `test/validation.md`. The complete wave arrays are needed for
pixelwise phase-sign and boundary parity tests.

Before local extraction, independent original-code double and single-precision
runs measured relative L2 disagreement 0 and 9.561154009425683e-7. The evaluator
will require relative complex L2 <= 1e-5 (about 10.5 times measured single
precision roundoff), plus observation-sideband relative consistency <= 1e-5.
Neither criterion estimates physical error of the standard method.
