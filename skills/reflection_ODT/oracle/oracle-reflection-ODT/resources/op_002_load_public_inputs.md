Public input loading notes

Expected metadata fields:
- `volume_shape`: `[4, 128, 128]`
- `res_um`: `[0.274, 0.274, 1.0]`
- `wavelength_um`: `0.632`
- `n0`: `1.54`
- `NA_obj`: `0.28`
- `illumination_rings`: four dictionaries in acquisition order
- `dz_layer`: `0.5`
- `dz_gap`: `10.0`

Measurements:
- File: `public/visible_data/cases/case_000/input_data/raw_data.npz`
- Key: `measurements`
- Raw shape: `(1, 56, 128, 128)`
- Solver shape after indexing batch 0: `(56, 128, 128)`
- Internal numeric dtype: float64 is preferred for stability.