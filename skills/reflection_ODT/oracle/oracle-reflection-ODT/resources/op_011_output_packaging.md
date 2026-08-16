Packaging notes

The submission array is the reconstructed RI contrast volume. Correct examples:

```
Path('outputs').mkdir(exist_ok=True)
np.savez('outputs/case_000.npz', delta_n=dn_recon.astype(np.float32))
```

or save float64 if the array is already float64 and finite. Incorrect transformations include adding the background index, inserting a leading batch axis, transposing to `(Ny,Nx,Nz)`, or saving intensities instead of `delta_n`.