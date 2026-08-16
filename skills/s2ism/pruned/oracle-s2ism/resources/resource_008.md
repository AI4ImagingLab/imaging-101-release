# Stopping behavior

The diagnostic quantities in the reference-style loop are:

```text
int_f = sum of plane Nz//2
int_bkg = total object sum - int_f
d_int_f = (new_int_f - old_int_f) / total_measured_counts
d_int_bkg = (new_int_bkg - old_int_bkg) / total_measured_counts
```

For this task, use fixed stopping rather than automatic stopping. In a Python loop this means:

```python
for k in range(50):
    X = update(X)
```

If reproducing the exact while-loop convention, note that the check occurs after computing a new iterate, and the saved result is the last updated object.
