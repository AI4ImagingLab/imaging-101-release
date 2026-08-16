# Optional metric formulas

For estimate `E` and comparison `Y`, both `(T,C,H,W)`:

```text
NCC_c = dot(E_c - mean(E_c), Y_c - mean(Y_c)) / (||E_c - mean(E_c)|| * ||Y_c - mean(Y_c)||)
NRMSE_c = sqrt(mean((E_c - Y_c)^2)) / (max(Y_c) - min(Y_c))
wNRMSE_c = mean_t ||(E[t,c]-Y[t,c]) * W[c]||_2 / ||Y[t,c] * W[c]||_2
```

Use a small denominator guard if implementing for arbitrary arrays. The metric weights are for diagnostics only and are not part of the output tensor.
