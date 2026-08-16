Loss convention

For each angle `l`:

```
pred_intensity_l = forward_single_angle(delta_n)
pred_amp_l = sqrt(pred_intensity_l + 1e-12)
meas_amp_l = sqrt(measured_intensity_l)
loss_l = sum((pred_amp_l - meas_amp_l)^2) / (128*128)
```

The total differentiable data loss is the sum of `loss_l` over all rings and angles. This normalization is per image, not per all-angle stack.