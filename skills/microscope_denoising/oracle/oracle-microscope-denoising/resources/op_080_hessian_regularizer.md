# Hessian regularizer

```python
def hessian_loss(x):  # x: (B,1,H,W)
    dx = x[:, :, :, 1:] - x[:, :, :, :-1]
    dy = x[:, :, 1:, :] - x[:, :, :-1, :]
    dxx = dx[:, :, :, 1:] - dx[:, :, :, :-1]
    dyy = dy[:, :, 1:, :] - dy[:, :, :-1, :]
    dxy = dy[:, :, :, 1:] - dy[:, :, :, :-1]
    dyx = dx[:, :, 1:, :] - dx[:, :, :-1, :]
    return dxx.pow(2).mean() + dyy.pow(2).mean() + dxy.pow(2).mean() + dyx.pow(2).mean()
```
