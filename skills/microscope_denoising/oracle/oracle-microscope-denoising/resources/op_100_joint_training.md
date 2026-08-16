# Joint training

```python
import numpy as np, torch, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

def train(y_hat, y_bar, psf, f_den, f_dec, psf_blur, hessian_loss,
          n_iters=30000, batch_size=4, lr=5e-4, mu=0.5, hess_weight=0.02,
          lr_decay_steps=10000, lr_decay_factor=0.5, device='cpu'):
    params = list(f_den.parameters()) + list(f_dec.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    sch = torch.optim.lr_scheduler.StepLR(opt, step_size=lr_decay_steps, gamma=lr_decay_factor)
    ds = TensorDataset(torch.from_numpy(y_hat).float(), torch.from_numpy(y_bar).float())
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)
    it = iter(loader); f_den.train(); f_dec.train()
    hist = []
    for step in range(n_iters):
        try: xb, yb = next(it)
        except StopIteration: it = iter(loader); xb, yb = next(it)
        xb, yb = xb.to(device), yb.to(device)
        out_den = f_den(xb)
        loss_den = F.mse_loss(out_den, yb)
        out_dec = f_dec(out_den)
        loss_dec = F.mse_loss(psf_blur(out_dec), yb) + hess_weight * hessian_loss(out_dec)
        loss = mu * loss_den + (1 - mu) * loss_dec
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        if step % 100 == 0:
            hist.append((loss.item(), loss_den.item(), loss_dec.item()))
    return f_den.cpu().eval(), f_dec.cpu().eval(), hist
```

Constants: mu=0.5, hess_weight=0.02, lr=5e-4, StepLR(10000, 0.5), batch=4, iters=30000.
