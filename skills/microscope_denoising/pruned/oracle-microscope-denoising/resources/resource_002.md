# Sliding-window inference

```python
import numpy as np, torch

def sw_infer(model, y_norm, patch_size=128, overlap=32, batch_size=8, device='cpu'):
    H, W = y_norm.shape; stride = patch_size - overlap
    patches, pos = [], []
    def add(i, j):
        patches.append(y_norm[i:i+patch_size, j:j+patch_size][None, None]); pos.append((i, j))
    for i in range(0, H - patch_size + 1, stride):
        for j in range(0, W - patch_size + 1, stride): add(i, j)
    if H > patch_size:
        for j in range(0, W - patch_size + 1, stride): add(H - patch_size, j)
    if W > patch_size:
        for i in range(0, H - patch_size + 1, stride): add(i, W - patch_size)
    if H > patch_size and W > patch_size: add(H - patch_size, W - patch_size)
    xb = torch.from_numpy(np.concatenate(patches, 0)).float()
    model.to(device).eval(); res = []
    with torch.no_grad():
        for b in range(0, len(xb), batch_size):
            res.append(model(xb[b:b+batch_size].to(device)).cpu().numpy())
    model.cpu()
    preds = np.concatenate(res, 0)[:, 0]
    out = np.zeros((H, W)); wgt = np.zeros((H, W))
    h = np.hanning(patch_size); h = 0.1 + 0.9 * h; blend = np.outer(h, h)
    for p, (i, j) in zip(preds, pos):
        out[i:i+patch_size, j:j+patch_size] += p * blend
        wgt[i:i+patch_size, j:j+patch_size] += blend
    return out / np.maximum(wgt, 1e-6)

def restore(f_den, f_dec, y, patch_size=128, overlap=32, device='cpu'):
    lo = float(np.percentile(y, 0)); hi = float(np.percentile(y, 100))
    y_norm = np.clip((y - lo)/(hi - lo + 1e-6), 0, 1).astype(np.float32)
    den_norm = sw_infer(f_den, y_norm, patch_size, overlap, device=device)
    denoised = (den_norm*(hi - lo) + lo).astype(np.float32)
    dec_in = np.clip(den_norm, 0, 1).astype(np.float32)
    dec_norm = sw_infer(f_dec, dec_in, patch_size, overlap, device=device)
    deconvolved = (dec_norm*(hi - lo) + lo).astype(np.float32)
    return denoised, deconvolved
```
