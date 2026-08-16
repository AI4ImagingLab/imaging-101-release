"""Cheap health check for the flow-DA pipeline. Not a submission.

Usage: python script_002.py [checkpoint_path]
Runs with the run-local python. Verifies checkpoint load, model forward,
interpolant coefficients, and a few guided EM steps (tiny settings).
"""
import os
import sys
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from the related step import load_drift_model  # noqa: E402


class Interpolant:
    def __init__(self, sigma_coef=1.0):
        self.sigma_coef = sigma_coef

    def _wide(self, t):
        return t[:, None, None, None]

    def sigma(self, t):
        return self.sigma_coef * self._wide(1 - t)


def taylor_est(model, xt, t_wide, bF, cond, mc_times):
    v = 2.0 / 3.0 - t_wide.sqrt() + (1.0 / 3.0) * (t_wide.sqrt()) ** 3
    hat = xt + bF * (1 - t_wide) + torch.randn_like(xt) * v
    t1 = torch.ones(xt.shape[0], device=xt.device)
    bF2 = model(hat, t1, None, cond=cond).requires_grad_(True)
    return [(xt + (bF + bF2) / 2 * (1 - t_wide) + torch.randn_like(xt) * v).requires_grad_(True)
            for _ in range(mc_times)]


def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "public/visible_data/pretrained_models/latest.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_drift_model(ckpt, device)
    interp = Interpolant(1.0)

    cond = torch.randn(1, 6, 128, 128, device=device)
    mask = (torch.rand(1, 1, 128, 128, device=device) < 0.1).float()
    obs = torch.randn(1, 1, 128, 128, device=device) * mask
    operator = lambda x: x * mask

    n_steps, mc_times, guidance_scale = 4, 2, 0.1
    ts = torch.linspace(0.0, 0.999, n_steps)
    dt = ts[1] - ts[0]
    ones = torch.ones(1, device=device)
    xt = cond[:, -1:, :, :].clone().requires_grad_(True)
    mu = xt
    for tsi in ts:
        t = (tsi * ones).to(device)
        bF = model(xt, t, None, cond=cond).requires_grad_(True)
        sigma = interp.sigma(t)
        es = taylor_est(model, xt, interp._wide(t), bF, cond, mc_times)
        diff = sum(obs - operator(e) for e in es) / len(es)
        norm = torch.linalg.norm(diff)
        g = torch.autograd.grad(norm, xt, allow_unused=True)[0]
        if g is None:
            g = torch.zeros_like(xt)
        mu = xt + bF * dt
        xt = mu + sigma * torch.randn_like(mu) * dt.sqrt() - guidance_scale * g
    assert tuple(mu.shape) == (1, 1, 128, 128), mu.shape
    print("smoke ok:", tuple(mu.shape), "device", device)


if __name__ == "__main__":
    main()
