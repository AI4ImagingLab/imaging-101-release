"""Reference aspheric sag and derivatives (PyTorch, differentiable).

Run-local helper. Import these functions from the solver's forward model.
"""
import torch


def aspheric_sag(r2, c, k=0.0, ai=None):
    """z = c*r2/(1+sqrt(1-(1+k)c^2 r2)) + higher order in r2."""
    tmp = r2 * c
    total = tmp / (1.0 + torch.sqrt(1.0 - (1.0 + k) * tmp * c))
    if ai is not None:
        higher = 0.0
        for i in reversed(range(len(ai))):
            higher = r2 * higher + ai[i]
        total = total + higher * r2 ** 2
    return total


def aspheric_dsag_dr2(r2, c, k=0.0, ai=None):
    """d sag / d r2."""
    alpha = (1.0 + k) * c ** 2 * r2
    s = torch.sqrt(1.0 - alpha)
    d = c * (1.0 + s - 0.5 * alpha) / (s * (1.0 + s) ** 2)
    if ai is not None:
        higher = 0.0
        for i in reversed(range(len(ai))):
            higher = r2 * higher + (i + 2) * ai[i]
        d = d + higher * r2
    return d


def surface_normal(x, y, c, k=0.0, ai=None):
    r2 = x ** 2 + y ** 2
    dsdr2 = 2.0 * aspheric_dsag_dr2(r2, c, k, ai)
    n = torch.stack((dsdr2 * x, dsdr2 * y, -torch.ones_like(x)), dim=-1)
    return n / torch.sqrt(torch.sum(n ** 2, dim=-1, keepdim=True))


if __name__ == '__main__':
    r2 = torch.linspace(0.0, 100.0, 5)
    c = torch.tensor(0.01)
    print('sag', aspheric_sag(r2, c))
    print('dsag', aspheric_dsag_dr2(r2, c))
