"""Tiny smoke test for the SSNP forward model + one gradient step.

Self-contained, CPU-friendly, runs from the eval run root. Mirrors the
normalized-resolution and operator conventions used by the full pipeline.
"""
import numpy as np
import torch

PI = np.pi


def build_grids(Ny, Nx, res):
    fx = torch.fft.fftfreq(Nx, dtype=torch.float64) / res[0]
    fy = torch.fft.fftfreq(Ny, dtype=torch.float64) / res[1]
    FY, FX = torch.meshgrid(fy, fx, indexing="ij")
    c_gamma = torch.sqrt(torch.clamp(1.0 - FX**2 - FY**2, min=1e-8))
    kz = c_gamma * (2.0 * PI * res[2])
    eva = torch.exp(torch.clamp((c_gamma - 0.2) * 5.0, max=0.0))
    return c_gamma, kz, eva


def incident(m, n_angles, NA, n0, Ny, Nx, res):
    na = NA / n0
    th = 2 * PI * m / n_angles
    c_a, c_b = na * np.cos(th), na * np.sin(th)
    nxr, nyr = Nx * res[0], Ny * res[1]
    c_a = int(c_a * nxr) / nxr
    c_b = int(c_b * nyr) / nyr
    x = torch.arange(Nx, dtype=torch.float64)
    y = torch.arange(Ny, dtype=torch.float64)
    px = torch.exp(2j * PI * c_a * res[0] * x)
    py = torch.exp(2j * PI * c_b * res[1] * y)
    tilt = (py[:, None] * px[None, :]).to(torch.complex128)
    cg = np.sqrt(max(1 - c_a**2 - c_b**2, 1e-8))
    kz_in = cg * (2 * PI * res[2])
    return tilt.clone(), tilt * (1j * kz_in)


def propagate(u, ud, kz, eva, dz):
    cos_kz = torch.cos(kz * dz) * eva
    sin_kz = torch.sin(kz * dz) * eva
    a, a_d = torch.fft.fft2(u), torch.fft.fft2(ud)
    a_n = cos_kz * a + (sin_kz / kz) * a_d
    a_dn = (-kz * sin_kz) * a + cos_kz * a_d
    return torch.fft.ifft2(a_n), torch.fft.ifft2(a_dn)


def scatter(u, ud, dn_slice, n0, res, dz=1.0):
    pf = (2 * PI * res[2] / n0) ** 2 * dz
    s = pf * dn_slice * (2.0 * n0 + dn_slice)
    return u, ud - s * u


def forward_single(dn, m, cfg):
    Ny, Nx = cfg["Ny"], cfg["Nx"]
    kz, eva, pupil = cfg["kz"], cfg["eva"], cfg["pupil"]
    u, ud = incident(m, cfg["n_angles"], cfg["NA"], cfg["n0"], Ny, Nx, cfg["res"])
    for iz in range(cfg["Nz"]):
        u, ud = propagate(u, ud, kz, eva, 1.0)
        u, ud = scatter(u, ud, dn[iz], cfg["n0"], cfg["res"])
    u, ud = propagate(u, ud, kz, eva, -cfg["Nz"] / 2.0)
    a, a_d = torch.fft.fft2(u), torch.fft.fft2(ud)
    a_f = (a - 1j * a_d / kz) * 0.5
    uf = torch.fft.ifft2(a_f)
    phi = torch.fft.ifft2(torch.fft.fft2(uf) * pupil)
    return torch.abs(phi) ** 2


def main():
    Nz, Ny, Nx, n_angles = 8, 32, 32, 2
    NA, n0, res = 0.65, 1.0, (0.2, 0.2, 0.2)
    c_gamma, kz, eva = build_grids(Ny, Nx, res)
    na_norm = NA / n0
    cutoff = np.sqrt(1.0 - na_norm**2)
    mask = torch.exp(torch.clamp(c_gamma - cutoff, max=0.01) * 10000.0)
    pupil = (mask / (1.0 + mask)).to(torch.complex128)
    cfg = dict(Nz=Nz, Ny=Ny, Nx=Nx, n_angles=n_angles, NA=NA, n0=n0,
              res=res, kz=kz, eva=eva, pupil=pupil)

    true = torch.zeros(Nz, Ny, Nx, dtype=torch.float64)
    true[Nz // 2, Ny // 2 - 2:Ny // 2 + 2, Nx // 2 - 2:Nx // 2 + 2] = 0.01
    meas_amp = torch.stack([torch.sqrt(forward_single(true, m, cfg) + 1e-12)
                            for m in range(n_angles)])
    assert torch.all(torch.isfinite(meas_amp)), "non-finite forward output"

    dn = torch.zeros(Nz, Ny, Nx, dtype=torch.float64, requires_grad=True)
    losses = []
    for step in range(3):
        tot = 0.0
        for m in range(n_angles):
            pred = torch.sqrt(forward_single(dn, m, cfg) + 1e-12)
            lm = torch.sum((pred - meas_amp[m]) ** 2) / (Ny * Nx)
            lm.backward()
            tot += lm.item()
        with torch.no_grad():
            dn -= 50.0 * dn.grad
            dn.clamp_(min=0.0)
            dn.grad.zero_()
        losses.append(tot)
    assert losses[-1] <= losses[0] + 1e-12, f"loss did not decrease: {losses}"
    out = dn.detach().cpu().numpy()[None, ...]
    assert out.shape == (1, Nz, Ny, Nx)
    print("smoke OK, losses=", losses, "out shape=", out.shape)


if __name__ == "__main__":
    main()
