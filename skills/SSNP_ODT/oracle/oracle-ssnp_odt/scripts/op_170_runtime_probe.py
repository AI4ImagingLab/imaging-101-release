"""Self-contained smoke test for the SSNP forward model + one autograd step.

Uses a tiny synthetic volume (no task data, no I/O) to verify that the P/Q
operators, camera extraction, amplitude loss, and gradient step all run and
produce finite, sensible values. Run from the eval run root:
    python scripts/op_170_runtime_probe.py
"""
import numpy as np
import torch


def build_grids(Ny, Nx, res, device):
    fx = torch.fft.fftfreq(Nx, dtype=torch.float64, device=device) / res[0]
    fy = torch.fft.fftfreq(Ny, dtype=torch.float64, device=device) / res[1]
    FY, FX = torch.meshgrid(fy, fx, indexing="ij")
    c_gamma = torch.sqrt(torch.clamp(1.0 - FX**2 - FY**2, min=1e-8))
    kz = c_gamma * (2.0 * np.pi * res[2])
    eva = torch.exp(torch.clamp((c_gamma - 0.2) * 5.0, max=0.0))
    na_norm = 0.65
    cutoff = np.sqrt(1.0 - na_norm**2)
    pmask = torch.exp(torch.clamp(c_gamma - cutoff, max=0.01) * 10000.0)
    pupil = (pmask / (1.0 + pmask)).to(torch.complex128)
    return c_gamma, kz, eva, pupil


def apply_P(u, ud, kz, eva, dz=1.0):
    cos_kz = torch.cos(kz * dz) * eva
    sin_kz = torch.sin(kz * dz) * eva
    a = torch.fft.fft2(u); a_d = torch.fft.fft2(ud)
    a_new = cos_kz * a + (sin_kz / kz) * a_d
    a_d_new = (-kz * sin_kz) * a + cos_kz * a_d
    return torch.fft.ifft2(a_new), torch.fft.ifft2(a_d_new)


def apply_Q(u, ud, dn_slice, res_z, n0, dz=1.0):
    pf = (2.0 * np.pi * res_z / n0) ** 2 * dz
    scatter = pf * dn_slice * (2.0 * n0 + dn_slice)
    return u, ud - scatter * u


def main():
    device = "cpu"
    Nz, Ny, Nx = 6, 16, 16
    n0 = 1.0
    res = (0.2, 0.2, 0.2)
    c_gamma, kz, eva, pupil = build_grids(Ny, Nx, res, device)

    # incident plane wave (angle 0)
    tilt = torch.ones(Ny, Nx, dtype=torch.complex128, device=device)
    kz_in = 2.0 * np.pi * res[2]
    u0, ud0 = tilt.clone(), tilt * (1j * kz_in)

    dn = torch.zeros(Nz, Ny, Nx, dtype=torch.float64, device=device, requires_grad=True)

    def forward_single(dn):
        u, ud = u0.clone(), ud0.clone()
        for iz in range(Nz):
            u, ud = apply_P(u, ud, kz, eva, 1.0)
            u, ud = apply_Q(u, ud, dn[iz], res[2], n0, 1.0)
        u, ud = apply_P(u, ud, kz, eva, -Nz / 2.0)
        a = torch.fft.fft2(u); a_d = torch.fft.fft2(ud)
        a_f = (a - 1j * a_d / kz) * 0.5
        uf = torch.fft.ifft2(a_f)
        phi = torch.fft.ifft2(torch.fft.fft2(uf) * pupil)
        return torch.abs(phi) ** 2

    I0 = forward_single(dn)
    assert torch.isfinite(I0).all(), "non-finite forward output"
    assert (I0 >= 0).all(), "negative intensity"
    print("forward ok: I range [%.4f, %.4f]" % (I0.min().item(), I0.max().item()))

    # fake measurement from a small scatterer
    with torch.no_grad():
        dn_true = torch.zeros_like(dn); dn_true[Nz // 2, Ny // 2, Nx // 2] = 0.01
        meas_amp = torch.sqrt(forward_single(dn_true) + 1e-12)

    n_pixels = Ny * Nx
    losses = []
    for step in range(3):
        pred_amp = torch.sqrt(forward_single(dn) + 1e-12)
        loss = torch.sum((pred_amp - meas_amp) ** 2) / n_pixels
        loss.backward()
        assert torch.isfinite(dn.grad).all(), "non-finite grad"
        with torch.no_grad():
            dn -= 50.0 * dn.grad
            dn.clamp_(min=0.0)
            dn.grad.zero_()
        losses.append(loss.item())
        print("step %d loss=%.6e" % (step, losses[-1]))
    assert losses[-1] <= losses[0] + 1e-12, "loss did not decrease"
    print("PROBE PASS")


if __name__ == "__main__":
    main()
