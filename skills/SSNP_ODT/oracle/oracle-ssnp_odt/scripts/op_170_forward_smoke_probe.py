"""Cheap SSNP convention sanity probe (diagnostic only).

Reimplements the operators on a tiny grid to verify conventions are wired
correctly. Self-contained; no external data needed.
"""
import numpy as np
import torch


def build_grid(nx, ny, res):
    fx = torch.fft.fftfreq(nx, dtype=torch.float64) / res[0]
    fy = torch.fft.fftfreq(ny, dtype=torch.float64) / res[1]
    FY, FX = torch.meshgrid(fy, fx, indexing="ij")
    c_gamma = torch.sqrt(torch.clamp(1.0 - FX**2 - FY**2, min=1e-8))
    kz = c_gamma * (2.0 * np.pi * res[2])
    eva = torch.exp(torch.clamp((c_gamma - 0.2) * 5.0, max=0.0))
    return kz, eva


def P(u, ud, kz, eva, dz):
    ck = torch.cos(kz * dz) * eva
    sk = torch.sin(kz * dz) * eva
    a = torch.fft.fft2(u); ad = torch.fft.fft2(ud)
    an = ck * a + (sk / kz) * ad
    adn = (-kz * sk) * a + ck * ad
    return torch.fft.ifft2(an), torch.fft.ifft2(adn)


def Q(u, ud, dn2d, res, n0, dz=1.0):
    pf = (2.0 * np.pi * res[2] / n0) ** 2 * dz
    return u, ud - pf * dn2d * (2.0 * n0 + dn2d) * u


def main():
    nx = ny = 32
    nz = 8
    n0 = 1.0
    res = (0.2, 0.2, 0.2)
    kz, eva = build_grid(nx, ny, res)

    u = torch.ones(ny, nx, dtype=torch.complex128)
    ud = u * (1j * kz.mean())

    # 1) P with dz=0 is identity
    u0, ud0 = P(u, ud, kz, eva, dz=0.0)
    assert torch.allclose(u0, u, atol=1e-9), "P(dz=0) not identity on u"
    assert torch.allclose(ud0, ud, atol=1e-9), "P(dz=0) not identity on ud"

    # 2) zero-volume forward stays finite
    uu, uud = u.clone(), ud.clone()
    dn = torch.zeros(nz, ny, nx, dtype=torch.float64)
    for iz in range(nz):
        uu, uud = P(uu, uud, kz, eva, dz=1.0)
        uu, uud = Q(uu, uud, dn[iz], res, n0)
    assert torch.isfinite(uu).all(), "zero-volume forward produced non-finite"

    # 3) positive slab changes the field and stays finite
    dn2 = dn.clone(); dn2[nz // 2] = 0.01
    vv, vvd = u.clone(), ud.clone()
    for iz in range(nz):
        vv, vvd = P(vv, vvd, kz, eva, dz=1.0)
        vv, vvd = Q(vv, vvd, dn2[iz], res, n0)
    assert torch.isfinite(vv).all(), "scattered forward produced non-finite"
    assert not torch.allclose(vv, uu), "scattering had no effect"

    print("SSNP smoke probe PASS: P(dz=0)=I, forward finite, scattering active")


if __name__ == "__main__":
    main()
