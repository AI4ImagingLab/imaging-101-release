"""
SSNP Forward Model for Intensity Diffraction Tomography
=========================================================

Physical model (Split-Step Non-Paraxial):

    The SSNP model propagates a state vector Φ = (φ, ∂φ/∂z) through
    the 3D sample by alternating two operators on each axial slice:

    P (diffraction): free-space propagation in Fourier space
        ┌ cos(kz·Δz)       sin(kz·Δz)/kz ┐   ┌ â  ┐
        │                                  │ · │    │
        └ -kz·sin(kz·Δz)   cos(kz·Δz)    ┘   └ â_d┘

    Q (scattering): local RI modulation in real space
        u_d -= k0²·(n0² - n²(x,y,z))·Δz · u

    The output intensity is:  I = |F · P_NA · P_Δzf · Φ(zn)|²
    where F extracts the forward-propagating component and P_NA
    is the pupil function.

Implementation uses PyTorch for GPU acceleration and autograd support.
"""

from dataclasses import dataclass
import numpy as np
import torch


@dataclass
class SSNPConfig:
    """
    Physical parameters for the SSNP-IDT model.

    Attributes
    ----------
    volume_shape : tuple of int
        (Nz, Ny, Nx) — number of voxels in each dimension.
    res : tuple of float
        Normalised resolution (dx·n0/λ, dy·n0/λ, dz·n0/λ).
    n0 : float
        Background refractive index.
    NA : float
        Objective numerical aperture.
    wavelength_um : float
        Illumination wavelength in micrometers.
    res_um : tuple of float
        Voxel size in micrometers (dx, dy, dz).
    n_angles : int
        Number of illumination angles.
    """
    volume_shape: tuple
    res: tuple
    n0: float
    NA: float
    wavelength_um: float
    res_um: tuple
    n_angles: int

    @classmethod
    def from_metadata(cls, metadata: dict) -> "SSNPConfig":
        """Construct SSNPConfig from a metadata dictionary.

        The official ``ssnp`` codebase parameterises propagation using the
        dimensionless sampling

            res = xyz / lambda0 * n0

        where ``xyz`` is the physical voxel size.  If metadata stores voxel
        size in micrometres, it must therefore be converted before use.
        """
        n0 = metadata["n0"]
        wavelength = metadata["wavelength_um"]
        res_um = tuple(metadata["res_um"])
        res = tuple(size * n0 / wavelength for size in res_um)
        return cls(
            volume_shape=tuple(metadata["volume_shape"]),
            res=res,
            n0=n0,
            NA=metadata["NA"],
            wavelength_um=wavelength,
            res_um=res_um,
            n_angles=metadata["n_angles"],
        )


class SSNPForwardModel:
    """
    SSNP-based intensity diffraction tomography forward model.

    Implements the split-step non-paraxial propagation through a 3D RI
    distribution and computes resulting intensity images for multiple
    illumination angles.

    Parameters
    ----------
    config : SSNPConfig
        Physical parameters.
    device : str
        PyTorch device ('cpu' or 'cuda').
    """

    def __init__(self, config: SSNPConfig, device: str = "cpu"):
        self.config = config
        self.device = torch.device(device)
        self.dtype = torch.complex128
        self.real_dtype = torch.float64

        nz, ny, nx = config.volume_shape
        self.nz = nz
        self.ny = ny
        self.nx = nx

        # Precompute frequency-domain quantities
        self.kz = self._compute_kz()
        self.eva_mask = self._compute_evanescent_mask()
        self.pupil = self._compute_pupil()

        # c_gamma (direction cosine) for frequency grid
        self.c_gamma = self._compute_c_gamma()

    # ──────────────────────────────────────────────────────────────────────
    # Precomputation
    # ──────────────────────────────────────────────────────────────────────

    def _compute_c_gamma(self) -> torch.Tensor:
        """
        Compute cos(gamma) = sqrt(1 - sin²α - sin²β) on the 2D frequency grid.

        Returns shape (Ny, Nx) float64 tensor.
        """
        res = self.config.res
        fx = torch.fft.fftfreq(self.nx, dtype=self.real_dtype, device=self.device) / res[0]
        fy = torch.fft.fftfreq(self.ny, dtype=self.real_dtype, device=self.device) / res[1]
        FY, FX = torch.meshgrid(fy, fx, indexing="ij")

        eps = 1e-8
        c_gamma = torch.sqrt(torch.clamp(1.0 - FX**2 - FY**2, min=eps))
        return c_gamma

    def _compute_kz(self) -> torch.Tensor:
        """
        Compute kz = c_gamma * (2π · res_z) on the 2D frequency grid.

        kz is the axial spatial frequency in normalised units, used in
        the P operator's rotation matrix.

        Returns shape (Ny, Nx) float64 tensor.
        """
        c_gamma = self._compute_c_gamma()
        kz = c_gamma * (2.0 * np.pi * self.config.res[2])
        return kz.to(self.device)

    def _compute_evanescent_mask(self) -> torch.Tensor:
        """
        Damping mask for evanescent waves.

        exp(min((c_gamma - 0.2) * 5, 0))

        Returns shape (Ny, Nx) float64 tensor.
        """
        c_gamma = self._compute_c_gamma()
        mask = torch.exp(torch.clamp((c_gamma - 0.2) * 5.0, max=0.0))
        return mask.to(self.device)

    def _compute_pupil(self) -> torch.Tensor:
        """
        Binary pupil function with smooth cutoff at NA.

        Uses a sigmoid-like function centred at c_gamma = sqrt(1 - (NA/n0)²).

        Returns shape (Ny, Nx) complex128 tensor.
        """
        c_gamma = self._compute_c_gamma()
        na_norm = self.config.NA / self.config.n0
        cutoff = np.sqrt(1.0 - na_norm**2)
        mask = torch.exp(torch.clamp(c_gamma - cutoff, max=0.01) * 10000.0)
        mask = mask / (1.0 + mask)
        return mask.to(dtype=self.dtype, device=self.device)

    # ──────────────────────────────────────────────────────────────────────
    # Incident field
    # ──────────────────────────────────────────────────────────────────────

    def _make_incident_field(self, angle_idx: int) -> tuple:
        """
        Construct a tilted plane wave and its z-derivative for the given
        illumination angle.

        The illumination direction is:
            kx_in = NA · cos(2π·m/n_angles)
            ky_in = NA · sin(2π·m/n_angles)

        The tilt is applied by truncating to the nearest integer frequency
        to maintain periodicity (matching the original code's `trunc=True`).

        Parameters
        ----------
        angle_idx : int
            Index m of the illumination angle (0 to n_angles-1).

        Returns
        -------
        u  : (Ny, Nx) complex128 tensor — tilted plane wave
        ud : (Ny, Nx) complex128 tensor — z-derivative of the field
        """
        na = self.config.NA / self.config.n0
        n_angles = self.config.n_angles
        res = self.config.res

        theta = 2.0 * np.pi * angle_idx / n_angles
        c_a = na * np.cos(theta)  # direction cosine along x
        c_b = na * np.sin(theta)  # direction cosine along y

        # Truncate to nearest integer frequency for periodicity
        norm_x = self.nx * res[0]
        norm_y = self.ny * res[1]
        c_a = int(c_a * norm_x) / norm_x
        c_b = int(c_b * norm_y) / norm_y

        # Pixel coordinates
        x = torch.arange(self.nx, dtype=self.real_dtype, device=self.device)
        y = torch.arange(self.ny, dtype=self.real_dtype, device=self.device)

        # Phase ramps
        phase_x = 2j * np.pi * c_a * res[0] * x
        phase_y = 2j * np.pi * c_b * res[1] * y

        tilt = torch.exp(phase_y).unsqueeze(1) * torch.exp(phase_x).unsqueeze(0)
        tilt = tilt.to(dtype=self.dtype, device=self.device)

        # Compute kz for this illumination direction
        c_gamma_in = np.sqrt(max(1.0 - c_a**2 - c_b**2, 1e-8))
        kz_in = c_gamma_in * (2.0 * np.pi * res[2])

        u = tilt.clone()
        ud = tilt * (1j * kz_in)

        return u, ud

    # ──────────────────────────────────────────────────────────────────────
    # SSNP operators
    # ──────────────────────────────────────────────────────────────────────

    def _apply_propagation(self, u: torch.Tensor, ud: torch.Tensor,
                           dz: float = 1.0) -> tuple:
        """
        P operator: free-space propagation by dz slices.

        In Fourier space, applies the 2×2 rotation matrix:
            â_new  =  cos(kz·dz) · â  + sin(kz·dz)/kz · â_d
            â_d_new = -kz·sin(kz·dz) · â + cos(kz·dz) · â_d

        Evanescent components are damped by the evanescent mask.

        Parameters
        ----------
        u  : (Ny, Nx) complex tensor — field
        ud : (Ny, Nx) complex tensor — z-derivative
        dz : float — propagation distance in slice units

        Returns
        -------
        u_new, ud_new : propagated field and derivative
        """
        kz = self.kz
        eva = self.eva_mask

        cos_kz = torch.cos(kz * dz) * eva
        sin_kz = torch.sin(kz * dz) * eva

        a = torch.fft.fft2(u)
        a_d = torch.fft.fft2(ud)

        # 2×2 matrix multiply
        a_new = cos_kz * a + (sin_kz / kz) * a_d
        a_d_new = (-kz * sin_kz) * a + cos_kz * a_d

        u_new = torch.fft.ifft2(a_new)
        ud_new = torch.fft.ifft2(a_d_new)

        return u_new, ud_new

    def _apply_scattering(self, u: torch.Tensor, ud: torch.Tensor,
                          dn_slice: torch.Tensor, dz: float = 1.0) -> tuple:
        """
        Q operator: scattering by a single RI slice.

        ud -= k0² · (2·n0·Δn + Δn²) · Δz_physical · u

        where Δz_physical = (2π·res_z / n0)² · dz is the normalised
        physical thickness, and k0 = 2π/λ.

        Parameters
        ----------
        u        : (Ny, Nx) complex — field
        ud       : (Ny, Nx) complex — z-derivative
        dn_slice : (Ny, Nx) real — RI contrast Δn at this z slice
        dz       : float — slice thickness in normalised units

        Returns
        -------
        u, ud_new : field unchanged, derivative updated
        """
        n0 = self.config.n0
        res_z = self.config.res[2]

        # Phase factor matching the original code:
        # phase_factor = (2π·res_z / n0)² · dz
        phase_factor = (2.0 * np.pi * res_z / n0) ** 2 * dz

        # Scattering potential: k0²(n0² - n²) ≈ -k0²(2·n0·Δn + Δn²)
        scatter = phase_factor * dn_slice * (2.0 * n0 + dn_slice)

        ud_new = ud - scatter * u

        return u, ud_new

    # ──────────────────────────────────────────────────────────────────────
    # Field extraction at the camera plane
    # ──────────────────────────────────────────────────────────────────────

    def _split_forward_backward(self, u: torch.Tensor,
                                ud: torch.Tensor) -> tuple:
        """
        Decompose (field, derivative) into forward and backward components.

        In Fourier space:
            â_f = (â + â_d / (j·kz)) / 2
            â_b = â - â_f

        Parameters
        ----------
        u  : (Ny, Nx) complex — total field
        ud : (Ny, Nx) complex — z-derivative

        Returns
        -------
        uf, ub : forward and backward component fields
        """
        kz = self.kz

        a = torch.fft.fft2(u)
        a_d = torch.fft.fft2(ud)

        # Forward component in Fourier space
        a_f = (a - 1j * a_d / kz) * 0.5
        a_b = a - a_f

        uf = torch.fft.ifft2(a_f)
        ub = torch.fft.ifft2(a_b)

        return uf, ub

    def _extract_forward_component(self, u: torch.Tensor,
                                   ud: torch.Tensor) -> torch.Tensor:
        """
        Back-propagate to focal plane, apply pupil, extract forward component.

        1. Back-propagate from exit plane (z=zn) to focal plane (z=zn/2)
           by propagating dz = -nz/2 slices (no scattering)
        2. Discard backward component
        3. Apply binary pupil filter in Fourier space

        Parameters
        ----------
        u  : (Ny, Nx) complex — field at exit plane
        ud : (Ny, Nx) complex — z-derivative at exit plane

        Returns
        -------
        phi_out : (Ny, Nx) complex — forward field at camera plane
        """
        # Back-propagate to focal plane
        u_bp, ud_bp = self._apply_propagation(u, ud, dz=-self.nz / 2.0)

        # Extract forward component only
        uf, _ = self._split_forward_backward(u_bp, ud_bp)

        # Apply pupil in Fourier space
        a_f = torch.fft.fft2(uf)
        a_f = a_f * self.pupil
        phi_out = torch.fft.ifft2(a_f)

        return phi_out

    # ──────────────────────────────────────────────────────────────────────
    # Full forward model
    # ──────────────────────────────────────────────────────────────────────

    def forward_single(self, dn_volume: torch.Tensor,
                       angle_idx: int) -> torch.Tensor:
        """
        Simulate intensity image for a single illumination angle.

        Pipeline:
            1. Construct incident tilted plane wave
            2. Propagate through sample (P·Q for each slice)
            3. Back-propagate, pupil filter, extract forward
            4. Take intensity |φ_out|²

        Parameters
        ----------
        dn_volume : (Nz, Ny, Nx) real tensor — RI contrast volume
        angle_idx : int — illumination angle index

        Returns
        -------
        intensity : (Ny, Nx) real tensor
        """
        u, ud = self._make_incident_field(angle_idx)

        # Propagate through each slice: P then Q
        for iz in range(self.nz):
            u, ud = self._apply_propagation(u, ud, dz=1.0)
            u, ud = self._apply_scattering(u, ud, dn_volume[iz], dz=1.0)

        # Extract forward component at camera
        phi_out = self._extract_forward_component(u, ud)

        # Intensity
        intensity = torch.abs(phi_out) ** 2
        return intensity

    def forward(self, dn_volume: torch.Tensor,
                n_angles: int = None) -> torch.Tensor:
        """
        Simulate intensity images for all illumination angles.

        Parameters
        ----------
        dn_volume : (Nz, Ny, Nx) real tensor — RI contrast volume
        n_angles  : int or None — number of angles (defaults to config)

        Returns
        -------
        intensities : (n_angles, Ny, Nx) real tensor
        """
        if n_angles is None:
            n_angles = self.config.n_angles

        intensities = []
        for m in range(n_angles):
            I_m = self.forward_single(dn_volume, m)
            intensities.append(I_m)

        return torch.stack(intensities, dim=0)

    def simulate_measurements(self, dn_volume: torch.Tensor) -> torch.Tensor:
        """Convenience wrapper for forward()."""
        return self.forward(dn_volume)

    # ──────────────────────────────────────────────────────────────────────
    # Convenience
    # ──────────────────────────────────────────────────────────────────────

    def __repr__(self):
        c = self.config
        return (
            f"SSNPForwardModel("
            f"volume={c.volume_shape}, "
            f"NA={c.NA}, "
            f"n0={c.n0}, "
            f"λ={c.wavelength_um}μm, "
            f"angles={c.n_angles}, "
            f"device={self.device})"
        )
