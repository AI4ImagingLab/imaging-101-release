"""Local homogeneous-sphere Lorenz-Mie forward mathematics.
Adapted from David G. Grier et al., pylorenzmie; GPL-3.0. See NOTICE.
Serialization, discovery, GPU backends and GUI layers are omitted.
"""
from __future__ import annotations
import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Iterator

Coordinates = np.ndarray
Properties = dict[str, float]
Coefficients = np.ndarray
Field = np.ndarray
Image = np.ndarray
class LMObject:
    @property
    def logger(self): return logging.getLogger(__name__)
    @property
    def properties(self): return {}
    @properties.setter
    def properties(self, values):
        for key,value in values.items():
            if hasattr(self,key): setattr(self,key,value)
    @staticmethod
    def meshgrid(shape):
        y,x=np.indices(shape); return np.array([x.ravel(),y.ravel(),np.zeros(x.size)])

@dataclass
class Particle(LMObject):
    x_p: float = 0.0
    y_p: float = 0.0
    z_p: float = 100.0
    x_0: float = field(repr=False, default=0.0)
    y_0: float = field(repr=False, default=0.0)
    z_0: float = field(repr=False, default=0.0)

    def __len__(self) -> int:
        return 1

    def __iter__(self) -> Iterator['Particle']:
        return iter([self])

    def __getitem__(self, index: int) -> 'Particle':
        if index != 0:
            raise IndexError('Particle index out of range')
        return self

    @property
    def r_p(self) -> Coordinates:
        return np.asarray([self.x_p, self.y_p, self.z_p])

    @r_p.setter
    def r_p(self, r_p: Coordinates) -> None:
        self.x_p, self.y_p, self.z_p = r_p

    @property
    def r_0(self) -> Coordinates:
        return np.asarray([self.x_0, self.y_0, self.z_0])

    @r_0.setter
    def r_0(self, r_0: Coordinates) -> None:
        self.x_0, self.y_0, self.z_0 = r_0

    @LMObject.properties.getter
    def properties(self) -> Properties:
        return {'x_p': self.x_p, 'y_p': self.y_p, 'z_p': self.z_p}

    def ab(self, n_m: float | complex=1.0 + 0j, wavelength: float=0.532) -> Coefficients:
        raise NotImplementedError('Particle subclasses must define scattering coefficients')

@dataclass
class Sphere(Particle):
    a_p: float = 1.0
    n_p: float = 1.5
    k_p: float = 0.0

    @property
    def d_p(self) -> float:
        return 2.0 * self.a_p

    @d_p.setter
    def d_p(self, d_p: float) -> None:
        self.a_p = d_p / 2.0

    @Particle.properties.getter
    def properties(self) -> Properties:
        return {**super().properties, 'a_p': self.a_p, 'n_p': self.n_p, 'k_p': self.k_p}

    def ab(self, n_m: float | complex=1.42, wavelength: float=0.532) -> Coefficients:
        return Sphere.mie_coefficients(self.a_p, self.n_p, self.k_p, n_m, wavelength)

    @staticmethod
    def wiscombe_yang(x: float, m: float | complex) -> int:
        xl = np.abs(x)
        if xl <= 8.0:
            ns = np.floor(xl + 4.0 * np.cbrt(xl) + 1.0)
        elif xl <= 4200.0:
            ns = np.floor(xl + 4.05 * np.cbrt(xl) + 2.0)
        else:
            ns = np.floor(xl + 4.0 * np.cbrt(xl) + 2.0)
        xm = np.abs(x * m)
        xm_1 = np.abs(np.roll(x, -1) * m)
        nstop = np.max([ns, np.max(xm), np.max(xm_1)])
        return int(nstop)

    @staticmethod
    def neves_pisignano(x: float, precision: float=6.0) -> int:
        nstop = x + 0.76 * np.cbrt(precision * precision * x) - 4.1
        return int(nstop)

    @staticmethod
    def mie_coefficients(a_p: float, n_p: float, k_p: float, n_m: float | complex, wavelength: float) -> Coefficients:
        k = 2.0 * np.pi / wavelength
        k *= np.real(n_m)
        x = k * a_p
        m = (n_p + 1j * k_p) / n_m
        nmax = Sphere.wiscombe_yang(x, m)
        ab = np.empty((nmax + 1, 2), np.complex128)
        D1 = np.empty(nmax + 1, np.complex128)
        D3 = np.empty(nmax + 1, np.complex128)
        Ψ = np.empty(nmax + 1, np.complex128)
        ζ = np.empty(nmax + 1, np.complex128)
        D1[nmax] = 0.0
        D3[0] = 1j
        z = x * m
        for n in range(nmax, 0, -1):
            D1[n - 1] = n / z - 1.0 / (D1[n] + n / z)
        Ha = D1.copy()
        Hb = D1.copy()
        z = x
        for n in range(nmax, 0, -1):
            D1[n - 1] = n / z - 1.0 / (D1[n] + n / z)
        Ψ[0] = np.sin(z)
        ζ[0] = -1j * np.exp(1j * z)
        Ψζ = 0.5 * (1.0 - np.exp(2j * z))
        for n in range(1, nmax + 1):
            Ψ[n] = Ψ[n - 1] * (n / z - D1[n - 1])
            ζ[n] = ζ[n - 1] * (n / z - D3[n - 1])
            Ψζ *= (n / z - D1[n - 1]) * (n / z - D3[n - 1])
            D3[n] = D1[n] + 1j / Ψζ
        n = np.arange(nmax + 1)
        Ψr = np.roll(Ψ, 1)
        ζr = np.roll(ζ, 1)
        fac = Ha / m + n / x
        ab[:, 0] = (fac * Ψ - Ψr) / (fac * ζ - ζr)
        fac = Hb * m + n / x
        ab[:, 1] = (fac * Ψ - Ψr) / (fac * ζ - ζr)
        ab[0, :] = 0j
        return ab

@dataclass
class Instrument(LMObject):
    wavelength: float = 0.447
    magnification: float = 0.048
    noise: float = 0.05
    n_m: float = 1.34

    @LMObject.properties.getter
    def properties(self) -> Properties:
        return {'n_m': self.n_m, 'wavelength': self.wavelength, 'magnification': self.magnification, 'noise': self.noise}

    def wavenumber(self, in_medium: bool=True, scaled: bool=True) -> float:
        k = 2.0 * np.pi / self.wavelength
        if in_medium:
            k *= self.n_m
        if scaled:
            k *= self.magnification
        return k

class LorenzMie(LMObject):
    method: str = 'numpy'

    def __init__(self, coordinates: Coordinates | None=None, particle: Particle | list[Particle] | None=None, instrument: Instrument | None=None) -> None:
        super().__init__()
        self.coordinates = coordinates
        self.particle = particle or Sphere()
        self.instrument = instrument or Instrument()

    def __repr__(self) -> str:
        classname = self.__class__.__qualname__
        r = f'{classname}(instrument, particle)'
        inst = f'instrument={self.instrument!r}'.replace(',', ',\n\t')
        part = f'particle={self.particle!r}'.replace(',', ',\n\t')
        return '\n    '.join([r, inst, part])

    @property
    def properties(self) -> Properties:
        return {**self.particle.properties, **self.instrument.properties}

    @properties.setter
    def properties(self, properties: Properties) -> None:
        for name, value in properties.items():
            if hasattr(self.particle, name):
                setattr(self.particle, name, value)
            elif hasattr(self.instrument, name):
                setattr(self.instrument, name, value)
            elif hasattr(self, name):
                setattr(self, name, value)

    @property
    def coordinates(self) -> Coordinates:
        return self._coordinates

    @coordinates.setter
    def coordinates(self, coordinates: Coordinates | None) -> None:
        self.logger.debug('Setting coordinates')
        if coordinates is None:
            c = self.meshgrid((201, 201))
        else:
            c = np.atleast_2d(coordinates)
        ndim, npts = c.shape
        if ndim > 3:
            raise ValueError(f'Incompatible shape: coordinates.shape={coordinates.shape!r}')
        self._coordinates = np.vstack([c, np.zeros((3 - ndim, npts))])
        self._allocate()

    def _allocate(self) -> None:
        self.logger.debug('Allocating buffers')
        shape = self.coordinates.shape
        self.buffers = [np.empty(shape, dtype=complex) for _ in range(4)]
        self._field = np.empty(shape, dtype=complex)

    def hologram(self, **kwargs) -> Image:
        field = self.field(**kwargs)
        field[0, :] += 1.0
        return np.sum(field.real ** 2 + field.imag ** 2, axis=0)

    def scattered_field(self, particle: 'Particle', **kwargs) -> Field:
        k = self.instrument.wavenumber()
        r_p = particle.r_p + particle.r_0
        kdr = k * (self.coordinates - r_p[:, None])
        ab = particle.ab(self.instrument.n_m, self.instrument.wavelength)
        return self.lorenzmie(ab, kdr, **kwargs) * np.exp(-1j * k * r_p[2])

    def field(self, **kwargs) -> Field:
        self._field.fill(0.0 + 0j)
        for particle in self.particle:
            self._field += self.scattered_field(particle, **kwargs)
        return self._field

    def lorenzmie(self, ab: Coefficients, kdr: Coordinates, cartesian: bool=True, bohren: bool=True) -> Field:
        norders = ab.shape[0]
        Mo1n, Ne1n, Es, Ec = self.buffers
        kx = kdr[0, :]
        ky = kdr[1, :]
        kz = -kdr[2, :]
        shape = kx.shape
        kρ = np.hypot(kx, ky)
        kr = np.hypot(kρ, kz)
        sinkr = np.sin(kr)
        coskr = np.cos(kr)
        cosθ = kz / kr
        sinθ = kρ / kr
        off_axis = kρ > 0.0
        cosφ = np.divide(kx, kρ, out=np.ones(shape), where=off_axis)
        sinφ = np.divide(ky, kρ, out=np.zeros(shape), where=off_axis)
        factor = 1j * np.sign(kz) if bohren else -1j * np.sign(kz)
        ξ_nm2 = coskr + factor * sinkr
        ξ_nm1 = sinkr - factor * coskr
        π_nm1 = np.zeros(shape)
        π_n = np.ones(shape)
        Mo1n[0].fill(0j)
        swisc = np.empty(shape)
        twisc = np.empty(shape)
        Es.fill(0j)
        factor = 1.0
        for n in range(1, norders):
            np.multiply(π_n, cosθ, out=swisc)
            np.subtract(swisc, π_nm1, out=twisc)
            τ_n = π_nm1 - n * twisc
            ξ_n = (2.0 * n - 1.0) * (ξ_nm1 / kr) - ξ_nm2
            Dn = n * (ξ_n / kr) - ξ_nm1
            Mo1n[1] = π_n * ξ_n
            Mo1n[2] = τ_n * ξ_n
            Ne1n[0] = (n * n + n) * Mo1n[1]
            Ne1n[1] = τ_n * Dn
            Ne1n[2] = π_n * Dn
            factor *= 1j
            En = factor * (2.0 * n + 1.0) / (n * n + n)
            an = 1j * En * ab[n, 0]
            bn = En * ab[n, 1]
            Es[0] += an * Ne1n[0]
            Es[1] += an * Ne1n[1] - bn * Mo1n[1]
            Es[2] += an * Ne1n[2] - bn * Mo1n[2]
            π_nm1 = π_n
            π_n = swisc + (1.0 + 1.0 / n) * twisc
            ξ_nm2 = ξ_nm1
            ξ_nm1 = ξ_n
        Es[0] *= cosφ * sinθ / kr
        Es[1] *= cosφ
        Es[2] *= sinφ
        Es /= kr
        if cartesian:
            Ec[0] = Es[0] * sinθ * cosφ
            Ec[0] += Es[1] * cosθ * cosφ
            Ec[0] -= Es[2] * sinφ
            Ec[1] = Es[0] * sinθ * sinφ
            Ec[1] += Es[1] * cosθ * sinφ
            Ec[1] += Es[2] * cosφ
            Ec[2] = Es[0] * cosθ - Es[1] * sinθ
            return Ec
        else:
            return Es
