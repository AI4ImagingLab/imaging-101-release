# Rubidium Voigt Vapour Magnetometry

> Reconstruct magnetic field, vapour temperature and input polarization from three complete measured rubidium transmission spectra whose Zeeman-split lines overlap and broaden.
> Domain: physics | Keywords: atomic spectroscopy, magneto-optics, parameter inference | Difficulty: Hard

## Background

An atomic vapour changes the absorption and phase of light near an optical resonance. Magnetic fields shift and mix atomic energy levels, changing both transition frequencies and strengths. Thermal atomic motion broadens these lines, while temperature changes the vapour density. The resulting spectrum contains information about the local magnetic and thermodynamic environment.

A transverse magnetic field creates polarization-dependent optical modes. Their transmission depends on the incoming polarization as well as the field and temperature, making these quantities coupled in an inverse measurement. Such measurements support compact magnetometers, thermometry and optical frequency filtering.

## Problem Description

Infer the magnetic field magnitude B, vapour temperature T and actual linear input angle theta separately for three real normalized transmission spectra in a1mm natural-rubidium D2 cell. Nominal polarization labels0,45,90degrees identify acquisition conditions and are not fixed fitted angles. The magnetic field is transverse to light propagation, along laboratory x. In the weak-probe limit, atomic Hamiltonians include fine/hyperfine and Zeeman interactions. Summing thermally populated dipole-transition Voigt responses yields complex susceptibilities $\chi_+,\chi_-,\chi_0$ for the two circular and the pi transitions. Define
$$d=1+(\chi_++\chi_-)/2,\qquad g=i(\chi_+-\chi_-)/2,$$
$$n_\parallel=\sqrt{1+\chi_0},\qquad n_\perp=\sqrt{d+g^2/d},$$
$$S_0(\nu)=\cos^2\theta\,|\exp(in_\parallel k l)|^2+\sin^2\theta\,|\exp(in_\perp k l)|^2.$$
Here i is the imaginary unit, $n_\parallel,n_\perp$ are complex refractive indices for transverse eigenmodes parallel/perpendicular to the field, k is optical wavenumber in m^-1, l cell length in m, nu detuning in MHz, and S0 normalized transmitted intensity. Theta is measured from laboratory x toward y in radians. Field-induced mixing, Doppler broadening and polarization weighting make the inverse coupled and numerically nonconvex.

Save output/reconstruction.npz with every required finite real array:

| Key | Shape | Dtype | Description |
| --- | --- | --- | --- |
| reconstruction_0 | (1, 86491) | float64 | Modeled normalized S0 at all0degree-condition detunings, dimensionless |
| reconstruction_45 | (1, 83165) | float64 | Modeled normalized S0 at all45degree-condition detunings, dimensionless |
| reconstruction_90 | (1, 85604) | float64 | Modeled normalized S0 at all90degree-condition detunings, dimensionless |
| magnetic_field | (1, 3) | float64 | Positive B in tesla; condition axis0,45,90 |
| temperature | (1, 3) | float64 | Positive vapour T in kelvin; same axis |
| polarization | (1, 3) | float64 | Theta in radians in[0,pi/2]; same axis |

Each full spectrum and each physical parameter is evaluated, without spectral shifts, gain fitting, angle alignment or downsampling. Physical parameters and complete response must jointly agree with a member of a finite independently reproduced method-support set within relative-field, absolute-temperature/angle and normalized spectral-RMS limits; observation RMS is checked separately. This accommodates empirically demonstrated stochastic optimization stability, not a physical confidence interval. Independent true B/T/theta measurements are unavailable; agreement with the selected method does not eliminate observed residual structure.

## Data Description

| Key | Shape | Dtype | Description |
| --- | --- | --- | --- |
| detuning_0 | (1, 86491) | float64 | Optical detuning for nominal0degree condition, MHz, increasing source order |
| transmission_0 | (1, 86491) | float64 | Measured normalized transmitted intensity S0, dimensionless |
| detuning_45 | (1, 83165) | float64 | Detuning for nominal45degree condition, MHz |
| transmission_45 | (1, 83165) | float64 | Measured normalized S0, dimensionless |
| detuning_90 | (1, 85604) | float64 | Detuning for nominal90degree condition, MHz |
| transmission_90 | (1, 85604) | float64 | Measured normalized S0, dimensionless |

All native samples are retained. Frequencies were supplied in GHz and converted to MHz by multiplying by1000; no spectral interpolation or binning of observations is imposed.

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
| --- | --- | --- | --- |
| Species;transition | Natural Rb;D2 | none | Weak-probe atomic vapour |
| Cell length | 0.001 | m | Uniform optical path |
| Field geometry | polar90;azimuth0 | degree;degree | B along x;light propagates along z |
| Nominal input angles | 0,45,90 | degree | Labels only;actual angles unknown |
| Rb85;Rb87 fractions | 0.7217;0.2783 | none | Natural abundance convention |
| Optical wavelength | 7.802413272e-7 | m | k=2pi/wavelength |
| Natural linewidth | 6.065 | MHz | Natural angular decay rate=2pi*linewidth*1e6 |
| Extra buffer broadening;line shift | 0;0 | MHz;MHz | No additional fitted broadening or frequency shift |
| Electronic spin;orbital g factor | 0.5;1 | none | Atomic Hamiltonian convention |
| Electron spin g factor | 2.00231930436256 | none | Source numerical constant |
| Rb85 I;As;Ap;Bp;isotope shift | 2.5;1011.910813;25.038;26.011;21.734 | none;MHz;MHz;MHz;MHz | Nuclear spin,ground/excited dipole,excited quadrupole,frequency offset |
| Rb87 I;As;Ap;Bp;isotope shift | 1.5;3417.341305452145;84.7185;12.4965;-56.361 | none;MHz;MHz;MHz;MHz | Same conventions |
| Nuclear g factors85;87 | -0.00029364;-0.0009951414 | none | Nuclear Zeeman terms |
| Atomic masses85;87 | 84.911789732;86.90918052 | atomic mass unit | Thermal Doppler velocities |
| Fine-structure coefficient | 7123000 | MHz | Full uncoupled excited-state Hamiltonian |
| Bohr magneton | 9.2740100783e-24 | J T^-1 | Field-energy conversion |
| Planck constant h | 6.62607015e-34 | J s | Energy/frequency conversion |
| Reduced Planck constant | 1.0545718176461565e-34 | J s | h/(2pi) source precision |
| Boltzmann constant | 1.380649e-23 | J K^-1 | Thermal populations/density |
| Atomic mass unit | 1.6605390666e-27 | kg | Source mass conversion |
| Vacuum permittivity | 8.8541878128e-12 | F m^-1 | Dipole susceptibility scale |
| Light speed | 299792458 | m s^-1 | Frequency-dependent optical wavenumber |
| Temperature conversion | Celsius+273.15;population branch+273.16 | K | Retained source conventions |
| Transition discretization | Integer-MHz centers,toward zero;source strength cutoff | MHz | Selected published implementation convention |
| Observation errors | Not separately supplied | none | Unweighted full-spectrum residual model |
| Spatial model | Uniform cell and transverse field | none | No field gradient or arbitrary-angle geometry |

### Forward calibration laws

The following vapour-pressure and optical conventions are part of the measurement model. For thermodynamic temperature $\mathcal T$ in K, let dimensionless pressure ratio P be
$$P(\mathcal T)=\begin{cases}
10^{4.857-4215/\mathcal T},&\mathcal T<312.46,\\
10^{8.316-4275/\mathcal T-1.3102\log_{10}\mathcal T},&\mathcal T\ge312.46.
\end{cases}
\qquad N(\mathcal T)=\frac{101325P(\mathcal T)}{k_B\mathcal T}.$$
Here N is number density in m^-3 and $k_B$ is the stated Boltzmann constant. The nominal zero extra buffer width does not remove density self-broadening. Here $\epsilon_0$ is vacuum permittivity and $\hbar$ the reduced Planck constant. With wavelength $\lambda$, natural frequency linewidth $\gamma_0=6.065$ MHz, $\Gamma_0=2\pi\gamma_0\,10^6$ s^-1, and reduced dipole scale $d_0$ in C m, the selected model uses
$$d_0=3\sqrt{\frac{\epsilon_0\hbar(2\gamma_0\,10^6)\lambda^3}{8\pi}},
\qquad \Gamma=\Gamma_0+2\pi\Gamma_0N\sqrt{2}\left(\frac{\lambda}{2\pi}\right)^3.$$
The additional factor2pi in the self-broadening term is the preserved source convention.

For isotope a with mass M, thermal speed $u=\sqrt{2k_B\mathcal T/M}$, angular detuning D in s^-1 and $k_D=2\pi/\lambda+D/c$, define the normalized complex profile
$$V_a(D)=\frac{i\sqrt{\pi}}{2k_Du}\,
w\left(\frac{D}{k_Du}+\frac{i\Gamma}{2k_Du}\right),
\qquad w(z)=e^{-z^2}\operatorname{erfc}(-iz).$$
Here c is light speed and erfc is the complementary error function. Real V gives dispersion and imaginary V absorption. The three susceptibilities are
$$\chi_q(\nu)=\frac{2Nd_0^2}{\hbar\epsilon_0}
\sum_a F_a\sum_j W_{a,q,j}\,
V_a\!\left(2\pi\,10^6[\nu-\nu_{a,q,j}]\right).$$
Index q denotes Left/Right/Z (the two circular and the pi components), $F_a$ is the stated isotope fraction, and j enumerates allowed transitions. A transition weight is the squared dipole eigenvector overlap divided by3 and multiplied by normalized ground-state Boltzmann population. For ground energy Eg in MHz, population is proportional to $\exp[-(E_g-\min E_g)h\,10^6/(k_B\mathcal T_p)]$, normalized over the complete isotope ground manifold. The retained source population temperature is $\mathcal T_p=\mathcal T+0.01$ K. Keep overlaps only when their square exceeds0.0005.

Transition centers are the excited-minus-ground energies in MHz, truncated toward zero to integer MHz. The profile is evaluated on an auxiliary1MHz grid with half-width equal to the integer part of1.1times(maximum absolute observed detuning plus maximum absolute transition center), then linearly interpolated for each shifted observed detuning. This fixes the selected numerical forward convention without changing or discarding any native observation.

## Method Hints

Use a full uncoupled-basis atomic Hamiltonian and thermally broadened transition response, followed by transverse dielectric-mode propagation. Fit the three physical quantities jointly with a global differential-evolution family. Normalized intensity is invariant to common input-field gain, so a unit-length real polarization vector removes that redundant degree of freedom. Preserve fine/hyperfine mixing and assess residuals across the complete measured band.

## References

- Keaveney,Adams,Hughes(2018), ElecSus extension, [doi:10.1016/j.cpc.2017.12.001](https://doi.org/10.1016/j.cpc.2017.12.001); [arXiv:1708.05305v1](https://arxiv.org/abs/1708.05305v1), Figure2 and AppendixB.
- Zentile etal.(2015), atomic susceptibility and Hamiltonian method, [doi:10.1016/j.cpc.2014.11.023](https://doi.org/10.1016/j.cpc.2014.11.023).
