"""Explicit RbD2 fine/hyperfine/Zeeman and weak-probe mathematics.
Copyright2014-2018 M.A.Zentile,J.Keaveney,L.Weller,D.Whiting,C.S.Adams,I.G.Hughes.
Adapted from selected ElecSus1933456b functions under Apache2; see LICENSE/NOTICE.
No upstream/domain-library imports; only the bound RbD2 Voigt physical branch.
"""
from numpy import zeros,sqrt,pi,dot,exp,sin,cos,array,amax,arange,concatenate,argmin,matmul,identity,append,transpose,kron
import numpy as np
from numpy import log10
from scipy.linalg import eigh
from scipy.special import wofz
from scipy.interpolate import interp1d


S=0.5

gs=2.00231930436256

muB=9.2740100783e-24

kB=1.380649e-23

amu=1.6605390666e-27

e0=8.8541878128e-12

a0=5.29177210903e-11

hbar=1.0545718176461565e-34

c=299792458.0

e=1.602176634e-19

h=6.62607015e-34

class Rb85:
    """Constants relating to the rubidium-85 atom"""
    I = 2.5
    As = 1011.910813
    gI = -0.00029364
    mass = 84.911789732 * amu
    FS = 7123000.0

class Rb87:
    """Constants relating to the rubidium-87 atom"""
    I = 1.5
    As = 3417.341305452145
    gI = -0.0009951414
    mass = 86.90918052 * amu
    FS = 7123000.0

class RbD2Transition:
    """Constants relating to the rubidium D2 transition"""
    wavelength = 7.802413272e-07
    wavevectorMagnitude = 2.0 * pi / wavelength
    NatGamma = 6.065
    dipoleStrength = 3.0 * sqrt(e0 * hbar * (2.0 * NatGamma * 10.0 ** 6) * wavelength ** 3 / (8.0 * pi))
    v0 = 384230426600000.0

class Rb85_D2:
    """Constants relating to rubidium-85 and the D2 transition"""
    Ap = 25.038
    Bp = 26.011
    IsotopeShift = 21.734

class Rb87_D2:
    """Constants relating to rubidium-87 and the D2 transition"""
    Ap = 84.7185
    Bp = 12.4965
    IsotopeShift = -56.361

def raising_operator(jj):
    b = 0
    dim = int(2 * jj + 1)
    jp = zeros((dim, dim))
    z = arange(dim)
    m = jj - z
    while b < dim - 1:
        mm = m[b + 1]
        jp[b, b + 1] = sqrt(jj * (jj + 1) - mm * (mm + 1))
        b = b + 1
    return jp

def jx(jj):
    jp = raising_operator(jj)
    jm = transpose(jp)
    jx = 0.5 * (jp + jm)
    return jx

def jy(jj):
    jp = raising_operator(jj)
    jm = transpose(jp)
    jy = 0.5j * (jm - jp)
    return jy

def jz(jj):
    jp = raising_operator(jj)
    jm = transpose(jp)
    jz = 0.5 * (dot(jp, jm) - dot(jm, jp))
    return jz

def sz(L, S, I):
    Sz = jz(S)
    gL = int(2 * L + 1)
    Li = identity(gL)
    gI = int(2 * I + 1)
    Ii = identity(gI)
    sz = kron(kron(Li, Sz), Ii)
    return sz

def lz(L, S, I):
    gS = int(2 * S + 1)
    Si = identity(gS)
    Lz = jz(L)
    gI = int(2 * I + 1)
    Ii = identity(gI)
    lz = kron(kron(Lz, Si), Ii)
    return lz

def Iz(L, S, I):
    gS = int(2 * S + 1)
    gL = int(2 * L + 1)
    Si = identity(gS)
    Li = identity(gL)
    Iz_num = jz(I)
    Iz = kron(kron(Li, Si), Iz_num)
    return Iz

def Hfs(L, S, I):
    """Provides the L dot S matrix (fine structure)"""
    gS = int(2 * S + 1)
    Sx = jx(S)
    Sy = jy(S)
    Sz = jz(S)
    Si = identity(gS)
    gL = int(2 * L + 1)
    Lx = jx(L)
    Ly = jy(L)
    Lz = jz(L)
    Li = identity(gL)
    gJ = gL * gS
    Jx = kron(Lx, Si) + kron(Li, Sx)
    Jy = kron(Ly, Si) + kron(Li, Sy)
    Jz = kron(Lz, Si) + kron(Li, Sz)
    J2 = dot(Jx, Jx) + dot(Jy, Jy) + dot(Jz, Jz)
    gI = int(2 * I + 1)
    Ii = identity(gI)
    gF = gJ * gI
    Fi = identity(gF)
    Hfs = 0.5 * (kron(J2, Ii) - L * (L + 1) * Fi - S * (S + 1) * Fi)
    return Hfs

def Hhfs(L, S, I):
    """Provides the I dot J matrix (hyperfine structure interaction)"""
    gS = int(2 * S + 1)
    Sx = jx(S)
    Sy = jy(S)
    Sz = jz(S)
    Si = identity(gS)
    gL = int(2 * L + 1)
    Lx = jx(L)
    Ly = jy(L)
    Lz = jz(L)
    Li = identity(gL)
    gJ = gL * gS
    Jx = kron(Lx, Si) + kron(Li, Sx)
    Jy = kron(Ly, Si) + kron(Li, Sy)
    Jz = kron(Lz, Si) + kron(Li, Sz)
    Ji = identity(gJ)
    J2 = dot(Jx, Jx) + dot(Jy, Jy) + dot(Jz, Jz)
    gI = int(2 * I + 1)
    gF = gJ * gI
    Ix = jx(I)
    Iy = jy(I)
    Iz = jz(I)
    Ii = identity(gI)
    Fx = kron(Jx, Ii) + kron(Ji, Ix)
    Fy = kron(Jy, Ii) + kron(Ji, Iy)
    Fz = kron(Jz, Ii) + kron(Ji, Iz)
    Fi = identity(gF)
    F2 = dot(Fx, Fx) + dot(Fy, Fy) + dot(Fz, Fz)
    Hhfs = 0.5 * (F2 - I * (I + 1) * Fi - kron(J2, Ii))
    return Hhfs

def Bbhfs(L, S, I):
    """Calculates electric quadrupole matrix.

    Calculates the part in square brakets from
    equation (8) in manual
    """
    gS = int(2 * S + 1)
    Sx = jx(S)
    Sy = jy(S)
    Sz = jz(S)
    Si = identity(gS)
    gL = int(2 * L + 1)
    Lx = jx(L)
    Ly = jy(L)
    Lz = jz(L)
    Li = identity(gL)
    gJ = gL * gS
    Jx = kron(Lx, Si) + kron(Li, Sx)
    Jy = kron(Ly, Si) + kron(Li, Sy)
    Jz = kron(Lz, Si) + kron(Li, Sz)
    gI = int(2 * I + 1)
    gF = gJ * gI
    Ix = jx(I)
    Iy = jy(I)
    Iz = jz(I)
    Fi = identity(gF)
    IdotJ = kron(Jx, Ix) + kron(Jy, Iy) + kron(Jz, Iz)
    IdotJ2 = dot(IdotJ, IdotJ)
    if I != 0:
        Bbhfs = 1.0 / (6 * I * (2 * I - 1)) * (3 * IdotJ2 + 3.0 / 2 * IdotJ - I * (I + 1) * 15.0 / 4 * Fi)
    else:
        Bbhfs = 0
    return Bbhfs

def numDenRb(T):
    """Calculates the rubidium number density"""
    if T < 312.46:
        p = 10.0 ** (4.857 - 4215.0 / T)
    else:
        p = 10.0 ** (8.316 - 4275.0 / T - 1.3102 * log10(T))
    NumberDensity = 101325.0 * p / (kB * T)
    return NumberDensity

def FreqStren(groundLevels, excitedLevels, groundDim, excitedDim, Dline, hand, BoltzmannFactor=True, T=293.16):
    """ 
	Calculate transition frequencies and strengths by taking dot
	products of relevant parts of the ground / excited state eigenvectors 
	"""
    transitionFrequency = zeros(groundDim * 2 * groundDim)
    transitionStrength = zeros(groundDim * 2 * groundDim)
    transNo = 0
    if BoltzmannFactor:
        groundEnergies = array(groundLevels)[:, 0].real
        lowestEnergy = groundLevels[0][argmin(groundLevels, axis=0)[0]].real
        BoltzDist = exp(-(groundEnergies - lowestEnergy) * h * 1000000.0 / (kB * T))
        BoltzDist = BoltzDist / BoltzDist.sum()
    if hand == 'Right':
        bottom = 1
        top = groundDim + 1
    elif hand == 'Z':
        bottom = groundDim + 1
        top = 2 * groundDim + 1
    elif hand == 'Left':
        bottom = 2 * groundDim + 1
        top = excitedDim + 1
    if Dline == 'D1':
        interatorList = list(range(groundDim))
    elif Dline == 'D2':
        interatorList = list(range(groundDim, excitedDim))
    for gg in range(groundDim):
        for ee in interatorList:
            cleb = dot(groundLevels[gg][1:], excitedLevels[ee][bottom:top]).real
            cleb2 = cleb * cleb
            if cleb2 > 0.0005:
                transitionFrequency[transNo] = int(-groundLevels[gg][0].real + excitedLevels[ee][0].real)
                if BoltzmannFactor:
                    transitionStrength[transNo] = 1.0 / 3 * cleb2 * BoltzDist[gg]
                else:
                    transitionStrength[transNo] = 1.0 / 3 * cleb2 * 1.0 / groundDim
                transNo += 1
    return (transitionFrequency, transitionStrength, transNo)

def add_voigt(d, DoppTemp, atomMass, wavenumber, gamma, voigtwidth, ltransno, lenergy, lstrength, rtransno, renergy, rstrength, ztransno, zenergy, zstrength):
    xpts = len(d)
    npts = 2 * voigtwidth + 1
    detune = 2.0 * pi * 1000000.0 * (arange(npts) - voigtwidth)
    wavenumber = wavenumber + detune / c
    u = sqrt(2.0 * kB * DoppTemp / atomMass)
    ku = wavenumber * u
    a = gamma / ku
    b = detune / ku
    y = 1j * (0.5 * sqrt(pi) / ku) * wofz(b + 0.5j * a)
    ab = y.imag
    disp = y.real
    f_ab = interp1d(detune, ab)
    f_disp = interp1d(detune, disp)
    lab = zeros(xpts)
    ldisp = zeros(xpts)
    for line in range(ltransno + 1):
        xc = lenergy[line]
        lab += lstrength[line] * f_ab(2.0 * pi * (d - xc) * 1000000.0)
        ldisp += lstrength[line] * f_disp(2.0 * pi * (d - xc) * 1000000.0)
    rab = zeros(xpts)
    rdisp = zeros(xpts)
    for line in range(rtransno + 1):
        xc = renergy[line]
        rab += rstrength[line] * f_ab(2.0 * pi * (d - xc) * 1000000.0)
        rdisp += rstrength[line] * f_disp(2.0 * pi * (d - xc) * 1000000.0)
    zab = zeros(xpts)
    zdisp = zeros(xpts)
    for line in range(ztransno + 1):
        xc = zenergy[line]
        zab += zstrength[line] * f_ab(2.0 * pi * (d - xc) * 1000000.0)
        zdisp += zstrength[line] * f_disp(2.0 * pi * (d - xc) * 1000000.0)
    return (lab, ldisp, rab, rdisp, zab, zdisp)

class Hamiltonian(object):

    def __init__(self, Isotope, Trans='D2', gL=1.0, Bfield=4200.0):
        if Trans != 'D2' or Isotope not in ['Rb85', 'Rb87']:
            raise ValueError('Bound RbD2 model only')
        atom = Rb85 if Isotope == 'Rb85' else Rb87
        atom_transition = Rb85_D2 if Isotope == 'Rb85' else Rb87_D2
        self.atom = atom
        self.ds = int((2 * S + 1) * (2 * atom.I + 1))
        self.dp = 3 * self.ds
        self.groundManifold, self.groundEnergies = self.groundStateManifold(atom.gI, atom.I, atom.As, atom_transition.IsotopeShift, Bfield)
        self.excitedManifold, self.excitedEnergies = self.excitedStateManifold(gL, atom.gI, atom.I, atom_transition.Ap, atom_transition.Bp, Bfield)

    def groundStateManifold(self, gI, I, A_hyp_coeff, IsotopeShift, Bfield):
        """Function to produce the ground state manifold"""
        ds = int((2 * S + 1) * (2 * I + 1))
        As = A_hyp_coeff
        S_StateHamiltonian = As * Hhfs(0.0, S, I) + IsotopeShift * identity(ds)
        Ez = muB * Bfield * 0.0001 / (hbar * 2.0 * pi * 1000000.0)
        S_StateHamiltonian += Ez * (gs * sz(0.0, S, I) + gI * Iz(0.0, S, I))
        EigenSystem = eigh(S_StateHamiltonian)
        EigenValues = EigenSystem[0].real
        EigenVectors = EigenSystem[1]
        stateManifold = append([EigenValues], EigenVectors, axis=0)
        sortedManifold = sorted(transpose(stateManifold), key=lambda i: i[0])
        return (sortedManifold, EigenValues)

    def excitedStateManifold(self, gL, gI, I, A_hyp_coeff, B_hyp_coeff, Bfield):
        """Function to produce the excited state manifold"""
        dp = int(3 * (2 * S + 1) * (2 * I + 1))
        FS = self.atom.FS
        Ap = A_hyp_coeff
        Bp = B_hyp_coeff
        if Bp == 0.0:
            P_StateHamiltonian = FS * Hfs(1.0, S, I) + FS * identity(dp) + Ap * Hhfs(1.0, S, I)
        if Bp != 0.0:
            P_StateHamiltonian = FS * Hfs(1.0, S, I) - FS / 2.0 * identity(dp) + Ap * Hhfs(1.0, S, I)
            P_StateHamiltonian += Bp * Bbhfs(1.0, S, I)
        E = muB * (Bfield * 0.0001) / (hbar * 2.0 * pi * 1000000.0)
        P_StateHamiltonian += E * (gL * lz(1.0, S, I) + gs * sz(1.0, S, I) + gI * Iz(1.0, S, I))
        ep = eigh(P_StateHamiltonian)
        EigenValues = ep[0].real
        EigenVectors = ep[1]
        stateManifold = append([EigenValues], EigenVectors, axis=0)
        sortedManifold = sorted(transpose(stateManifold), key=lambda i: i[0])
        return (sortedManifold, EigenValues)


def susceptibility(detuning,temperature,magnetic_field):
    """MHz detuning, Celsius vapour temperature, Gauss field; return chi+,chi-,chi0."""
    transitions=[];energies=[]
    for name in ['Rb85','Rb87']:
        model=Hamiltonian(name,'D2',1.,magnetic_field)
        rows=[FreqStren(model.groundManifold,model.excitedManifold,model.ds,model.dp,'D2',hand,True,temperature+273.16) for hand in ['Left','Right','Z']]
        transitions.append(rows);energies.extend(r[0] for r in rows)
    density=numDenRb(temperature+273.15)
    const=RbD2Transition
    gamma0=2*pi*const.NatGamma*1e6
    gamma=gamma0+2*pi*gamma0*density*1.414213562373095*(const.wavelength/(2*pi))**3
    width=int(1.1*(amax(abs(detuning))+amax(abs(concatenate(energies)))))
    chi=[]
    prefactor=2*density*const.dipoleStrength**2/(hbar*e0)
    for rows,atom in zip(transitions,[Rb85,Rb87]):
        flat=[value for row in rows for value in (row[2],row[0],row[1])]
        curves=add_voigt(detuning,temperature+273.15,atom.mass,const.wavevectorMagnitude,gamma,width,*flat)
        chi.append(np.array([curves[1]+1j*curves[0],curves[3]+1j*curves[2],curves[5]+1j*curves[4]]))
    rb85fraction=.7217
    return prefactor*(rb85fraction*chi[0]+(1-rb85fraction)*chi[1])

def transmission(detuning,temperature,magnetic_field,ex,ey,length=.001):
    """Exact selected Voigt dielectric-mode propagation for linear input."""
    left,right,z=susceptibility(detuning,temperature,magnetic_field)
    dielectric_x=.5*(2+left+right)
    dielectric_xy=.5j*(right-left)
    n1=np.sqrt(dielectric_x+dielectric_xy**2/dielectric_x)
    n2=np.sqrt(1+z)
    wavevector=RbD2Transition.wavevectorMagnitude
    parallel=np.abs(np.exp(1j*n2*wavevector*length))**2
    perpendicular=np.abs(np.exp(1j*n1*wavevector*length))**2
    return (ex*ex*parallel+ey*ey*perpendicular)/(ex*ex+ey*ey)
