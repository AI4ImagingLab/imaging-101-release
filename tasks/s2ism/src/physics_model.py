"""
Forward model for s2ISM.

Handles PSF estimation, simulation, and the forward convolution model that maps
a 3-D fluorescent object to multi-channel ISM detector measurements.
"""

import numpy as np
from scipy.signal import convolve, argrelmin, argrelmax
from scipy.special import kl_div
from scipy.stats import pearsonr

import brighteyes_ism.simulation.PSF_sim as psf_sim
import brighteyes_ism.analysis.Tools_lib as tools

from .preprocessing import find_parameters, find_mag, find_misalignment


# ==========================================
# PSF Estimator
# ==========================================

class GridFinder(psf_sim.GridParameters):
    def __init__(self, grid_par: psf_sim.GridParameters = None):
        psf_sim.GridParameters.__init__(self)
        if grid_par is not None:
            vars(self).update(vars(grid_par))
        self.shift = None

    def estimate(self, dset, wl_ex, wl_em, na):
        from brighteyes_ism.analysis.APR_lib import ShiftVectors
        ref = dset.shape[-1] // 2
        usf = 50
        shift, _ = ShiftVectors(dset[5:-5, 5:-5, :], usf, ref, filter_sigma=1)
        par = find_parameters(shift, self.geometry, self.name)
        self.shift = par[0] * self.pxsizex
        self.rotation = par[1]
        self.mirroring = par[2]
        self.M = np.round(
            find_mag(self.shift, wl_ex=wl_ex, wl_em=wl_em,
                     pxpitch=self.pxpitch, pxdim=self.pxdim, na=na))


def psf_width(pxsizex: float, pxsizez: float, Nz: int,
              simPar: psf_sim.simSettings, spad_size, stack='positive') -> int:
    if stack == 'positive' or stack == 'negative':
        z = pxsizez * Nz
    else:
        z = pxsizez * (Nz // 2)

    M2 = 3
    w0 = simPar.airy_unit / 2
    z_r = (np.pi * w0**2 * simPar.n) / simPar.wl
    w_z = w0 * np.sqrt(1 + (M2 * z / z_r)**2)
    Nx = int(np.round((2 * (w_z + spad_size) / pxsizex)))
    if Nx % 2 == 0:
        Nx += 1
    return Nx


def find_max_discrepancy(correlation: np.ndarray, gridpar: psf_sim.GridParameters,
                         mode: str, graph: bool):
    if mode == 'KL':
        idx = np.asarray(argrelmax(correlation)).ravel()[0]
    elif mode == 'Pearson':
        idx = np.asarray(argrelmin(correlation)).ravel()[0]
    else:
        raise Exception("Discrepancy method unknown.")

    optimal_depth = idx * gridpar.pxsizez

    return optimal_depth


def conditioning(gridPar: psf_sim.GridParameters, exPar: psf_sim.simSettings = None,
                 emPar: psf_sim.simSettings = None, stedPar: psf_sim.simSettings = None,
                 mode='Pearson', stack='positive', input_psf=None):
    if input_psf is None:
        if exPar is None or emPar is None:
            raise Exception("PSF is not an input. PSF parameters are needed.")

        gridPar.Nx = psf_width(gridPar.pxsizex, gridPar.pxsizez, gridPar.Nz, exPar,
                               gridPar.spad_size())
        PSF, detPSF, exPSF = psf_sim.SPAD_PSF_3D(gridPar, exPar, emPar,
                                                    stedPar=stedPar, spad=None, stack=stack)

        npx = int(np.round(((gridPar.N // 2) * gridPar.pxpitch + gridPar.pxdim / 2)
                            / gridPar.M / gridPar.pxsizex))

        PSF = tools.CropEdge(PSF, npx, edges='l,r,u,d', order='zxyc')
        detPSF = tools.CropEdge(detPSF, npx, edges='l,r,u,d', order='zxyc')
        exPSF = tools.CropEdge(exPSF, npx, edges='l,r,u,d', order='zxy')
    else:
        PSF, detPSF, exPSF = input_psf

    for i in range(gridPar.Nz):
        PSF[i] /= np.sum(PSF[i])

    corr = np.empty(gridPar.Nz)
    if mode == 'KL':
        for i in range(gridPar.Nz):
            corr[i] = kl_div(PSF[0, ...].flatten(), PSF[i, ...].flatten()).sum()
    elif mode == 'Pearson':
        for i in range(gridPar.Nz):
            corr[i] = pearsonr(PSF[0, ...].flatten(), PSF[i, ...].flatten())[0]

    return corr, [PSF, detPSF, exPSF]


def find_out_of_focus_from_param(pxsizex: float = None, exPar: psf_sim.simSettings = None,
                                 emPar: psf_sim.simSettings = None,
                                 grid: psf_sim.GridParameters = None,
                                 stedPar: psf_sim.simSettings = None,
                                 mode: str = 'Pearson', stack: str = 'symmetrical',
                                 graph: bool = False):
    if exPar is None:
        raise Exception("PSF parameters are needed.")
    if emPar is None:
        raise Exception("PSF parameters are needed.")
    if pxsizex is None and grid is None:
        raise Exception("Pixel size is needed as input.")

    if grid is None:
        range_z = 1.5 * exPar.depth_of_field
        nz = 60
        gridPar = psf_sim.GridParameters()
        gridPar.Nz = nz
        gridPar.pxsizez = np.round(range_z / nz)
        gridPar.pxsizex = pxsizex
    else:
        gridPar = grid.copy()

    Nx_temp = psf_width(pxsizex, gridPar.pxsizez, gridPar.Nz, exPar, gridPar.spad_size())
    gridPar.Nx = Nx_temp

    correlation, PSF = conditioning(gridPar=gridPar, emPar=emPar,
                                    exPar=exPar, stedPar=stedPar, mode=mode,
                                    stack=stack)

    optimal_depth = find_max_discrepancy(correlation=correlation, gridpar=gridPar,
                                         mode=mode, graph=graph)

    return optimal_depth, PSF


def find_upsampling(pxsize_exp: int, pxsize_sim: int = 4):
    ups = np.arange(1, np.floor(pxsize_exp)).astype(int)
    l = int(len(ups))
    res = np.empty(l)
    for i in range(l):
        res[i] = (pxsize_exp / ups[i] - pxsize_sim) ** 2
    index = np.argmin(res)
    ups_opt = ups[index]
    return ups_opt


def psf_estimator_from_data(data: np.ndarray, exPar: psf_sim.simSettings,
                            emPar: psf_sim.simSettings, grid: psf_sim.GridParameters,
                            downsample: bool = True, stedPar: psf_sim.simSettings = None,
                            z_out_of_focus: str = 'ToFind', n_photon_excitation: int = 1,
                            stack='symmetrical', check_alignment: bool = False):
    N = int(np.sqrt(data.shape[-1]))

    if isinstance(z_out_of_focus, str) and z_out_of_focus == 'ToFind':
        pxsizez, _ = find_out_of_focus_from_param(grid.pxsizex, exPar, emPar,
                                                    mode='KL', stack='positive')
    else:
        pxsizez = float(z_out_of_focus)

    grid_simul = GridFinder(grid)
    grid_simul.estimate(data, exPar.wl, emPar.wl, emPar.na)
    grid_simul.Print()

    if downsample is True:
        ups = find_upsampling(grid_simul.pxsizex, pxsize_sim=int(emPar.airy_unit / 100))
    else:
        ups = 1

    Nx_simul = psf_width(grid_simul.pxsizex, grid_simul.pxsizez, grid_simul.Nz, exPar,
                          grid_simul.spad_size(), stack=stack)

    pxsize_simul = grid_simul.pxsizex / ups
    Nx_up = Nx_simul * ups

    grid_simul.Nx = Nx_up
    grid_simul.pxsizex = pxsize_simul
    grid_simul.pxsizez = pxsizez
    grid_simul.N = N

    if check_alignment is True:
        tip, tilt = find_misalignment(data, grid.pxpitch, grid.M, exPar.na, exPar.wl)
        exPar.abe_index = [1, 2]
        exPar.abe_ampli = [tip, tilt]

    Psf, detPsf, exPsf = psf_sim.SPAD_PSF_3D(grid_simul, exPar, emPar,
                                                n_photon_excitation=n_photon_excitation,
                                                stedPar=stedPar, spad=None, stack=stack)

    if check_alignment is True:
        from .preprocessing import realign_psf
        Psf = realign_psf(Psf)

    if downsample:
        Psf_ds = tools.DownSample(Psf, ups, order='zxyc')
        detPsf_ds = tools.DownSample(detPsf, ups, order='zxyc')
        exPsf_ds = tools.DownSample(exPsf, ups, order='zxy')
    else:
        Psf_ds = Psf
        detPsf_ds = detPsf
        exPsf_ds = exPsf

    return Psf_ds, detPsf_ds, exPsf_ds


def forward_model(ground_truth, psf):
    """Apply the ISM forward model: convolve ground truth with PSF per channel per z-plane."""
    Nz, Ny, Nx, Nch = psf.shape
    blurred = np.empty([Nz, ground_truth.shape[1], ground_truth.shape[2], Nch])
    for i in range(Nch):
        for j in range(Nz):
            blurred[j, :, :, i] = convolve(ground_truth[j], psf[j, :, :, i], mode='same')
    return blurred
