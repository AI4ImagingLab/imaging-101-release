"""Thin-screen interference geometry, Fourier units and corrected peak covariance."""
import numpy as np

def arc_beta(doppler_hz,kappa):
    """β in cycles/m for Doppler Hz and κ=D_eff/V_parallel² in s²/m."""
    return .5*kappa*np.asarray(doppler_hz)**2

def physical_kappa(author_curvature):
    """Source β curvature uses mHz on horizontal axis, not angular frequency."""
    return 2e6*np.asarray(author_curvature)

def fit_parabola(x,y):
    """Fit source peak, propagate full coefficient covariance independently.

    The source etaerr2 auxiliary formula omits coefficient powers/covariance.
    This correction changes no fitted peak or source empirical noise error.
    """
    scale=1000/np.ptp(x);scaled=x*scale
    coefficients,covariance=np.polyfit(scaled,y,2,cov=True)
    a,b,c=coefficients
    fitted=a*scaled**2+b*scaled+c
    peak=-b/(2*a)/scale
    jacobian=np.array([b/(2*a*a),-1/(2*a),0.])/scale
    uncertainty=np.sqrt(jacobian@covariance@jacobian)
    return fitted,peak,uncertainty

# Extracted selected mathematical stages, Daniel Reardon scintools, MIT license.

import scipy.constants as sc
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from copy import deepcopy as cp

def is_valid(array):
    return np.isfinite(array) * ~np.isnan(array)

def svd_reconstruct(arr):
    u, s, w = np.linalg.svd(arr)
    s[1:] = 0.0
    S = np.zeros([len(u), len(w)], np.complex128)
    S[:len(s), :len(s)] = np.diag(s)
    return np.dot(np.dot(u, S), w)

def svd_model(arr):
    model = svd_reconstruct(arr)
    arr = arr / np.abs(model)
    return (arr, model)

def get_window(nt, nf, window='hanning', frac=0.1):
    if window.lower() == 'hanning':
        cw = np.hanning(np.floor(frac * nt))
        sw = np.hanning(np.floor(frac * nf))
    elif window.lower() == 'hamming':
        cw = np.hamming(np.floor(frac * nt))
        sw = np.hamming(np.floor(frac * nf))
    elif window.lower() == 'blackman':
        cw = np.blackman(np.floor(frac * nt))
        sw = np.blackman(np.floor(frac * nf))
    elif window.lower() == 'bartlett':
        cw = np.bartlett(np.floor(frac * nt))
        sw = np.bartlett(np.floor(frac * nf))
    else:
        print('Window unknown.. Please add it!')
    chan_window = np.insert(cw, int(np.ceil(len(cw) / 2)), np.ones([nt - len(cw)]))
    subint_window = np.insert(sw, int(np.ceil(len(sw) / 2)), np.ones([nf - len(sw)]))
    return (chan_window, subint_window)

class Spectrum:

    def trim_edges(self, bandwagon_frac=0.5):
        self.dyn[np.isnan(self.dyn)] = 0
        nc = len(self.dyn[0, :])
        nr = len(self.dyn[:, 0])
        if len(np.argwhere(self.dyn[0, :] == 0)) > bandwagon_frac * nc:
            self.dyn[0, :] = np.zeros(np.shape(self.dyn[0, :]))
        rowsum = sum(abs(self.dyn[0, :]))
        while rowsum == 0:
            self.dyn = np.delete(self.dyn, 0, axis=0)
            self.freqs = np.delete(self.freqs, 0)
            if len(np.argwhere(self.dyn[0, :] == 0)) > bandwagon_frac * nc:
                self.dyn[0, :] = np.zeros(np.shape(self.dyn[0, :]))
            rowsum = sum(abs(self.dyn[0, :]))
        if len(np.argwhere(self.dyn[-1, :] == 0)) > bandwagon_frac * nc:
            self.dyn[-1, :] = np.zeros(np.shape(self.dyn[-1, :]))
        rowsum = sum(abs(self.dyn[-1, :]))
        while rowsum == 0:
            self.dyn = np.delete(self.dyn, -1, axis=0)
            self.freqs = np.delete(self.freqs, -1)
            if len(np.argwhere(self.dyn[-1, :] == 0)) > bandwagon_frac * nc:
                self.dyn[-1, :] = np.zeros(np.shape(self.dyn[-1, :]))
            rowsum = sum(abs(self.dyn[-1, :]))
        if len(np.argwhere(self.dyn[:, 0] == 0)) > bandwagon_frac * nr:
            self.dyn[:, 0] = np.zeros(np.shape(self.dyn[:, 0]))
        colsum = sum(abs(self.dyn[:, 0]))
        while colsum == 0:
            self.dyn = np.delete(self.dyn, 0, axis=1)
            self.times = np.delete(self.times, 0)
            if len(np.argwhere(self.dyn[:, 0] == 0)) > bandwagon_frac * nr:
                self.dyn[:, 0] = np.zeros(np.shape(self.dyn[:, 0]))
            colsum = sum(abs(self.dyn[:, 0]))
        if len(np.argwhere(self.dyn[:, -1] == 0)) > bandwagon_frac * nr:
            self.dyn[:, -1] = np.zeros(np.shape(self.dyn[:, -1]))
        colsum = sum(abs(self.dyn[:, -1]))
        while colsum == 0:
            self.dyn = np.delete(self.dyn, -1, axis=1)
            self.times = np.delete(self.times, -1)
            if len(np.argwhere(self.dyn[:, -1] == 0)) > bandwagon_frac * nr:
                self.dyn[:, -1] = np.zeros(np.shape(self.dyn[:, -1]))
            colsum = sum(abs(self.dyn[:, -1]))
        self.mjd += np.min(self.times) / 86400
        self.times -= np.min(self.times)
        self.nchan = len(self.freqs)
        self.bw = round(max(self.freqs) - min(self.freqs) + self.df, 3)
        self.freq = round(np.mean(self.freqs), 3)
        self.nsub = len(self.times)
        self.dt = round(np.mean(np.diff(self.times)), 3)
        self.tobs = round(max(self.times) + self.dt, 3)
        self.df = self.bw / self.nchan

    def correct_dyn(self):
        if hasattr(self, 'svd_model'):
            print('Warning: An svd_model exists. Check before applying twice')
        dyn = self.dyn
        dyn[np.isnan(dyn)] = 0
        dyn, model = svd_model(dyn)
        self.svd_model = model
        self.dyn = dyn

    def calc_sspec(self, lamsteps=False, window='hanning', window_frac=0.1):
        if lamsteps:
            if not hasattr(self, 'lamdyn'):
                self.scale_dyn()
            dyn = cp(self.lamdyn)
        else:
            dyn = cp(self.dyn)
        nf = np.shape(dyn)[0]
        nt = np.shape(dyn)[1]
        dyn = dyn - np.mean(dyn)
        if window is not None:
            chan_window, subint_window = get_window(nt, nf, window=window, frac=window_frac)
            dyn = np.multiply(chan_window, dyn)
            dyn = np.transpose(np.multiply(subint_window, np.transpose(dyn)))
        nrfft = int(2 ** (np.ceil(np.log2(nf)) + 1))
        ncfft = int(2 ** (np.ceil(np.log2(nt)) + 1))
        dyn = dyn - np.mean(dyn)
        simpw = dyn
        simf = np.fft.fft2(simpw, s=[nrfft, ncfft])
        simf = np.real(np.multiply(simf, np.conj(simf)))
        sec = np.fft.fftshift(simf)
        sec = sec[int(nrfft / 2):][:]
        td = np.array(list(range(0, int(nrfft / 2))))
        fd = np.array(list(range(int(-ncfft / 2), int(ncfft / 2))))
        fdop = np.reshape(np.multiply(fd, 1000.0 / (ncfft * self.dt)), [len(fd)])
        tdel = np.reshape(np.divide(td, nrfft * self.df), [len(td)])
        if lamsteps:
            beta = np.divide(td, nrfft * self.dlam)
        with np.errstate(divide='ignore'):
            sec = 10 * np.log10(sec)
        if lamsteps:
            self.lamsspec = sec
        else:
            self.sspec = sec
        self.fdop = fdop
        self.tdel = tdel
        if lamsteps:
            self.beta = beta

    def scale_dyn(self):
        arin = cp(self.dyn)
        nf, nt = np.shape(arin)
        freqs = cp(self.freqs)
        lams = np.divide(sc.c, freqs * 10 ** 6)
        dlam = (np.max(lams) - np.min(lams)) / len(freqs)
        lam_eq = np.arange(np.min(lams) + 1e-10, np.max(lams) - 1e-10, dlam)
        self.dlam = dlam
        feq = np.round(np.divide(sc.c, lam_eq) / 10 ** 6, 6)
        arout = np.zeros([len(lam_eq), int(nt)])
        for it in range(0, nt):
            f = interp1d(freqs, arin[:, it], kind='cubic')
            if max(feq) > max(freqs):
                feq[np.argmax(feq)] = max(freqs)
            if min(feq) < min(freqs):
                feq[np.argmin(feq)] = min(freqs)
            arout[:, it] = f(feq)
        self.lamdyn = np.flipud(arout)
        self.lam = np.flipud(lam_eq)
        self.nlam = len(self.lam)

    def norm_sspec(self, eta=None, delmax=None, startbin=1, cutmid=0, numsteps=None):
        delmax = np.max(self.tdel) if delmax is None else delmax
        yaxis = cp(self.beta)
        if not hasattr(self, 'lamsspec'):
            self.calc_sspec(lamsteps=True)
        sspec = cp(self.lamsspec)
        if not hasattr(self, 'betaeta') and eta is None:
            self.fit_arc(delmax=delmax, startbin=startbin)
        if eta is None:
            eta = self.betaeta
        medval = np.median(sspec[is_valid(sspec) * np.array(np.abs(sspec) > 0)])
        maxval = np.max(sspec[is_valid(sspec) * np.array(np.abs(sspec) > 0)])
        vmin = medval - 3
        vmax = maxval - 3
        ind = np.argmin(abs(self.tdel - delmax))
        sspec = sspec[startbin:ind, :]
        nr, nc = np.shape(sspec)
        sspec[:, int(nc / 2 - np.floor(cutmid / 2)):int(nc / 2 + np.floor(cutmid / 2))] = np.nan
        tdel = yaxis[startbin:ind]
        nr, nc = np.shape(sspec)
        fdop = self.fdop
        maxfdop = 1 * np.sqrt(tdel[-1] / eta)
        if maxfdop > max(fdop):
            maxfdop = max(fdop)
        nfdop = 2 * len(fdop[abs(fdop) <= maxfdop]) if numsteps is None else numsteps
        if nfdop % 2 != 0:
            nfdop += 1
        fdopnew = np.linspace(-1, 1, nfdop)
        normSspec = []
        mask = []
        isspectot = np.zeros(np.shape(fdopnew))
        for ii in range(0, len(tdel)):
            itdel = tdel[ii]
            imaxfdop = 1 * np.sqrt(itdel / eta)
            ifdop = fdop[abs(fdop) <= imaxfdop] / np.sqrt(itdel / eta)
            isspec = sspec[ii, abs(fdop) <= imaxfdop]
            normline = np.interp(fdopnew, ifdop, isspec)
            mask.append(np.abs(fdopnew) > np.max(np.abs(ifdop)))
            normSspec.append(normline)
            isspectot = np.add(isspectot, normline)
        mask = np.array(mask).squeeze()
        normSspec = np.array(normSspec).squeeze()
        mask += np.isnan(normSspec)
        normSspec = np.ma.array(normSspec, mask=mask)
        self.mask = mask
        self.powerspectrum = np.ma.mean(np.power(10, normSspec / 10), axis=1)
        xdata = np.sqrt(tdel)
        ydata = np.sqrt(tdel) * self.powerspectrum
        xdata = xdata[~np.isnan(xdata)]
        ydata = ydata[~np.isnan(ydata)]
        alpha = -11 / 3
        index = np.argmin(np.abs(xdata - 10))
        amp = ydata[index] * xdata[index] ** (-alpha)
        wn = np.min(ydata)
        arc_spectrum = amp * xdata ** alpha
        self.weights = np.ones(np.shape(arc_spectrum))
        isspecavg = np.ma.average(normSspec, axis=0, weights=self.weights.squeeze()).squeeze()
        self.normsspecavg = isspecavg
        self.normsspec = normSspec
        self.normsspec_tdel = tdel
        self.normsspec_fdop = fdopnew
        return

    def fit_arc(self, delmax=None, numsteps=10000.0, startbin=3, cutmid=3, etamax=None, etamin=None, low_power_diff=-1, high_power_diff=-0.5, constraint=[0, np.inf], nsmooth=5):
        if not hasattr(self, 'tdel'):
            self.calc_sspec()
        delmax = np.max(self.tdel) if delmax is None else delmax
        if not hasattr(self, 'lamsspec'):
            self.calc_sspec(lamsteps=True)
        sspec = np.array(cp(self.lamsspec))
        yaxis = cp(self.beta)
        ind = np.argmin(abs(self.tdel - delmax))
        ymax = self.beta[ind]
        nr, nc = np.shape(sspec)
        a = np.array(sspec[int(nr / 2):, int(nc / 2 + np.ceil(cutmid / 2)):].ravel())
        b = np.array(sspec[int(nr / 2):, 0:int(nc / 2 - np.floor(cutmid / 2))].ravel())
        noise = np.std(np.concatenate((a, b)))
        ind = np.argmin(abs(self.tdel - delmax))
        sspec[0:startbin, :] = np.nan
        sspec[:, int(nc / 2 - np.floor(cutmid / 2)):int(nc / 2 + np.ceil(cutmid / 2))] = np.nan
        sspec = sspec[0:ind, :]
        yaxis = yaxis[0:ind]
        noise = np.sqrt(np.sum(np.power(noise, 2))) / np.sqrt(len(yaxis) * 2)
        self.noise = noise
        if etamax is None:
            etamax = ymax / ((self.fdop[1] - self.fdop[0]) * cutmid) ** 2
        if etamin is None:
            etamin = (yaxis[1] - yaxis[0]) * startbin / max(self.fdop) ** 2
        try:
            len(etamin)
            etamin_array = np.array(etamin).squeeze()
            etamax_array = np.array(etamax).squeeze()
        except TypeError:
            etamin_array = np.array([etamin])
            etamax_array = np.array([etamax])
        max_sqrt_eta = np.sqrt(np.max(etamax_array))
        min_sqrt_eta = np.sqrt(np.min(etamin_array))
        sqrt_eta_all = np.linspace(min_sqrt_eta, max_sqrt_eta, int(numsteps))
        for iarc in range(0, len(etamin_array)):
            if len(etamin_array) == 1:
                etamin = etamin
                etamax = etamax
            else:
                etamin = etamin_array.squeeze()[iarc]
                etamax = etamax_array.squeeze()[iarc]
            sqrt_eta = sqrt_eta_all[(sqrt_eta_all <= np.sqrt(etamax)) * (sqrt_eta_all >= np.sqrt(etamin))]
            numsteps_new = len(sqrt_eta)
            etaArray = []
            self.norm_sspec(eta=etamin, delmax=delmax, startbin=startbin, cutmid=cutmid, numsteps=numsteps_new)
            norm_sspec = self.normsspecavg.squeeze()
            etafrac_array = self.normsspec_fdop
            ind1 = np.argwhere(etafrac_array >= 0)
            ind2 = np.argwhere(etafrac_array < 0)
            norm_sspec_avg = np.add(norm_sspec[ind1], np.flip(norm_sspec[ind2], axis=0)) / 2
            nspec = 1
            etafrac_array_avg_orig = 1 / etafrac_array[ind1].squeeze()
            for dummy in range(0, nspec):
                etafrac_array_avg = etafrac_array_avg_orig
                spec = np.array(norm_sspec_avg)
                spec = spec.squeeze()
                filt_ind = is_valid(spec)
                spec = np.flip(spec[filt_ind], axis=0)
                etafrac_array_avg = np.flip(etafrac_array_avg[filt_ind], axis=0)
                etaArray = etamin * etafrac_array_avg ** 2
                ind = np.argwhere(etaArray < etamax)
                etaArray = etaArray[ind].squeeze()
                spec = spec[ind].squeeze()
                norm_sspec_avg_filt = savgol_filter(spec, nsmooth, 1)
                indrange = np.argwhere((etaArray > constraint[0]) * (etaArray < constraint[1]))
                sumpow_inrange = norm_sspec_avg_filt[indrange]
                ind = np.argmin(np.abs(norm_sspec_avg_filt - np.max(sumpow_inrange)))
                max_power = norm_sspec_avg_filt[ind]
                power = max_power
                ind1 = 1
                while power > max_power + low_power_diff and ind + ind1 < len(norm_sspec_avg_filt) - 1:
                    ind1 += 1
                    power = norm_sspec_avg_filt[ind - ind1]
                power = max_power
                ind2 = 1
                while power > max_power + high_power_diff and ind + ind2 < len(norm_sspec_avg_filt) - 1:
                    ind2 += 1
                    power = norm_sspec_avg_filt[ind + ind2]
                xdata = etaArray[int(ind - ind1):int(ind + ind2)]
                ydata = spec[int(ind - ind1):int(ind + ind2)]
                yfit, eta, etaerr = fit_parabola(xdata, ydata)
                if np.mean(np.gradient(np.diff(yfit))) > 0:
                    raise ValueError('Fit returned a forward parabola.')
                eta = eta
                etaerr2 = etaerr
                power = max_power
                ind1 = 1
                while power > max_power - noise and ind - ind1 > 1:
                    power = norm_sspec_avg_filt[ind - ind1]
                    ind1 += 1
                power = max_power
                ind2 = 1
                while power > max_power - noise and ind + ind2 < len(norm_sspec_avg_filt) - 1:
                    ind2 += 1
                    power = norm_sspec_avg_filt[ind + ind2]
                etaerr = np.abs(etaArray[int(ind - ind1)] - etaArray[int(ind + ind2)]) / 2
                self.eta_array = etaArray
                sigma = self.noise * 1
                self.norm_sspec_avg = spec
                prob = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((spec - np.max(spec)) / sigma) ** 2)
                self.prob_eta_peak = prob
                if iarc == 0:
                    if dummy == 1:
                        self.betaeta_right = eta
                        self.betaetaerr_right = etaerr / np.sqrt(2)
                        self.betaetaerr2_right = etaerr2 / np.sqrt(2)
                    else:
                        self.betaeta = eta
                        self.betaetaerr = etaerr / np.sqrt(2)
                self.parabola_standard_error = etaerr2
            self.norm_delmax = delmax
