"""
Gauss-Newton material decomposition solver for dual-energy CT.

Adapted from gjadick/dex-ct-sim matdecomp.py optimize_sino_cpu().
The algorithm minimises the Poisson negative log-likelihood in the sinogram
domain using analytical gradient and Hessian per detector bin.
"""

import numpy as np


def gauss_newton_decompose(sinograms, spectra, mus, n_iters=20,
                           dE=1.0, eps=1e-6, verbose=True):
    """Gauss-Newton material decomposition in the sinogram domain.

    Minimises the Poisson negative log-likelihood per sinogram bin:
        L(a) = sum_m [nu_m(a) - g_m * ln(nu_m(a))]
    where nu_m(a) = sum_E I_{0,m}(E) * exp(-sum_k a_k * mu_k(E)) * dE.

    This is a direct adaptation of optimize_sino_cpu() from the reference code
    (gjadick/dex-ct-sim). The outer loop is over projection views; the inner
    loop performs Newton iterations. All detector bins within one view are
    processed simultaneously (vectorised over bins).

    Parameters
    ----------
    sinograms : ndarray, shape (nMeas, nBins, nAngles)
        Measured photon counts. nMeas = number of energy measurements (2).
    spectra : ndarray, shape (nMeas, nE)
        Incident photon spectra I_0(E) for each measurement.
    mus : ndarray, shape (nMats, nE)
        Mass attenuation coefficients for each basis material.
    n_iters : int
        Number of Gauss-Newton iterations per view.
    dE : float
        Energy bin width in keV.
    eps : float
        Initial value for material line integrals.
    verbose : bool
        Print progress every 20 views.

    Returns
    -------
    material_sinograms : ndarray, shape (nMats, nBins, nAngles)
        Estimated density line integrals for each material.
    """
    nMeas, nBins, nAngles = sinograms.shape
    nMats, nE = mus.shape

    # Working array: (nAngles, nBins, nMats) -- matches reference layout
    sino_aa = np.full((nAngles, nBins, nMats), eps, dtype=np.float64)

    # Precompute spectrum-weighted attenuation terms
    # ssff[m, k, :, e] = I0_m(E) * mu_k(E) for gradient
    # shape: (nMeas, nMats, 1, nE)  -- broadcast over bins
    i0 = spectra * dE  # (nMeas, nE)
    ssff = i0[:, np.newaxis, np.newaxis, :] * mus[np.newaxis, :, np.newaxis, :]
    # ssff: (nMeas, nMats, 1, nE)

    # ssff2[m, k, l, :, e] = I0_m(E) * mu_k(E) * mu_l(E) for Hessian
    mu_outer = mus[:, np.newaxis, :] * mus[np.newaxis, :, :]  # (nMats, nMats, nE)
    ssff2 = i0[:, np.newaxis, np.newaxis, np.newaxis, :] * \
            mu_outer[np.newaxis, :, :, np.newaxis, :]
    # ssff2: (nMeas, nMats, nMats, 1, nE)

    # Measured counts reshaped for view loop: sinograms is (nMeas, nBins, nAngles)
    g = sinograms  # (nMeas, nBins, nAngles)

    for j in range(nAngles):
        if verbose and j % 40 == 0:
            print(f"  View {j}/{nAngles}")

        g_j = g[:, :, j]  # (nMeas, nBins)

        for _k in range(n_iters):
            # Current material estimates: sino_aa[j] has shape (nBins, nMats)
            a = sino_aa[j]  # (nBins, nMats)

            # Attenuation: exp(-sum_k a_k * mu_k(E))
            # a: (nBins, nMats), mus: (nMats, nE)
            # exponent: (nBins, nE) = sum over materials
            exponent = -a @ mus  # (nBins, nE)
            exponent = np.clip(exponent, -700, 700)
            atten = np.exp(exponent)  # (nBins, nE)

            # Predicted counts: nu_m = sum_E I0_m(E) * atten(E) * dE
            # i0: (nMeas, nE), atten: (nBins, nE)
            nu = atten @ i0.T  # (nBins, nMeas)
            nu = nu.T  # (nMeas, nBins)

            # Avoid division by zero
            nu = np.maximum(nu, 1e-20)

            # Gradient of nu w.r.t. a_k:
            # dnu_m/da_k = -sum_E I0_m(E) * mu_k(E) * atten(E) * dE
            # ssff: (nMeas, nMats, 1, nE), atten: (1, 1, nBins, nE)
            nu_grad = -np.sum(
                ssff * atten[np.newaxis, np.newaxis, :, :], axis=3
            )  # (nMeas, nMats, nBins)

            # Hessian of nu:
            # d^2 nu_m / da_k da_l = sum_E I0_m(E) * mu_k(E) * mu_l(E) * atten(E) * dE
            nu_hess = np.sum(
                ssff2 * atten[np.newaxis, np.newaxis, np.newaxis, :, :], axis=4
            )  # (nMeas, nMats, nMats, nBins)

            # Poisson NLL gradient:
            # dL/da_k = -sum_m (g_m/nu_m - 1) * dnu_m/da_k
            residual = g_j / nu - 1.0  # (nMeas, nBins)
            dF = -np.sum(
                residual[:, np.newaxis, :] * nu_grad, axis=0
            )  # (nMats, nBins)

            # Poisson NLL Hessian:
            # H_kl = -sum_m [(g_m/nu_m - 1) * d^2 nu/da_k da_l
            #                - (g_m/nu_m^2) * dnu/da_k * dnu/da_l]
            H = -np.sum(
                residual[:, np.newaxis, np.newaxis, :] * nu_hess
                - (g_j / (nu * nu))[:, np.newaxis, np.newaxis, :] *
                  (nu_grad[:, np.newaxis, :, :] * nu_grad[:, :, np.newaxis, :]),
                axis=0
            )  # (nMats, nMats, nBins)

            # Solve H @ delta_a = dF for each bin
            # H: (nMats, nMats, nBins) -> (nBins, nMats, nMats)
            H_t = H.transpose(2, 0, 1)  # (nBins, nMats, nMats)
            dF_t = dF.T  # (nBins, nMats)

            # Newton update: a -= H^{-1} @ dF
            # For 2x2, use explicit inverse for speed
            if nMats == 2:
                det = H_t[:, 0, 0] * H_t[:, 1, 1] - H_t[:, 0, 1] * H_t[:, 1, 0]
                det = np.where(np.abs(det) < 1e-30, 1e-30, det)
                inv00 = H_t[:, 1, 1] / det
                inv01 = -H_t[:, 0, 1] / det
                inv10 = -H_t[:, 1, 0] / det
                inv11 = H_t[:, 0, 0] / det
                step0 = inv00 * dF_t[:, 0] + inv01 * dF_t[:, 1]
                step1 = inv10 * dF_t[:, 0] + inv11 * dF_t[:, 1]
                step = np.stack([step0, step1], axis=1)
            else:
                try:
                    step = np.linalg.solve(H_t, dF_t)
                except np.linalg.LinAlgError:
                    step = np.zeros_like(dF_t)

            sino_aa[j] -= step

            # Clamp to non-negative (density line integrals must be >= 0)
            sino_aa[j] = np.maximum(sino_aa[j], 0.0)

    # Reshape to (nMats, nBins, nAngles)
    result = sino_aa.transpose(2, 1, 0)  # (nMats, nBins, nAngles)
    return result


def reconstruct_material_maps(material_sinograms, theta, image_size,
                              pixel_size=0.1):
    """Reconstruct material density maps from decomposed sinograms using FBP.

    Parameters
    ----------
    material_sinograms : ndarray, shape (nMats, nBins, nAngles)
        Estimated density line integrals per material (in g/cm^2).
    theta : ndarray, shape (nAngles,)
        Projection angles in degrees.
    image_size : int
        Output image size.
    pixel_size : float
        Pixel size in cm (used to convert from physical to pixel units for FBP).

    Returns
    -------
    material_maps : ndarray, shape (nMats, image_size, image_size)
        Reconstructed density maps for each material (g/cm^3).
    """
    from .physics_model import fbp_reconstruct

    nMats = material_sinograms.shape[0]
    maps = np.zeros((nMats, image_size, image_size), dtype=np.float64)
    for k in range(nMats):
        # Convert sinogram from physical (g/cm^2) to pixel units before FBP
        sino_pixels = material_sinograms[k] / pixel_size
        maps[k] = fbp_reconstruct(sino_pixels, theta,
                                  output_size=image_size)
    return maps
