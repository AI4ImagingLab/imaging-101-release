"""Visualization and metrics for MCR hyperspectral reconstruction."""

import numpy as np


def compute_metrics(estimate, reference):
    """Compute NRMSE and NCC between estimate and reference.

    Parameters
    ----------
    estimate : ndarray
        Reconstructed array (any shape, will be flattened).
    reference : ndarray
        Ground truth array (same shape as estimate).

    Returns
    -------
    metrics : dict
        Keys: 'nrmse', 'ncc'.
    """
    est = estimate.ravel().astype(np.float64)
    ref = reference.ravel().astype(np.float64)

    dynamic_range = ref.max() - ref.min()
    rmse = np.sqrt(np.mean((est - ref) ** 2))
    nrmse = rmse / dynamic_range if dynamic_range > 0 else float("inf")

    norm_est = np.linalg.norm(est)
    norm_ref = np.linalg.norm(ref)
    ncc = float(np.dot(est, ref) / (norm_est * norm_ref)) if (norm_est > 0 and norm_ref > 0) else 0.0

    return {"nrmse": nrmse, "ncc": ncc}


def compute_method_metrics(result, conc_ravel, spectra, hsi_noisy):
    """Compute per-method comparison metrics.

    Parameters
    ----------
    result : dict
        Single method result from run_all_methods().
    conc_ravel : ndarray, shape (n_pixels, n_components)
        True concentrations.
    spectra : ndarray, shape (n_components, n_freq)
        True spectra.
    hsi_noisy : ndarray, shape (n_pixels, n_freq)
        Noisy observations.

    Returns
    -------
    metrics : dict
        Concentration metrics, spectral metrics, and reconstruction MSE.
    """
    mcr_obj = result["mcr"]
    select = result["select"]

    C_retr = mcr_obj.C_opt_[:, select]
    ST_retr = mcr_obj.ST_opt_[select, :]

    conc_metrics = compute_metrics(C_retr, conc_ravel)
    spec_metrics = compute_metrics(ST_retr, spectra)
    data_metrics = compute_metrics(mcr_obj.D_opt_, hsi_noisy)

    delta_C = conc_ravel - C_retr
    delta_ST = spectra - ST_retr
    delta_D = hsi_noisy - mcr_obj.D_opt_

    return {
        "conc_nrmse": conc_metrics["nrmse"],
        "conc_ncc": conc_metrics["ncc"],
        "spec_nrmse": spec_metrics["nrmse"],
        "spec_ncc": spec_metrics["ncc"],
        "data_nrmse": data_metrics["nrmse"],
        "data_ncc": data_metrics["ncc"],
        "mse": result["mse"],
        "delta_C_median": float(np.median(delta_C)),
        "delta_C_iqr": float(np.subtract(*np.percentile(delta_C, [75, 25]))),
        "delta_ST_median": float(np.median(delta_ST)),
        "delta_ST_iqr": float(np.subtract(*np.percentile(delta_ST, [75, 25]))),
        "delta_D_median": float(np.median(delta_D)),
        "delta_D_iqr": float(np.subtract(*np.percentile(delta_D, [75, 25]))),
    }


def plot_spectral_components(ax, wn, spectra, title="Spectral Components"):
    """Plot spectral components on a given axes."""
    for k in range(spectra.shape[0]):
        ax.plot(wn, spectra[k], label="Spectrum {}".format(k))
    ax.set_xlabel("Frequency (au)")
    ax.set_ylabel("Intensity (au)")
    ax.legend(loc="best")
    ax.set_title(title)


def plot_concentration_maps(fig, conc, suptitle=None):
    """Plot concentration maps for each component plus total.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to plot into (should be created with appropriate size).
    conc : ndarray, shape (M, N, n_components)
        Concentration maps.
    suptitle : str, optional
        Super-title for the figure.
    """
    import matplotlib.pyplot as plt

    n_components = conc.shape[2]
    for k in range(n_components):
        ax = fig.add_subplot(1, n_components + 1, k + 1)
        im = ax.imshow(conc[..., k])
        fig.colorbar(im, ax=ax, fraction=0.025)
        ax.axis("off")
        ax.set_title("Component {}".format(k))

    ax = fig.add_subplot(1, n_components + 1, n_components + 1)
    im = ax.imshow(conc.sum(axis=-1), clim=[0.9, 1.1])
    fig.colorbar(im, ax=ax, fraction=0.025)
    ax.axis("off")
    ax.set_title("Total Conc.")

    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()


def plot_comparison_boxplots(results, conc_ravel, spectra, hsi_noisy, method_names):
    """Create the 6-panel comparison figure (boxplots + convergence).

    Parameters
    ----------
    results : list of dict
        Output of run_all_methods().
    conc_ravel : ndarray
        True concentrations.
    spectra : ndarray
        True spectra.
    hsi_noisy : ndarray
        Noisy observations.
    method_names : list of str
        Method names for labels.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 2, figsize=(5.5, 6))

    # Delta S boxplot
    delta_s = []
    for r in results:
        sel = r["select"]
        delta_s.append((spectra - r["mcr"].ST_opt_[sel, :]).ravel())

    out = axes[0, 0].boxplot(delta_s, whis=[0, 100], patch_artist=True)
    for i, box in enumerate(out["boxes"]):
        box.set_facecolor("C{}".format(i))
        box.set_alpha(0.5)
        out["medians"][i].set_color("k")
    axes[0, 0].set_xticklabels([""] * len(results))
    axes[0, 0].set_ylabel("$\\Delta$S (au)")

    # Delta C boxplot
    delta_c = []
    for r in results:
        sel = r["select"]
        delta_c.append((conc_ravel - r["mcr"].C_opt_[:, sel]).ravel())

    out = axes[1, 0].boxplot(delta_c, whis=[0, 100], patch_artist=True)
    for i, box in enumerate(out["boxes"]):
        box.set_facecolor("C{}".format(i))
        box.set_alpha(0.5)
        out["medians"][i].set_color("k")
    axes[1, 0].set_xticklabels([""] * len(results))
    axes[1, 0].set_ylabel("$\\Delta$C (au)")

    # Delta D boxplot
    delta_d = []
    for r in results:
        delta_d.append((hsi_noisy - r["mcr"].D_opt_).ravel())

    out = axes[2, 0].boxplot(delta_d, whis=[0, 100], patch_artist=True)
    for i, box in enumerate(out["boxes"]):
        box.set_facecolor("C{}".format(i))
        box.set_alpha(0.5)
        out["medians"][i].set_color("k")
    axes[2, 0].set_xticklabels(method_names, rotation=45, ha="right")
    axes[2, 0].set_ylabel("$\\Delta$D (au)")
    axes[2, 0].set_ylim(-2500, 2500)

    # MSE convergence
    for i, r in enumerate(results):
        ls = "--" if i == 4 else "-"
        axes[0, 1].semilogy(r["mcr"].err, label=method_names[i], ls=ls, lw=0.5)
    axes[0, 1].set_ylabel("MSE (au)")
    axes[0, 1].set_xlabel("N Iterations")
    axes[0, 1].legend(frameon=False, fontsize=6)

    # Time per iteration
    times = [r["elapsed"] / r["n_iter"] for r in results]
    axes[1, 1].bar(range(len(results)), times, color=["C{}".format(i) for i in range(len(results))])
    axes[1, 1].set_ylabel("Time per Iter. (s)")
    axes[1, 1].set_xticks(range(len(results)))
    axes[1, 1].set_xticklabels([""] * len(results))

    # N iterations
    n_iters = [r["n_iter_opt"] for r in results]
    axes[2, 1].bar(range(len(results)), n_iters, color=["C{}".format(i) for i in range(len(results))])
    axes[2, 1].set_ylabel("N Iterations")
    axes[2, 1].set_xticks(range(len(results)))
    axes[2, 1].set_xticklabels(method_names, rotation=45, ha="right")

    fig.tight_layout(pad=0, w_pad=0, h_pad=0)
    return fig


def plot_method_result(wn, result, conc_shape, method_name):
    """Plot retrieved spectra and concentration maps for one method.

    Parameters
    ----------
    wn : ndarray, shape (n_freq,)
        Wavenumber axis.
    result : dict
        Single method result.
    conc_shape : tuple
        (M, N) spatial dimensions.
    method_name : str
        Name for the figure title.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    mcr_obj = result["mcr"]
    select = result["select"]

    fig = plt.figure(figsize=(5.5, 4))

    ax = fig.add_subplot(2, 2, 1)
    for k_idx, k in enumerate(select):
        ax.plot(wn, mcr_obj.ST_opt_[k], label=str(k_idx))
    ax.set_xlabel("Frequency (au)")
    ax.set_ylabel("Intensity (au)")
    ax.legend(loc="upper right", frameon=False, fontsize=7)
    ax.set_title("Retrieved Spectra")

    for k_idx, k in enumerate(select):
        ax = fig.add_subplot(2, 2, k_idx + 2)
        im = ax.imshow(mcr_obj.C_opt_[:, k].reshape(conc_shape), clim=[0, 1])
        fig.colorbar(im, ax=ax, fraction=0.025)
        ax.axis("off")
        ax.set_title("Component {}".format(k_idx))

    fig.suptitle(method_name, y=0.98)
    fig.tight_layout()
    return fig
