import numpy as np


def compute_metrics(y_true, y_pred, params_pred, params_true=None):
    """
    Computes quantitative metrics for the inversion result.

    Args:
        y_true (np.ndarray): Measured spectrum.
        y_pred (np.ndarray): Fitted spectrum.
        params_pred (dict): Retrieved parameters.
        params_true (dict, optional): Ground truth parameters.

    Returns:
        dict: Dictionary of computed metrics.
    """
    metrics = {}

    # MSE
    mse = float(np.mean((y_true - y_pred) ** 2))
    metrics['mse'] = mse

    # PSNR
    if mse > 0:
        psnr = float(10 * np.log10(1.0 / mse))
        metrics['psnr_dB'] = psnr

    # NCC (cosine similarity)
    norm_true = np.linalg.norm(y_true)
    norm_pred = np.linalg.norm(y_pred)
    if norm_true > 0 and norm_pred > 0:
        ncc = float(np.dot(y_true.ravel(), y_pred.ravel()) / (norm_true * norm_pred))
        metrics['ncc'] = ncc

    # NRMSE (normalised by dynamic range of reference)
    drange = float(y_true.max() - y_true.min())
    if drange > 0:
        nrmse = float(np.sqrt(mse) / drange)
        metrics['nrmse'] = nrmse

    # Temperature error
    if params_true is not None:
        T_true = params_true['temperature']
        T_pred = params_pred['temperature']
        metrics['temperature_error_K'] = float(abs(T_true - T_pred))
        metrics['temperature_true_K'] = float(T_true)
        metrics['temperature_pred_K'] = float(T_pred)

    return metrics


def plot_inversion_result(nu_axis, y_measured, y_pred, params_pred,
                          y_ground_truth=None, params_true=None,
                          save_path='inversion_result.png'):
    """
    Plots the inversion result and saves to file.

    Args:
        nu_axis (np.ndarray): Wavenumber axis.
        y_measured (np.ndarray): Measured (noisy) spectrum.
        y_pred (np.ndarray): Fitted spectrum.
        params_pred (dict): Retrieved parameters.
        y_ground_truth (np.ndarray, optional): Clean ground truth spectrum.
        params_true (dict, optional): Ground truth parameters.
        save_path (str): Path to save the figure.
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.plot(nu_axis, y_measured, 'k.', label='Measured')
    if y_ground_truth is not None and params_true is not None:
        plt.plot(nu_axis, y_ground_truth, 'b--', linewidth=1.5,
                 label=f'Ground Truth (T={params_true["temperature"]}K)')
    plt.plot(nu_axis, y_pred, 'r-', linewidth=2,
             label=f'Fit (T={params_pred["temperature"]:.0f}K)')
    plt.xlabel('Wavenumber (cm-1)')
    plt.ylabel('Normalized Intensity')
    plt.title('CARS Inversion Result')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Plot saved to '{save_path}'")
