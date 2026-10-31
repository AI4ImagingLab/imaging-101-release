"""Independent derived-observation operator for the v2 gradient estimate.

No reference reconstruction or reference-solver imports are used.
"""
from pathlib import Path
import numpy as np


def derived_gradients(raw_path):
    with np.load(raw_path, allow_pickle=False) as f:
        image = f['image'][0].astype(np.float64)
        carriers = f['reference_vectors'][0]
    image -= image.mean()
    n, m = image.shape
    y, x = np.indices((n, m))
    frequency_y = np.fft.fftfreq(n)[:, None]
    frequency_x = np.fft.fftfreq(m)[None, :]
    lengths = np.linalg.norm(carriers, axis=1)
    sigma = int(np.ceil(1 / lengths.min()))
    halfwidth = lengths.mean() / 2.5
    step = halfwidth / 3
    window = np.exp(-2*np.pi**2*sigma**2*(frequency_y**2+frequency_x**2))
    signals = []
    for reference in carriers:
        best = np.zeros((n, m), dtype=np.complex128)
        for ky in np.arange(reference[0]-halfwidth, reference[0]+halfwidth, step):
            for kx in np.arange(reference[1]-halfwidth, reference[1]+halfwidth, step):
                value = np.fft.ifft2(np.fft.fft2(image*np.exp(2j*np.pi*(ky*y+kx*x)))*window)
                value *= np.exp(-2j*np.pi*((ky-reference[0])*y+(kx-reference[1])*x))
                choose = np.abs(value) > np.abs(best)
                best[choose] = value[choose]
        signals.append(best)
    signals = np.asarray(signals)
    interior = np.zeros((n, m))
    interior[2*sigma:-2*sigma, 2*sigma:-2*sigma] = 1
    amplitude = np.abs(signals)*(interior+1e-6)
    phase = np.angle(signals)
    wrap = lambda a: (a+np.pi)%(2*np.pi)-np.pi
    K = 2*np.pi*carriers
    gradients = []
    for axis in (2, 1):
        phase_delta = wrap(np.diff(phase, axis=axis))
        rows, columns = phase_delta.shape[1:]
        weights = amplitude[:, :rows, :columns]**2
        # Solve each 3-by-2 weighted least-squares problem independently.
        normal = np.einsum('jrc,ja,jb->rcab', weights, K, K)
        rhs = np.einsum('jrc,jrc,ja->rca', weights, phase_delta, K)
        gradients.append(wrap(np.linalg.solve(normal, rhs[..., None])[..., 0].transpose(2, 0, 1)))
    confidence = np.sum(amplitude**2, axis=0)
    wx = np.minimum(confidence[:, :-1], confidence[:, 1:])
    wy = np.minimum(confidence[:-1], confidence[1:])
    return gradients[0], gradients[1], wx, wy


def residual_metrics(displacement_nm, raw_path):
    gx, gy, wx, wy = derived_gradients(raw_path)
    u = displacement_nm / .88
    def divergence(a, b):
        result = np.zeros(u.shape)
        result[:, :, :-1] -= a
        result[:, :, 1:] += a
        result[:, :-1, :] -= b
        result[:, 1:, :] += b
        return result
    rhs = divergence(wx*gx, wy*gy)
    residual = divergence(wx*(np.diff(u, axis=2)-gx), wy*(np.diff(u, axis=1)-gy))
    stationarity = np.linalg.norm(residual.reshape(2, -1), axis=1)/np.linalg.norm(rhs.reshape(2, -1), axis=1)
    return {'stationarity_max': float(stationarity.max()), 'mean_displacement_nm': float(np.max(np.abs(u.mean(axis=(1,2))*.88)))}
