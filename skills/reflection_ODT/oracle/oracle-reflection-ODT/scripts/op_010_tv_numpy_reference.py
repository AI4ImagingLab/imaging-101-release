import numpy as np

def gradient_2d(img):
    dx = np.zeros_like(img)
    dy = np.zeros_like(img)
    dx[:, :-1] = img[:, 1:] - img[:, :-1]
    dy[:-1, :] = img[1:, :] - img[:-1, :]
    return dx, dy

def divergence_2d(px, py):
    div = np.zeros_like(px)
    div[:, 0] = px[:, 0]
    div[:, 1:-1] = px[:, 1:-1] - px[:, :-2]
    div[:, -1] = -px[:, -2]
    div[0, :] += py[0, :]
    div[1:-1, :] += py[1:-1, :] - py[:-2, :]
    div[-1, :] += -py[-2, :]
    return div

def tv_2d_proximal_single(img, tau, n_iter=20):
    if tau <= 0:
        return img.copy()
    px = np.zeros_like(img)
    py = np.zeros_like(img)
    for _ in range(n_iter):
        div_p = divergence_2d(px, py)
        x_hat = img - tau * div_p
        gx, gy = gradient_2d(x_hat)
        px_new = px + gx / (8.0 * tau)
        py_new = py + gy / (8.0 * tau)
        norm = np.maximum(np.sqrt(px_new**2 + py_new**2), 1.0)
        px = px_new / norm
        py = py_new / norm
    return img - tau * divergence_2d(px, py)

if __name__ == '__main__':
    rng = np.random.default_rng(0)
    img = rng.normal(size=(8, 8))
    out = tv_2d_proximal_single(img, 4e-6, 20)
    print('input std', float(img.std()), 'output std', float(out.std()))
