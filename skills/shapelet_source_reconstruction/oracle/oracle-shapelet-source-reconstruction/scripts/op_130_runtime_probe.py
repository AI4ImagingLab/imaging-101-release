#!/usr/bin/env python
"""Self-contained smoke test for grid, shapelet, and response-matrix shapes."""
import math
import sys
import numpy as np
import scipy.ndimage


def make_grid(numPix, d):
    N = int(numPix); a = np.arange(N)
    m = np.dstack(np.meshgrid(a, a)).reshape(-1, 2)
    return (m[:, 0] - (N - 1) / 2.0) * d, (m[:, 1] - (N - 1) / 2.0) * d


def array2image(v):
    n = int(np.sqrt(len(v)))
    return v.reshape(n, n)


def image2array(im):
    return im.reshape(im.shape[0] * im.shape[1])


def re_size(im, f):
    if f == 1:
        return im.copy()
    nx, ny = im.shape
    return im.reshape(nx // f, f, ny // f, f).mean(3).mean(1)


def pre_calc(x, y, beta, n_max):
    x_ = x / beta; y_ = y / beta; m = len(x)
    Hx = np.zeros((n_max + 1, m)); Hy = np.zeros((n_max + 1, m))
    ex = np.exp(-x_**2 / 2); ey = np.exp(-y_**2 / 2)
    for n in range(n_max + 1):
        pf = 1.0 / np.sqrt(2.0**n * np.sqrt(np.pi) * math.factorial(n))
        c = np.zeros(n + 1); c[n] = 1
        Hx[n] = np.polynomial.hermite.hermval(x_, c) * pf * ex
        Hy[n] = np.polynomial.hermite.hermval(y_, c) * pf * ey
    return Hx, Hy


def iterate(n_max):
    num = (n_max + 1) * (n_max + 2) // 2
    n1, n2 = 0, 0
    for i in range(num):
        yield i, n1, n2
        if n1 == 0:
            n1 = n2 + 1; n2 = 0
        else:
            n1 -= 1; n2 += 1


def main():
    fails = []
    N, ss, dpix = 8, 2, 0.05
    x, y = make_grid(N * ss, dpix / ss)
    if len(x) != (N * ss) ** 2:
        fails.append('grid length')
    v = np.arange((N * ss) ** 2, dtype=float)
    if not np.array_equal(image2array(array2image(v)), v):
        fails.append('reshape round-trip')
    n_max = 3
    count = sum(1 for _ in iterate(n_max))
    if count != (n_max + 1) * (n_max + 2) // 2:
        fails.append('ordering count')
    Hx, Hy = pre_calc(x, y, 0.15, n_max)
    rows = []
    for _, a, b in iterate(n_max):
        im = array2image(np.nan_to_num(Hx[a] * Hy[b]))
        im = scipy.ndimage.gaussian_filter(im, 1.0, mode='constant', truncate=5)
        im = re_size(im, ss) * dpix**2
        rows.append(image2array(im))
    A = np.array(rows)
    if A.shape != (count, N * N):
        fails.append('A shape %r' % (A.shape,))
    M = (A * np.ones(N * N)[None, :]).dot(A.T)
    if M.shape != (count, count) or not np.all(np.isfinite(M)):
        fails.append('M shape/finite')
    if fails:
        print('PROBE FAIL:', '; '.join(fails)); sys.exit(1)
    print('PROBE OK'); sys.exit(0)


if __name__ == '__main__':
    main()
