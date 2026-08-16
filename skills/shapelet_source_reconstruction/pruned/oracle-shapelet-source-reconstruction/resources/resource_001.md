Cartesian Hermite shapelets (numpy only).

```python
import math
import numpy as np

def pre_calc_shapelets(x, y, beta, n_max, center_x=0.0, center_y=0.0):
    x_ = (x - center_x) / beta
    y_ = (y - center_y) / beta
    m = len(x)
    H_x = np.zeros((n_max + 1, m)); H_y = np.zeros((n_max + 1, m))
    exp_x = np.exp(-x_**2 / 2.0); exp_y = np.exp(-y_**2 / 2.0)
    for n in range(n_max + 1):
        pref = 1.0 / np.sqrt(2.0**n * np.sqrt(np.pi) * math.factorial(n))
        c = np.zeros(n + 1); c[n] = 1
        cut = np.sqrt(n + 1) * 5.0
        mx = np.abs(x_) < cut; my = np.abs(y_) < cut
        hx = np.zeros(m); hy = np.zeros(m)
        if np.any(mx): hx[mx] = np.polynomial.hermite.hermval(x_[mx], c)
        if np.any(my): hy[my] = np.polynomial.hermite.hermval(y_[my], c)
        H_x[n] = hx * pref * exp_x
        H_y[n] = hy * pref * exp_y
    return H_x, H_y

def iterate_n1_n2(n_max):
    num = (n_max + 1) * (n_max + 2) // 2
    n1, n2 = 0, 0
    for i in range(num):
        yield i, n1, n2
        if n1 == 0: n1 = n2 + 1; n2 = 0
        else: n1 -= 1; n2 += 1

def shapelet_function(x, y, amp, n_max, beta, cx=0.0, cy=0.0):
    H_x, H_y = pre_calc_shapelets(x, y, beta, n_max, cx, cy)
    f = np.zeros(len(x))
    for i, n1, n2 in iterate_n1_n2(n_max):
        f += amp[i] * H_x[n1] * H_y[n2]
    return np.nan_to_num(f)

def shapelet_basis_list(x, y, n_max, beta, cx=0.0, cy=0.0):
    H_x, H_y = pre_calc_shapelets(x, y, beta, n_max, cx, cy)
    return [np.nan_to_num(H_x[n1] * H_y[n2]) for _, n1, n2 in iterate_n1_n2(n_max)]
```
