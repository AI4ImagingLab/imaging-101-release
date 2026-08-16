# CG inner loop (Dai-Yuan + backtracking)

```python
def direction_dai_yuan(xp, grad1, grad0=None, dir_prev=None):
    if dir_prev is None:
        return -grad1
    diff = grad1 - grad0
    denom = float(xp.sum(xp.conj(dir_prev) * diff).real) + 1e-32
    beta = float(xp.sum(xp.abs(grad1) ** 2).real) / denom
    return -grad1 + beta * dir_prev

def line_search(cost_fn, x, d, step_length, cost_x=None, step_shrink=0.5):
    if cost_x is None:
        cost_x = cost_fn(x)
    while True:
        x_new = x + step_length * d
        cost_new = cost_fn(x_new)
        if cost_new <= cost_x:
            break
        step_length *= step_shrink
        if step_length < 1e-32:
            return 0.0, cost_x, x
    return step_length, cost_new, x_new

def conjugate_gradient(obj, data, theta, tilt, num_iter, step_length, xp,
                       gradient, cost_function):
    def cost_fn(x):
        return cost_function(x, data, theta, tilt)
    cost = None; grad0 = None; dir_ = None
    for _ in range(num_iter):
        grad1 = gradient(obj, data, theta, tilt)
        dir_ = direction_dai_yuan(xp, grad1, grad0, dir_)
        grad0 = grad1
        step_length, cost, obj = line_search(cost_fn, obj, dir_, step_length, cost_x=cost)
    if cost is None:
        cost = cost_fn(obj)
    return obj, cost
```
