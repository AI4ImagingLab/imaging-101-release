"""Explicit directional min-plus MGM/SGM updates, without a stereo library.

Adapted from MGM, copyright (C) 2015 Gabriele Facciolo,
Carlo de Franchis, Enric Meinhardt; distributed under AGPL-3.0 (LICENSE).
"""
import numpy as np

# q,r,s,t offsets (x,y), increasing-x flag, increasing-y flag, row-major flag.
PASSES = (
    ((-1,0),(0,-1),(-1,-1),(1,-1),1,1,1),
    ((1,0),(0,1),(1,1),(-1,1),0,0,1),
    ((0,1),(-1,0),(-1,1),(-1,-1),1,0,0),
    ((0,-1),(1,0),(1,-1),(1,1),0,1,0),
    ((-1,-1),(1,-1),(0,-1),(1,0),0,1,1),
    ((1,-1),(1,1),(1,0),(0,1),0,0,0),
    ((1,1),(-1,1),(0,1),(-1,0),1,0,1),
    ((-1,1),(-1,-1),(-1,0),(0,-1),1,1,0),
)

def minplus_message(values, p1, p2):
    """Min over labels with V(0)=0,V(1)=p1,V(other)=p2, minus min(values)."""
    values = np.asarray(values, dtype=np.float32)
    minimum = values.min(axis=-1, keepdims=True)
    adjacent = np.full_like(values, np.inf)
    adjacent[..., 1:] = values[..., :-1]
    adjacent[..., :-1] = np.minimum(adjacent[..., :-1], values[..., 1:])
    return np.minimum(np.minimum(values, adjacent+np.float32(p1)), minimum+np.float32(p2))-minimum

def aggregate(cost, method='mgm', directions=8, p1=24., p2=96., capture_points=None):
    """Aggregate HWD float32 unary costs and return native labels plus diagnostics.

    MGM averages two predecessors; SGM and ocSGM use one. Only ocSGM/MGM
    subtract repeated unary costs. All four source boundary guards are retained.
    Wavefront groups contain independent pixels, so vectorization preserves the
    original causal traversal and float32 operation order.
    """
    if method not in ('mgm', 'sgm', 'ocsgm', 'wta') or directions not in (2,4,8):
        raise ValueError('Unsupported aggregation method or direction count')
    cost = np.asarray(cost, dtype=np.float32)
    h, w, _ = cost.shape
    points = None if capture_points is None else np.asarray(capture_points, dtype=int)
    def sample(volume):
        return volume[points[:,0], points[:,1]].copy()
    diagnostics = {'pass_samples': []}
    if points is not None:
        diagnostics['unary_samples'] = sample(cost)
    total = np.zeros_like(cost)
    if method == 'wta':
        total[:] = cost
    else:
        for setup in PASSES[:directions]:
            neighbours, ix, iy, row_major = setup[:4], *setup[4:]
            local = cost.copy()
            maxii, maxjj = (w, h) if row_major else (h, w)
            jj = np.arange(maxjj)
            for ii in range(maxii+2*maxjj):
                xx = ii-2*jj
                keep = (xx >= 0) & (xx < maxii)
                x, y = xx[keep], jj[keep]
                if not row_major:
                    x, y = y, x
                if not ix:
                    x = w-1-x
                if not iy:
                    y = h-1-y
                valid = np.ones(x.size, dtype=bool)
                for dx, dy in neighbours:
                    valid &= (x+dx >= 0) & (x+dx < w) & (y+dy >= 0) & (y+dy < h)
                x, y = x[valid], y[valid]
                if not x.size:
                    continue
                qx, qy = neighbours[0]
                edge = minplus_message(local[y+qy, x+qx], p1, p2)
                if method == 'mgm':
                    rx, ry = neighbours[1]
                    edge = edge/np.float32(2) + minplus_message(local[y+ry, x+rx], p1, p2)/np.float32(2)
                local[y, x] = cost[y, x] + edge
            total += local
            if points is not None:
                diagnostics['pass_samples'].append(sample(local))
        del local
    if points is not None:
        diagnostics['sum_before_correction_samples'] = sample(total)
    if method in ('mgm', 'ocsgm'):
        # Invalid correspondences deliberately become NaN through inf-inf, just
        # as in the source; they are excluded from WTA, never from observations.
        with np.errstate(invalid='ignore'):
            total -= np.float32(directions-1)*cost
    if points is not None:
        diagnostics['aggregate_samples'] = sample(total)
    total[~np.isfinite(total)] = np.inf
    if not np.isfinite(total).any(axis=-1).all():
        raise ValueError('A pixel has no finite aggregate label')
    return np.argmin(total, axis=-1), diagnostics
