"""Explicit integrated OU/top-hat covariance, extracted from JavelinTNG.
Copyright Ying Zu and JAVELIN contributors, GPL2; see LICENSE and validation.
Only the selected normal spectroscopic covariance is retained.
"""
import numpy as np
def _getcmat_delta_vec(jd1, jd2, tau, tspike1, scale1, tspike2, scale2):
    """Vectorized delta-transfer covariance for arrays of epochs."""
    return abs(scale1 * scale2) * np.exp(-np.abs(jd1 - jd2 - tspike1 + tspike2) / tau)


def _getcmat_lc_vec(id1, id2, jd1, jd2, tau,
                     slag1, swid1, scale1, slag2, swid2, scale2):
    """Vectorized continuum/line (or narrow-line/broad-line) covariance."""
    if id1 == 1 and id2 >= 2:
        tlow = jd2 - jd1 - slag2 - 0.5 * swid2
        thig = jd2 - jd1 - slag2 + 0.5 * swid2
        em = abs(scale2)
        w = swid2
    elif id2 == 1 and id1 >= 2:
        tlow = jd1 - jd2 - slag1 - 0.5 * swid1
        thig = jd1 - jd2 - slag1 + 0.5 * swid1
        em = abs(scale1)
        w = swid1
    elif id1 >= 2 and id2 >= 2 and swid1 <= 0.01:
        tlow = jd2 - (jd1 - slag1) - slag2 - 0.5 * swid2
        thig = jd2 - (jd1 - slag1) - slag2 + 0.5 * swid2
        em = abs(scale2 * scale1)
        w = swid2
    else:
        tlow = jd1 - (jd2 - slag2) - slag1 - 0.5 * swid1
        thig = jd1 - (jd2 - slag2) - slag1 + 0.5 * swid1
        em = abs(scale2 * scale1)
        w = swid1

    if w <= 0:
        return _getcmat_delta_vec(jd1, jd2, tau, slag1, scale1, slag2, scale2)

    out = np.empty(np.broadcast_shapes(np.shape(tlow), np.shape(thig)), dtype=float)
    neg = thig <= 0
    pos = tlow >= 0
    mid = ~(neg | pos)
    out[neg] = np.exp(thig[neg] / tau) - np.exp(tlow[neg] / tau)
    out[pos] = np.exp(-tlow[pos] / tau) - np.exp(-thig[pos] / tau)
    out[mid] = 2.0 - np.exp(tlow[mid] / tau) - np.exp(-thig[mid] / tau)
    return tau * (em / w) * out


def _getcmat_lauto_vec(jd1, jd2, tau, slag, swid, scale):
    """Vectorized line auto-covariance."""
    w = swid
    tmid = jd1 - jd2
    if w <= 0:
        return scale * scale * np.exp(-np.abs(tmid) / tau)

    tlow = tmid - w
    thig = tmid + w
    outer = (thig <= 0) | (tlow >= 0)
    out = np.empty(np.broadcast_shapes(np.shape(jd1), np.shape(jd2)), dtype=float)
    out[outer] = (
        np.exp(-np.abs(tmid[outer]) / tau)
        * (np.exp(0.5 * w / tau) - np.exp(-0.5 * w / tau)) ** 2
    )
    inner = ~outer
    ti = tmid[inner]
    v = -2.0 * np.exp(-np.abs(ti) / tau)
    v += np.exp(-w / tau) * (np.exp(-ti / tau) + np.exp(ti / tau))
    v += np.where(ti >= 0, 2.0 * (w - ti) / tau, 2.0 * (w + ti) / tau)
    out[inner] = v
    return (scale * tau / w) ** 2 * out


def _getcmat_lcross_vec(id1, id2, jd1, jd2, tau,
                         slag1, swid1, scale1, slag2, swid2, scale2):
    """Vectorized covariance between two finite-width transfer functions."""
    if swid1 <= 0.01 or swid2 <= 0.01:
        return _getcmat_lc_vec(
            id1, id2, jd1, jd2, tau,
            slag1, swid1, scale1, slag2, swid2, scale2,
        )

    if swid1 >= swid2:
        t1 = slag1 - 0.5 * swid1
        t2 = slag1 + 0.5 * swid1
        t3 = slag2 - 0.5 * swid2
        t4 = slag2 + 0.5 * swid2
        bottleneck = swid2
        ti, tj = jd1, jd2
    else:
        t1 = slag2 - 0.5 * swid2
        t2 = slag2 + 0.5 * swid2
        t3 = slag1 - 0.5 * swid1
        t4 = slag1 + 0.5 * swid1
        bottleneck = swid1
        ti, tj = jd2, jd1

    dt = ti - tj
    tlow = dt - (t2 - t3)
    tm1 = dt - (t2 - t4)
    tm2 = dt - (t1 - t3)
    thig = dt - (t1 - t4)

    outer = (thig <= 0) | (tlow >= 0)
    out = np.empty(np.broadcast_shapes(np.shape(jd1), np.shape(jd2)), dtype=float)
    out[outer] = (
        np.exp(-np.abs(tlow[outer]) / tau)
        + np.exp(-np.abs(thig[outer]) / tau)
        - np.exp(-np.abs(tm1[outer]) / tau)
        - np.exp(-np.abs(tm2[outer]) / tau)
    )

    inner = ~outer
    lo = tlow[inner]
    m1 = tm1[inner]
    m2 = tm2[inner]
    hi = thig[inner]
    v = (
        np.exp(lo / tau)
        + np.exp(-hi / tau)
        - np.exp(-np.abs(m1) / tau)
        - np.exp(-np.abs(m2) / tau)
    )
    v += np.where(m2 <= 0, 2.0 * hi / tau,
                  np.where(m1 <= 0, 2.0 * bottleneck / tau,
                           np.where(lo < 0, -2.0 * lo / tau, 0.0)))
    out[inner] = v
    return tau * tau * scale1 * scale2 / (swid1 * swid2) * out


def _covmat_normal_block(id1, id2, jd1, jd2, sigma, tau, p1, p2):
    """Vectorized normal SPEAR covariance for one pair of light-curve IDs."""
    slag1, swid1, scale1 = p1
    slag2, swid2, scale2 = p2
    if min(id1, id2) <= 0:
        return np.full(np.broadcast_shapes(np.shape(jd1), np.shape(jd2)), -sigma * sigma)
    if id1 == id2:
        if id1 == 1 or swid1 <= 0.01:
            v = _getcmat_delta_vec(jd1, jd2, tau, slag1, scale1, slag2, scale2)
        else:
            v = _getcmat_lauto_vec(jd1, jd2, tau, slag1, swid1, scale1)
    elif min(id1, id2) == 1:
        if max(swid1, swid2) <= 0.01:
            v = _getcmat_delta_vec(jd1, jd2, tau, slag1, scale1, slag2, scale2)
        else:
            v = _getcmat_lc_vec(id1, id2, jd1, jd2, tau,
                                 slag1, swid1, scale1, slag2, swid2, scale2)
    elif swid1 <= 0.01 and swid2 <= 0.01:
        v = _getcmat_delta_vec(jd1, jd2, tau, slag1, scale1, slag2, scale2)
    elif swid1 <= 0.01 or swid2 <= 0.01:
        v = _getcmat_lc_vec(id1, id2, jd1, jd2, tau,
                             slag1, swid1, scale1, slag2, swid2, scale2)
    else:
        v = _getcmat_lcross_vec(id1, id2, jd1, jd2, tau,
                                 slag1, swid1, scale1, slag2, swid2, scale2)
    return sigma * sigma * v


def _fill_normal_vectorized(mat, jd1, jd2, id1, id2, sigma, tau,
                            lags, wids, scales, cmin=0, cmax=-1):
    """Fill normal SPEAR covariance by vectorized ID blocks.

    This preserves the historical scalar formulas but removes the nested Python
    pair loop that made large mock-generation matrices prohibitively slow.
    """
    jd1 = np.ravel(jd1)
    jd2 = np.ravel(jd2)
    id1 = np.asarray(id1, dtype=int)
    id2 = np.asarray(id2, dtype=int)
    cmax = mat.shape[1] if cmax == -1 else int(cmax)
    col_range = np.arange(mat.shape[1])
    active_cols = (col_range >= int(cmin)) & (col_range < cmax)

    for a in np.unique(id1):
        rows = np.flatnonzero(id1 == a)
        if rows.size == 0:
            continue
        p1 = (lags[a - 1], wids[a - 1], scales[a - 1])
        t1 = jd1[rows][:, None]
        for b in np.unique(id2[active_cols]):
            cols = np.flatnonzero((id2 == b) & active_cols)
            if cols.size == 0:
                continue
            p2 = (lags[b - 1], wids[b - 1], scales[b - 1])
            t2 = jd2[cols][None, :]
            mat[np.ix_(rows, cols)] = _covmat_normal_block(
                int(a), int(b), t1, t2, sigma, tau, p1, p2
            )




def unpack(parameters):
    p=np.asarray(parameters);sigma,tau=np.exp(p[:2]);lag=np.r_[0,p[2::3]];width=np.r_[0,p[3::3]];scale=np.r_[1,p[4::3]]
    return sigma,tau,lag,width,scale

def covariance(t1,id1,t2,id2,parameters):
    t1=np.atleast_1d(t1);t2=np.atleast_1d(t2);id1=np.atleast_1d(id1);id2=np.atleast_1d(id2)
    out=np.empty((len(t1),len(t2)),order='F');_fill_normal_vectorized(out,t1,t2,id1,id2,*unpack(parameters));return out
