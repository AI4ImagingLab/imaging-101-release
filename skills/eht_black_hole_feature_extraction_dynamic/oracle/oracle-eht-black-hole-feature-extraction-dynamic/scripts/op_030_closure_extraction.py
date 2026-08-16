"""Closure phase / log closure amplitude extraction from per-baseline visibilities.

Self-contained; run with the run-local python. No private paths.
"""
import numpy as np
from itertools import combinations


def _find_triangles(station_ids):
    bmap = {}
    for idx, (s1, s2) in enumerate(station_ids):
        bmap[(s1, s2)] = (idx, +1)
        bmap[(s2, s1)] = (idx, -1)
    stations = sorted(set(station_ids.ravel()))
    tris = []
    for i, j, k in combinations(stations, 3):
        if (i, j) in bmap and (j, k) in bmap and (k, i) in bmap:
            i1, g1 = bmap[(i, j)]
            i2, g2 = bmap[(j, k)]
            i3, g3 = bmap[(k, i)]
            tris.append((i1, g1, i2, g2, i3, g3))
    return tris


def _find_quadrangles(station_ids):
    bmap = {}
    for idx, (s1, s2) in enumerate(station_ids):
        bmap[(min(s1, s2), max(s1, s2))] = idx
    stations = sorted(set(station_ids.ravel()))
    quads = []
    for i, j, k, l in combinations(stations, 4):
        pairs = [(i, j), (i, k), (i, l), (j, k), (j, l), (k, l)]
        if all(p in bmap for p in pairs):
            quads.append((bmap[(i, j)], bmap[(k, l)], bmap[(i, k)], bmap[(j, l)]))
    return quads


def extract_closure_indices(frame_data):
    vis = frame_data['vis']
    sigma = frame_data['vis_sigma']
    station_ids = frame_data['station_ids']

    tris = _find_triangles(station_ids)
    cphase_ind = [np.zeros(len(tris), dtype=np.int64) for _ in range(3)]
    cphase_sign = [np.zeros(len(tris), dtype=np.float64) for _ in range(3)]
    cphase_vals = np.zeros(len(tris))
    cphase_sig = np.zeros(len(tris))
    for t, (i1, g1, i2, g2, i3, g3) in enumerate(tris):
        cphase_ind[0][t], cphase_ind[1][t], cphase_ind[2][t] = i1, i2, i3
        cphase_sign[0][t], cphase_sign[1][t], cphase_sign[2][t] = g1, g2, g3
        a1 = np.angle(vis[i1]) * g1
        a2 = np.angle(vis[i2]) * g2
        a3 = np.angle(vis[i3]) * g3
        cphase_vals[t] = np.degrees(a1 + a2 + a3)
        m1 = np.abs(vis[i1]) + 1e-20
        m2 = np.abs(vis[i2]) + 1e-20
        m3 = np.abs(vis[i3]) + 1e-20
        sr = np.sqrt((sigma[i1] / m1) ** 2 + (sigma[i2] / m2) ** 2 + (sigma[i3] / m3) ** 2)
        cphase_sig[t] = np.degrees(sr)

    quads = _find_quadrangles(station_ids)
    camp_ind = [np.zeros(len(quads), dtype=np.int64) for _ in range(4)]
    camp_vals = np.zeros(len(quads))
    logca_vals = np.zeros(len(quads))
    logca_sig = np.zeros(len(quads))
    for q, (ij, kl, ik, jl) in enumerate(quads):
        camp_ind[0][q], camp_ind[1][q], camp_ind[2][q], camp_ind[3][q] = ij, kl, ik, jl
        aij = np.abs(vis[ij]) + 1e-20
        akl = np.abs(vis[kl]) + 1e-20
        aik = np.abs(vis[ik]) + 1e-20
        ajl = np.abs(vis[jl]) + 1e-20
        ca = (aij * akl) / (aik * ajl)
        camp_vals[q] = ca
        logca_vals[q] = np.log(ca)
        logca_sig[q] = np.sqrt((sigma[ij] / aij) ** 2 + (sigma[kl] / akl) ** 2
                               + (sigma[ik] / aik) ** 2 + (sigma[jl] / ajl) ** 2)

    return {
        'cphase_ind_list': cphase_ind,
        'cphase_sign_list': cphase_sign,
        'camp_ind_list': camp_ind,
        'cphase_data': {'cphase': cphase_vals, 'sigmacp': cphase_sig},
        'camp_data': {'camp': camp_vals},
        'logcamp_data': {'camp': logca_vals, 'sigmaca': logca_sig},
    }
