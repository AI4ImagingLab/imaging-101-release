"""Network, closure and temporal models extracted from MintPy conventions.

Copyright (c) 2013-2026 Zhang Yunjun, Heresh Fattahi and MintPy contributors.
Adapted under GPL-3.0-or-later; attribution and license text: test/validation.md.
"""
import datetime as dt
import itertools
import numpy as np


def network_matrices(dates, pairs):
    """Return float32 incidence A, integrated-velocity B, triangles C and dt/year."""
    calendar = [dt.datetime.strptime(str(int(x)), '%Y%m%d') for x in dates]
    years = np.array([(x - calendar[0]).days for x in calendar], np.float32) / 365.25
    intervals = np.diff(years).reshape(-1, 1)
    a = np.zeros((len(pairs), len(dates)), np.float32)
    b = np.zeros_like(a)
    edges = {}
    for i, (start, end) in enumerate(pairs):
        a[i, start] = -1
        a[i, end] = 1
        if start < end:
            b[i, start:end] = np.diff(years)[start:end]
        else:
            b[i, end:start] = -np.diff(years)[end:start]
        edges[(int(start), int(end))] = i
    triangles = []
    for i, j, k in itertools.combinations(range(len(dates)), 3):
        if (i, j) in edges and (j, k) in edges and (i, k) in edges:
            row = np.zeros(len(pairs), np.float32)
            row[edges[i, j]] = row[edges[j, k]] = 1
            row[edges[i, k]] = -1
            triangles.append(row)
    c = np.stack(triangles) if triangles else np.zeros((0, len(pairs)), np.float32)
    return a[:, 1:], b[:, :-1], c, intervals


def closure_integers(c, phase):
    """Integer closure cycles using wrap to [-pi, pi), preserving input precision."""
    closure = c @ phase
    wrapped = (closure + np.pi) % (2 * np.pi) - np.pi
    return np.round((closure - wrapped) / (2 * np.pi))


def temporal_model(dates, steps=(20170910, 20180613)):
    """Columns: offset, velocity, acceleration/2, then strict post-event steps."""
    calendar = [dt.datetime.strptime(str(int(x)), '%Y%m%d') for x in dates]
    # This calendar-year/365.25 convention deliberately matches the selected upstream.
    years = np.array([x.year + (x.timetuple().tm_yday - 1) / 365.25 for x in calendar])
    time = years - years[0]
    g = np.zeros((len(dates), 3 + len(steps)), np.float32)
    g[:, 0], g[:, 1], g[:, 2] = 1, time, time ** 2 / 2
    for i, step in enumerate(steps):
        g[:, 3+i] = np.asarray(dates) > step
    return g


def dem_design(baseline, incidence, dates, starting_range=799547.2189154713,
               range_spacing=34.94343172072985):
    """Scene-mean geometry column plus the five-column deformation model."""
    distance = np.mean(starting_range + np.arange(incidence.shape[1]) * range_spacing)
    sine = np.mean(np.sin(np.deg2rad(incidence)))
    return np.column_stack((baseline / (distance * sine), temporal_model(dates)))
