"""Original fixed-state scalar arithmetic, independently captured in Octave."""
from pathlib import Path
import numpy as np
from src.linear_algebra import SourceBlas
from src.solvers import _cubic_step


FIXTURE = Path(__file__).resolve().parents[1] / 'fixtures/upstream_scalar_roles.npz'


def test_source_cubic_operation_order():
    with np.load(FIXTURE, allow_pickle=False) as source:
        actual = _cubic_step(*(float(source[key]) for key in
            ('f', 'gtd', 'step', 'f_new', 'derivative_new')))
        assert actual == float(source['source_cubic'])


def test_source_dot_and_self_product_roles():
    with np.load(FIXTURE, allow_pickle=False) as source, SourceBlas() as backend:
        s, y = source['s'], source['y']
        ss, sy = backend.dot(s, s), backend.dot(s, y)
        assert ss == float(source['source_ss'])
        assert sy == float(source['source_sy'])
        assert ss / sy == float(source['source_alpha'])
    assert backend.process.returncode == 0
