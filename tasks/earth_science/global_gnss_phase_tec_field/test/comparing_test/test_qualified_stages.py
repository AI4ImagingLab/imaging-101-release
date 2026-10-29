from pathlib import Path
import numpy as np
from src.physics_model import design_matrix, slant_factor, real_harmonics
from src.solvers import LemkeTableau, lemke

FIX = Path(__file__).resolve().parents[1] / 'fixtures'

def test_qualified_forward_stages():
    with np.load(FIX/'qualified_stages.npz', allow_pickle=False) as z:
        d = {k[6:]: z[k] for k in z.files if k.startswith('input_')}
        np.testing.assert_allclose(slant_factor(d['el']), z['SF'], rtol=0, atol=0)
        np.testing.assert_allclose(real_harmonics(d['mlt1'],d['mcolat1'])*z['SF'][:,None],z['ac'],rtol=0,atol=3e-13)
        A,w = design_matrix(d)
        np.testing.assert_allclose(A.toarray(),z['A'],rtol=0,atol=3e-13)
        np.testing.assert_allclose(w,z['diagP'],rtol=1e-15,atol=1e-12)
        np.testing.assert_allclose(A.T@(w*d['rhs']),z['b'],rtol=2e-12,atol=3e-10)
        np.testing.assert_allclose((A.T@A.multiply(w[:,None])).toarray()[:12,:12],z['normal_probe'],rtol=2e-12,atol=3e-10)

def test_lemke_original_pivot_stages():
    with np.load(FIX/'lemke_stages.npz',allow_pickle=False) as z:
        t=LemkeTableau(z['M'],z['q'])
        np.testing.assert_array_equal(t.table,z['initial'])
        t.initialize()
        np.testing.assert_allclose(t.table,z['tableaux'][0],atol=2e-14,rtol=0)
        for j in range(1,len(z['tableaux'])):
            t.step()
            np.testing.assert_allclose(t.table,z['tableaux'][j],atol=2e-14,rtol=0)
            np.testing.assert_array_equal(t.labels,z['variable_indices'][j])
        np.testing.assert_allclose(lemke(z['M'],z['q']),z['z'],atol=2e-14,rtol=0)
