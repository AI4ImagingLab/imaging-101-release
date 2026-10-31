from pathlib import Path
import numpy as np
from src.physics_model import spectrum
T=Path(__file__).resolve().parents[2]
def test_original_spectrum():
    z=np.load(T/'test/fixtures/original_stages.npz',allow_pickle=False)
    for i in range(6):
        # Captured x is already sorted sin(elevation)/half-wavelength.
        prefix=f'strip_compute_{i}_';x=z[prefix+'x'];y=z[prefix+'y'];cf=z[prefix+'cf']
        h,p=spectrum(x,y)
        np.testing.assert_allclose(h,z[prefix+'px'],atol=1e-12)
        np.testing.assert_allclose(p,z[prefix+'pz'],rtol=1e-8,atol=1e-9)
def test_known_harmonic():
    x=np.linspace(.1,3.5,200)**1.1;y=3*np.cos(2*np.pi*2.5*x)+2*np.sin(2*np.pi*2.5*x)
    h,p=spectrum(x,y);assert abs(h[np.argmax(p)]-2.5)<.01
def test_full_bound_data():
    from src.solvers import reconstruct
    with np.load(T/'data/raw_data.npz',allow_pickle=False) as z:data={k:z[k][0] for k in z.files}
    result=reconstruct(data)
    with np.load(T/'evaluation/baseline_reference.npz',allow_pickle=False) as z:np.testing.assert_allclose(result,z['reconstruction'],atol=1e-7,rtol=0)
