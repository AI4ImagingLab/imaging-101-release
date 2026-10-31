from pathlib import Path
import numpy as np
from src import solvers
T=Path(__file__).resolve().parents[2]
F=T/'test/fixtures'

def test_original_complete_objective_and_two_simplex_updates():
    factory=getattr(solvers,'ring_objective',None)
    optimize=getattr(solvers,'nelder_mead',None)
    assert callable(factory) and callable(optimize),'selected simplex inverse is not implemented'
    with np.load(T/'data/raw_data.npz',allow_pickle=False) as z: cube=z['spectral_cube'][0]
    with np.load(F/'preprocessing.npz',allow_pickle=False) as z: mask=z['output_mask']
    with np.load(F/'simplex_0.npz',allow_pickle=False) as z: initial=z['output_points'][0].astype(np.float32)
    _,stats=__import__('src.preprocessing',fromlist=['search_mask']).search_mask(cube)
    scaled=(cube.astype(float)*(10/float(cube.max()))).astype(np.float32)
    objective=factory(scaled,mask,0,initial,stats['spread']*(10/float(cube.max())))
    trace=[]
    result=optimize(objective,initial,np.array([30,5,10],dtype=np.float32),trace=trace,diagnostic_iterations=2)
    assert not result['converged']
    for i,snapshot in enumerate(trace):
        with np.load(F/f'simplex_{i}.npz',allow_pickle=False) as z:
            np.testing.assert_allclose(snapshot['points'],z['output_points'],rtol=0,atol=5e-5)
            np.testing.assert_allclose(snapshot['values'],z['output_values'],rtol=0,atol=1e-6)
            np.testing.assert_allclose(snapshot['sum'],z['output_sum'],rtol=0,atol=5e-5)
            assert snapshot['evaluations']==int(z['output_evaluations'])
