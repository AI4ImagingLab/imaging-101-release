from pathlib import Path
import numpy as np
from src.preprocessing import load_observations,parameters
from src import solvers,physics_model
ROOT=Path(__file__).resolve().parents[2]
def test_independently_captured_original_stages(monkeypatch):
    fixture=np.load(ROOT/'test/fixtures/upstream_stages.npz',allow_pickle=False)
    observed=load_observations(ROOT/'data/raw_data.npz');record={};calls=[]
    original=solvers.QPBO
    def capture(D,V,ngb):
        index=len(calls);labels=original(D,V,ngb);calls.append(index)
        if index in [0,2,4,6]:
            for key,value in [('input_unary',D),('input_pairwise',V),('input_neighbours',ngb),('output_labels',labels)]:
                expected=fixture[f'cut{index}_{key}']
                if key in ['input_neighbours','output_labels']:np.testing.assert_array_equal(value,expected)
                else:np.testing.assert_allclose(value,expected,rtol=1e-6,atol=1e-12)
        return labels
    monkeypatch.setattr(solvers,'QPBO',capture)
    for name in ['pseudoinverse_and_projection_matrices','_B0_residuals','_R2_residuals']:
        base=getattr(physics_model,name)
        def wrapper(*args,_base=base,_name=name,**kwargs):
            value=_base(*args,**kwargs)
            if _name not in record:
                for i,v in enumerate(value if isinstance(value,tuple) else (value,)):np.testing.assert_allclose(v,fixture[_name+'_output'+str(i)],rtol=1e-6,atol=1e-12)
                record[_name]=True
            return value
        monkeypatch.setattr(physics_model,name,wrapper)
        if hasattr(solvers,name):monkeypatch.setattr(solvers,name,wrapper)
    before=observed['echoes'].copy();fresh=solvers.reconstruct(observed)
    np.testing.assert_array_equal(before,observed['echoes'])
    assert len(calls)==7 and len(record)==3
    with np.load(ROOT/'evaluation/baseline_reference.npz',allow_pickle=False) as reference:
        for key in fresh:np.testing.assert_array_equal(fresh[key],reference[key])
