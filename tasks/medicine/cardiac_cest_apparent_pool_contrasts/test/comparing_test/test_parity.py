from pathlib import Path
import numpy as np
from src.preprocessing import load_data,normalize
from src.solvers import fit_spectrum

ROOT=Path(__file__).resolve().parents[2]

def test_original_normalization():
    image,off,_=load_data(ROOT/'data/raw_data.npz');z,x,d=normalize(image,off)
    with np.load(ROOT/'test/fixtures/normalization.npz') as ref:
        np.testing.assert_array_equal(z,ref['imgs']);np.testing.assert_array_equal(x,ref['offsets'])
        for key in d:np.testing.assert_array_equal(d[key],ref[key])

def test_original_all_three_stages():
    image,off,mask=load_data(ROOT/'data/raw_data.npz');z,x,_=normalize(image,off);positions=np.argwhere(mask)
    for i in [0,len(positions)//2,len(positions)-1]:
        row,col=positions[i];_,_,d=fit_spectrum(z[row,col],x,capture=True)
        with np.load(ROOT/f'test/fixtures/fit_stage_{i}.npz') as ref:
            for j,s in enumerate(d['stages']):
                for key,v in s.items():np.testing.assert_allclose(v,ref[f'stage_{j}_{key}'],rtol=1e-7,atol=1e-8)

def test_fresh_full_pipeline(tmp_path):
    from src.main import run
    from evaluation.judge import evaluate
    target=tmp_path/'reconstruction.npz';run(ROOT/'data/raw_data.npz',target)
    report,code=evaluate(target);assert code==0,report
