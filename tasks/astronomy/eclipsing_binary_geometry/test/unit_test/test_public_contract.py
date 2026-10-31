from pathlib import Path
import ast,numpy as np
from src.physics_model import P,T0,Q

ROOT=Path(__file__).resolve().parents[2]

def test_visible_contract_and_model_constants_agree():
    text=(ROOT/'README.md').read_text(encoding='utf8')
    with np.load(ROOT/'data/raw_data.npz',allow_pickle=False) as f:
        assert set(f.files)=={'times','measurements','reported_uncertainty','group_bounds','response_flags'}
        for key in f.files:
            assert f[key].shape[0]==1
            assert f'| {key} |' in text
        assert f['times'].dtype==f['measurements'].dtype==f['reported_uncertainty'].dtype==np.float64
        assert np.array_equal(np.diff(f['group_bounds'][0],axis=1).ravel(),[7968,3490,239,2199,1564,28])
    for value in [P,T0,Q,.019771142,.019765685498705825,.3532,.2606,.2552,.2598,.1076,.2857]:assert str(value) in text
    constants={n.value for n in ast.walk(ast.parse((ROOT/'src/physics_model.py').read_text(encoding='utf8'))) if isinstance(n,ast.Constant) and isinstance(n.value,(int,float))}
    assert {.019771142,.019765685498705825,695700000.,299792458.,86400.}.issubset(constants)
    assert not any(host in text.lower() for host in ['github.com','github.io','githubusercontent.com','codeload.github.com','git.io'])
