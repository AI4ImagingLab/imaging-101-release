from pathlib import Path
import numpy as np
import pytest
from src.preprocessing import load_raw
from src.solvers import reconstruct
T=Path(__file__).resolve().parents[2]
@pytest.mark.parametrize('case',['key','shape','dtype','nan','index','intensity'])
def test_strict_public_input_rejections(tmp_path,case):
    with np.load(T/'data/raw_data.npz',allow_pickle=False) as f:a={k:f[k] for k in f.files}
    if case=='key':a.pop('links')
    if case=='shape':a['nodes']=a['nodes'][0]
    if case=='dtype':a['nodes']=a['nodes'].astype(np.float32)
    if case=='nan':a['fluorescence'][0,0]=np.nan
    if case=='index':a['links'][0,0,0]=8
    if case=='intensity':a['excitation'][0,0]=0
    p=tmp_path/'raw.npz';np.savez(p,**a)
    with pytest.raises(ValueError):load_raw(p)
@pytest.mark.parametrize('weight',[0.,-1.,float('nan'),float('inf')])
def test_invalid_regularization_rejected_before_computation(weight):
    with pytest.raises(ValueError):reconstruct({},regularization=weight)


def test_readme_acquisition_constants_and_all_input_keys():
    from src import physics_model as model
    from src.preprocessing import SHAPES
    text=(T/'README.md').read_text(encoding='utf8')
    for key in SHAPES:assert '| '+key+' |' in text
    rows={parts[1].strip():parts[2].strip() for line in text.splitlines() if line.startswith('|') and len(parts:=line.split('|'))>3}
    for label,value in [('Excitation absorption',model.MUAX),('Emission absorption',model.MUAM),('Excitation diffusion',model.KAPPAX),('Emission diffusion',model.KAPPAM),('Refractive index',model.RI)]:assert float(rows[label])==value
    assert 'github.com' not in text.lower() and 'githubusercontent' not in text.lower()
