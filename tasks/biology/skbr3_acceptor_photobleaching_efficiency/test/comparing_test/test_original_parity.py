from pathlib import Path
import numpy as np
import pytest
from src.solvers import reconstruct

ROOT=Path(__file__).resolve().parents[2]

@pytest.fixture(scope='module')
def measured():
    with np.load(ROOT/'data/raw_data.npz',allow_pickle=False) as a:raw=a['observations'][0]
    return reconstruct(raw)

@pytest.mark.parametrize('sample,label',list(enumerate(['donor','acceptor','target'])))
@pytest.mark.parametrize('name,t,c',[('donorBefore',0,0),('acceptorBefore',0,1),('donorAfter',1,0),('acceptorAfter',1,1)])
def test_all_original_preprocessing_stages(measured,sample,label,name,t,c):
    with np.load(ROOT/'test/fixtures/original_stages.npz',allow_pickle=False) as a:
        for stage,key in [('background','background'),('smooth','gaussian')]:
            np.testing.assert_array_equal(measured['stages'][sample][stage][t,c],a[f'{label}_{name}_{key}'])
        smooth=measured['stages'][sample]['smooth'][t,c]
        mask=smooth>100 if label=='donor' and name=='donorAfter' else measured['masks'][sample]
        selected=np.where(mask,smooth,np.float32(np.nan))
        np.testing.assert_array_equal(selected,a[f'{label}_{name}_selected'])
        np.testing.assert_array_equal(measured['masks'][sample],a[f'{label}_input_mask'])

def test_original_complete_field_and_factors(measured):
    with np.load(ROOT/'evaluation/baseline_reference.npz',allow_pickle=False) as a:
        np.testing.assert_array_equal(measured['reconstruction'],a['reconstruction'])
        np.testing.assert_array_equal(measured['correction_factors'],a['correction_factors'])
    assert measured['registration_shift']==(0,0)
    assert measured['masks'][2].sum()==29261
    assert (measured['reconstruction'][measured['masks'][2][None]]<0).sum()==4446

def test_original_and_independently_known_physics():
    from src.physics_model import corrected_efficiency
    with np.load(ROOT/'test/fixtures/original_known_physics.npz',allow_pickle=False) as a:
        raw=a['observations'];truth=a['truth_efficiency'];source=a['original_efficiency']
    local=corrected_efficiency(raw[0],raw[2],raw[1],[.93,.07,.04,.23])
    np.testing.assert_array_equal(local,source)
    assert np.max(abs(local-truth))<4e-7

def test_pipeline_entry_writes_all_fields(tmp_path):
    import shutil
    from src.main import run
    (tmp_path/'data').mkdir();shutil.copyfile(ROOT/'data/raw_data.npz',tmp_path/'data/raw_data.npz')
    result=run(tmp_path)
    with np.load(tmp_path/'output/reconstruction.npz',allow_pickle=False) as a:
        assert set(a.files)=={'reconstruction','correction_factors'}
        np.testing.assert_array_equal(a['reconstruction'],result['reconstruction'])
