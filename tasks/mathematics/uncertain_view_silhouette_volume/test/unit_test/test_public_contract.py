from pathlib import Path
import re
import numpy as np,pytest
from src.preprocessing import load_observations
T=Path(__file__).resolve().parents[2]

def test_public_allowlist_and_array_tables():
    text=(T/'README.md').read_text(encoding='utf-8')
    assert not re.search('github|git\\.io',text,re.I)
    with np.load(T/'data/raw_data.npz',allow_pickle=False) as z:
        assert set(z.files)=={'silhouettes','view_prior'}
        for k in z.files:assert re.search(r'\| '+k+r' \|',text)
    images,prior=load_observations(T/'data/raw_data.npz');assert images.shape==(128,128,8) and prior.shape==(8,3)
    assert '0.5' in text and '|100 |' in text and '|8 |' in text

@pytest.mark.parametrize('bad',['shape','dtype','nonfinite','range'])
def test_preprocessing_rejects_malformed_observation(tmp_path,bad):
    with np.load(T/'data/raw_data.npz') as z:x={k:z[k].copy() for k in z.files}
    if bad=='shape':x['view_prior']=x['view_prior'][0]
    elif bad=='dtype':x['silhouettes']=x['silhouettes'].astype(float)
    elif bad=='nonfinite':x['view_prior'].flat[0]=np.nan
    else:x['silhouettes'].flat[0]=2
    p=tmp_path/'raw.npz';np.savez_compressed(p,**x)
    with pytest.raises(ValueError):load_observations(p)
