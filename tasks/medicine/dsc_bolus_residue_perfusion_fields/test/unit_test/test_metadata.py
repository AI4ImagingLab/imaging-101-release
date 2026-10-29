from pathlib import Path
import numpy as np
from src.preprocessing import TE,TR
T=Path(__file__).resolve().parents[2]

def test_readme_acquisition_and_public_keys():
    text=(T/'README.md').read_text(encoding='utf8')
    assert f'| Echo time |{TE}|' in text and f'| Sampling interval |{TR}|' in text
    with np.load(T/'data/raw_data.npz',allow_pickle=False) as z:
        assert set(z.files)=={'signal','arterial_input','support'}
        for k in z.files:assert '| '+k+' |' in text and z[k].shape[0]==1
    assert '246.45' in text and '122.45' in text and '4.399999618530273' in text

def test_generator_seed_and_units():
    from src.generate_data import perturb_concentration
    y=np.arange(160,dtype=float).reshape(2,80)
    assert np.array_equal(perturb_concentration(y,0),y)
    assert np.array_equal(perturb_concentration(y,.01,124),perturb_concentration(y,.01,124))
