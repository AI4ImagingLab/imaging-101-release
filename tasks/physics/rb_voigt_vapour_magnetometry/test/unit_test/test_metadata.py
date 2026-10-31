from pathlib import Path
import numpy as np
from src.physics_model import Rb85,Rb87,RbD2Transition,muB,h,kB
def test_native_public_input_and_constants():
    root=Path(__file__).resolve().parents[2];readme=(root/'README.md').read_text(encoding='utf8')
    with np.load(root/'data/raw_data.npz',allow_pickle=False) as z:
        assert len(z.files)==6
        for angle,length in zip([0,45,90],[86491,83165,85604]):
            for prefix in ['detuning','transmission']:
                key=f'{prefix}_{angle}';assert z[key].shape==(1,length) and z[key].dtype==np.dtype('float64')
                assert f'| {key} | (1, {length}) | float64 |' in readme
    assert Rb85.I==2.5 and Rb87.I==1.5 and RbD2Transition.NatGamma==6.065
    assert str(muB) in readme and str(h) in readme and str(kB) in readme
    assert '0.001 | m' in readme and '0.7217;0.2783' in readme
