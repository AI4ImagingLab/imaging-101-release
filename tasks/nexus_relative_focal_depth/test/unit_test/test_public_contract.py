from pathlib import Path
import hashlib,re
import numpy as np
from src.preprocessing import load_observations,polynomial_operator

ROOT=Path(__file__).resolve().parents[2]


def test_native_public_observations_and_readme_constants():
    path=ROOT/'data/raw_data.npz'
    assert hashlib.sha256(path.read_bytes()).hexdigest()=='c4adf98cab47fb3d28d145ff5eb7d2e20f3f7986d95e0968c4bc49e06b62b89d'
    images=load_observations(path)
    assert images.shape==(19,1080,1920,3) and images.dtype==np.uint8
    text=(ROOT/'README.md').read_text(encoding='utf8')
    compact=re.sub(r'\s+','',text)
    assert '(1,19,1080,1920,3)' in compact and '(1,1080,1920)' in compact
    assert 'Blue,green,red' in text and 'balcony_0002.jpg' in text and 'balcony_0020.jpg' in text
    X,_=polynomial_operator()
    expected=(np.arange(19,dtype=np.float32)/np.float32(18)-np.float32(.5))*np.float32(20)
    np.testing.assert_array_equal(X[:,-2],expected)
    assert '−10+20k/18' in text and 'zero terminal forward difference' in text


def test_importing_main_keeps_plot_backend():
    import matplotlib,importlib
    before=matplotlib.get_backend()
    importlib.import_module('src.main')
    assert matplotlib.get_backend()==before
