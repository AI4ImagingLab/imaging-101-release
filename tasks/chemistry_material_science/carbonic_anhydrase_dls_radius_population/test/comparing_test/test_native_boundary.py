from pathlib import Path
import numpy as np
from src.preprocessing import load_data,transform
ROOT=Path(__file__).resolve().parents[2]
def test_native_endpoint_and_exact_one():
 t,g,r,us=load_data(ROOT/'data/raw_data.npz');assert np.any(us==5.0);assert np.count_nonzero(us<5)==np.count_nonzero(t<5e-6)
 beta,field,valid=transform(t,g,us);expected=np.exp(np.polyfit(t[us<5],np.log(g[us<5]-1),2)[-1]);np.testing.assert_array_equal(beta,expected)
 altered=g.copy();altered[100,0]=1.;altered[101,0]=.99;assert transform(t,altered,us)[2][0]==101
