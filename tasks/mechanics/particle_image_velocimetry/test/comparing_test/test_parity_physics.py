from pathlib import Path
import numpy as np
from src.preprocessing import load_frames,windows,prepare,normalize
from src.physics_model import correlate
from src.solvers import solve,subpixel_peak

def test_frozen_independent_upstream_stages():
    a,b=load_frames('data/raw_data.npz')
    with np.load('test/fixtures/upstream_stages.npz',allow_pickle=False) as ref:
        np.testing.assert_array_equal(windows(a),ref['windows_a'])
        aa,bb=prepare(a,b)
        np.testing.assert_allclose(aa,ref['normalized_masked_a'],rtol=0,atol=1e-10)
        np.testing.assert_allclose(bb,ref['normalized_b'],rtol=0,atol=1e-10)
        c=correlate(aa,bb)
        for ci in c:subpixel_peak(ci)
        np.testing.assert_allclose(c,ref['correlation'],rtol=0,atol=1e-10)
        v,s=solve(a,b)
        np.testing.assert_allclose(v,ref['velocity'],rtol=0,atol=1e-8)
        np.testing.assert_allclose(s,ref['confidence'],rtol=0,atol=1e-8)

def test_fft_against_direct_sum_independent_operator():
    rng=np.random.default_rng(913)
    a=rng.normal(size=(1,8,8));b=rng.normal(size=(1,8,8))
    aa,bb=normalize(a)[0],normalize(b)[0]
    direct=np.empty((8,8))
    for i in range(8):
        for j in range(8):
            direct[i,j]=sum(aa[y,x]*bb[(y+i-4)%8,(x+j-4)%8] for y in range(8) for x in range(8))/64
    np.testing.assert_allclose(correlate(a,b)[0],direct,rtol=0,atol=2e-15)

def test_identifiable_translation_sign_and_time_units():
    rng=np.random.default_rng(19)
    # Exactly periodic texture matches the declared circular boundary; no
    # finite-window loss-of-pairs bias is mixed into this units/sign check.
    tile=rng.normal(size=(64,64));a=np.tile(tile,(2,2));b=np.roll(a,(4,-3),axis=(0,1))
    v,s=solve(a,b,window=64,search=64,overlap=0)
    np.testing.assert_allclose(v[0,...,0],-3,atol=1e-12)
    np.testing.assert_allclose(v[0,...,1],4,atol=1e-12)
    v2,s2=solve(a,b,window=64,search=64,overlap=0,dt=2)
    np.testing.assert_array_equal(v2,v/2)
    np.testing.assert_array_equal(s,s2)

def test_cli_pipeline():
    from src.main import main
    main()
    with np.load('output/reconstruction.npz',allow_pickle=False) as data:
        assert data['reconstruction'].shape==(1,7,10,2)
        assert np.isfinite(data['confidence']).all()
