from pathlib import Path
import numpy as np
from src.physics_model import index,powerlaw,taper
from src.preprocessing import load_observations
from src.solvers import reconstruct
ROOT=Path(__file__).resolve().parents[2]
def stages():return reconstruct(*load_observations(ROOT/'data/raw_data.npz'))
def test_visible_zero_peak_and_energy_sign():
    low,core=load_observations(ROOT/'data/raw_data.npz');s=stages()
    assert np.argmax(low)==204
    assert abs(s['low_axis'][204])<1e-12
    assert abs(s['core_axis'][0]-359.2)<1e-12
    assert s['edge'].shape==(844,)
    np.testing.assert_array_equal(low,np.load(ROOT/'data/raw_data.npz')['low_loss_counts'][0])
def test_full_convolution_equation_independent_time_domain():
    s=stages();n=s['nfft'];L=np.pad(s['low_tapered'],(0,n-len(s['low_tapered'])));V=np.pad(s['core_tapered'],(0,n-len(s['core_tapered'])))
    # Direct time-domain circular convolution; independent of FFT quotient.
    def circular(a,b):return sum(a[j]*np.roll(b,j) for j in range(n))
    lhs=circular(L,s['full_reconstruction']);rhs=s['I0']*circular(s['gaussian'],V)
    assert np.max(abs(lhs-rhs))/np.max(abs(rhs))<1e-11
def test_gaussian_density_and_elastic_scaling():
    s=stages();g=s['gaussian'];assert abs(.2*g.sum()-1)<1e-12
    low,_=load_observations(ROOT/'data/raw_data.npz')
    expected=low[:index(s['threshold'],s['low_axis'][0])].copy()
    n=round(.02*2148);expected[:n]*=np.hanning(2*n)[:n]
    assert abs(expected.sum()-s['I0'])<1e-9
    assert abs(s['I0']-s['low_tapered'].sum())>1000
def test_hann_boundary_and_two_area_known_case():
    y=np.ones(1000);h=taper(y)
    assert h[0]==h[-1]==0 and h[20]==1 and h[-21]==1
    # r=0 count density: two areas equal, A is exactly counts/h.
    x=600+.2*np.arange(200);A,r=powerlaw(np.full(200,7.),x,600,638)
    assert abs(r)<1e-12 and abs(A-35)<1e-10
def test_plot_plural_count_to_density_units():
    from src.visualization import show_results
    import matplotlib.pyplot as plt
    s=stages()
    with np.load(ROOT/'test/fixtures/upstream_stages.npz') as f:
        fig=show_results(s,f)
        np.testing.assert_allclose(fig.axes[0].lines[0].get_ydata(),f['edge']/.2,rtol=1e-12,atol=1e-8)
        assert '0.2 eV' in fig.axes[0].lines[0].get_label()
        assert fig.axes[0].get_ylabel()=='Counts/eV'
        plt.close(fig)
def test_readme_constants_and_public_input_contract():
    text=(ROOT/'README.md').read_text(encoding='utf-8')
    with np.load(ROOT/'data/raw_data.npz',allow_pickle=False) as f:
        assert set(f.files)=={'low_loss_counts','core_loss_counts','low_loss_energy','core_loss_energy'}
        for k in f.files:assert '| '+k+' | (1, 2048) | float64 |' in text
    assert '3000' in text and '768.6' in text and '2.3548' in text
    assert '| Channel spacing | 0.2 | eV |' in text
    assert '| Nominal low-loss origin | -40 | eV |' in text
    assert '| Nominal core-loss origin | 360 | eV |' in text
    assert r'\log U_1-\log U_2' in text
    assert r'L\circledast S=I_0\,G\circledast V' in text
    assert r'\sqrt{2\pi}\sigma' in text
    assert not any(x in text.lower() for x in ['github.com','githubusercontent','baseline_reference.npz','ground_truth.npz'])
