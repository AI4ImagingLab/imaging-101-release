import numpy as np
import pytest
from src.physics_model import real_harmonics,detector_coefficients,angular_coupling,transfer_fields,frequency_matrix
from src.preprocessing import subset_indices,load_data
from src.solvers import tikhonov_inverse,isotropic_null_residuals
from src.visualization import coefficient_metrics,angular_samples,orientation_diagnostics

def test_known_isotropic_harmonic():
    y=real_harmonics(np.array([0.,np.pi/2]),np.array([0.,1.]))
    np.testing.assert_allclose(y[:,0],1/np.sqrt(4*np.pi),rtol=0,atol=1e-15)

def test_illumination_equals_known_dipole():
    t=np.linspace(0,np.pi,41);p=np.linspace(0,2*np.pi,41)
    y=real_harmonics(t,p,lmax=2);c=real_harmonics(0.,0.,lmax=2);c[1:]*=.4
    np.testing.assert_allclose(y@c,3/(4*np.pi)*np.cos(t)**2,atol=1e-15)

def test_aperture_support_and_zero_frequency():
    c=detector_coefficients(np.array([0.,2.,3.]),1.1,0)
    assert c[0,0]>0
    np.testing.assert_array_equal(c[1:],0.)

def test_angular_coupling_is_orthogonal_for_isotropic_detector():
    np.testing.assert_allclose(angular_coupling()[:6,:,0],np.eye(6)/np.sqrt(4*np.pi),atol=2e-15)

@pytest.mark.parametrize('n',[5,6,55,77,80])
def test_signed_frequency_coverage(n):
    k=np.abs(np.rint(np.fft.fftfreq(n)*n)).astype(int)
    assert k.max()==n//2
    for j in range(1,(n-1)//2+1):assert np.count_nonzero(k==j)==2

def test_highest_odd_mode_requires_both_partners():
    n=55;x=np.cos(2*np.pi*27*np.arange(n)/n);F=np.fft.fft(x);bad=F.copy();bad[28]=0
    assert abs(np.vdot(np.fft.ifft(bad).real,x)/np.vdot(x,x)-.5)<1e-14
    np.testing.assert_allclose(np.fft.ifft(F).real,x,atol=1e-14)

def test_forward_factors_even_under_signed_x_y():
    pol=np.array([[[0,1,0]],[[0,1,0]]]);h=transfer_fields(pol,(5,7,4));H=frequency_matrix(h,0,(5,7,4))
    np.testing.assert_array_equal(H[1,1],H[-1,-1])

def test_invalid_group_and_data(tmp_path):
    with pytest.raises(ValueError):subset_indices('invented')
    path=tmp_path/'x.npz';np.savez(path,unexpected=np.ones(1))
    with pytest.raises(ValueError):load_data(path)

@pytest.mark.parametrize('eta',[-1,float('nan'),float('inf')])
def test_invalid_regularization(eta):
    with pytest.raises(ValueError):tikhonov_inverse(np.zeros((1,1,2,1,2)),{},eta)

def test_known_metrics_without_mean_subtraction():
    r=np.tile(np.arange(1,5.)[:,None],(1,15));x=r*2
    row=coefficient_metrics(x,r)[0]
    assert abs(row['ncc']-1)<1e-15
    assert abs(row['nrmse']-np.sqrt(7.5)/3)<1e-15
    assert row['max_abs']==4

def test_isotropic_orientation_diagnostic():
    f=np.zeros((2,3,15));f[...,0]=1
    d=orientation_diagnostics(f)
    np.testing.assert_array_equal(d['gfa'],0)
    np.testing.assert_array_equal(d['negative_fraction'],0)
    assert d['angular_minimum'].min()>0

def test_display_directions_are_unit_vectors():
    p,_=angular_samples(128);np.testing.assert_allclose(np.linalg.norm(p,axis=1),1,atol=3e-16)

def test_readme_metadata_and_no_public_source_leaks():
    from pathlib import Path
    root=Path(__file__).resolve().parents[2];text=(root/'README.md').read_text(encoding='utf8')
    for section in ['Background','Problem Description','Data Description','Method Hints','References']:assert '## '+section in text
    for item in ['561','130,130,130','1.1,0.67','1.33','0.25','+8.9,−9.2','+7.7,−6.5','(1,77,55,80,21,2)','(1,30,30,30,21,2)']:assert item in text
    for url in ['github.com','githubusercontent.com','github.io','git.io','codeload.github']:assert url not in text.lower()

def test_known_isotropic_nulls_and_rfft_weighting():
    h={'Hxy':np.zeros((2,2,15,1),np.float32),'Hyz':np.zeros((2,3,15,1),np.float32),'Hx':np.ones(2),'Hz':np.ones(3,np.float32)}
    h['Hxy'][...,0,:]=1;h['Hyz'][...,0,:]=1
    r=isotropic_null_residuals(np.full((3,3,4,1,2),2.),h)
    assert r['uniform_isotropic']==0 and r['spatial_isotropic_unregularized']<1e-15
    assert abs(r['spatial_isotropic_regularized']-.7/2.7)<1e-15
