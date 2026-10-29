import numpy as np,pytest
from src.physics_model import shell_grid,shell_mean,forward_scattering,reciprocal_projection,center_integer,update_support,keep_connected

def test_hermitian_average_equals_independent_full_fft_and_axis_rotations():
    n=64;dx=3.75;rho=np.random.RandomState(0).random_sample((n,n,n));grid=shell_grid()
    q,profile=forward_scattering(rho,grid)
    axis=np.fft.fftfreq(n)*2*np.pi/dx;radius=np.sqrt(sum(a*a for a in np.meshgrid(axis,axis,axis,indexing='ij')));step=radius[radius>0].min()-1e-8;bins=np.linspace(0,int(radius.max()/step)*step,int(radius.max()/step)+1);labels=np.searchsorted(bins,radius,'right')-1;count=np.bincount(labels.ravel())
    full=np.bincount(labels.ravel(),weights=np.abs(np.fft.fftn(rho*dx**3)).ravel()**2)/count
    np.testing.assert_allclose(profile,full,rtol=1e-12,atol=1e-10)
    for axes in [(0,1),(0,2),(1,2)]:np.testing.assert_allclose(forward_scattering(np.rot90(rho,axes=axes),grid)[1],profile,rtol=1e-12,atol=1e-10)

def test_single_voxel_scattering_and_translation():
    rho=np.zeros((64,64,64));rho[32,32,32]=1/3.75**3
    np.testing.assert_allclose(forward_scattering(rho)[1],1.,rtol=0,atol=1e-14)
    np.testing.assert_allclose(forward_scattering(np.roll(rho,17,axis=2))[1],1.,rtol=0,atol=1e-14)

def test_projection_fits_every_shell_without_resetting_phase():
    grid=shell_grid();rng=np.random.RandomState(1);rho=rng.random_sample((64,64,64));target=np.exp(-grid['q']**2*100);selected=np.ones(len(target),dtype=bool)
    projected,_=reciprocal_projection(rho,target,selected,grid)
    np.testing.assert_allclose(shell_mean(np.abs(np.fft.rfftn(projected))**2,grid),target,rtol=1e-8,atol=1e-14)

def test_integer_center_and_support_are_not_hidden_alignment():
    rho=np.zeros((64,64,64));rho[5:8,15:18,20:23]=1;centered=center_integer(rho)
    assert np.all(np.array(np.unravel_index(centered.argmax(),centered.shape))>=30)
    np.testing.assert_allclose(forward_scattering(centered)[1],forward_scattering(rho)[1],rtol=1e-12,atol=1e-7)
    centered,support=update_support(rho,1.5);assert support.dtype==bool and support.any() and not support.all()

def test_connectivity_retains_largest_mass_component():
    rho=np.zeros((7,7,7));rho[1,1,1]=1;rho[5,5,5]=3;out,support=keep_connected(rho,rho>0)
    assert out.sum()==3 and support.sum()==1 and out[5,5,5]==3

def test_readme_metadata_constants():
    from pathlib import Path
    text=(Path(__file__).resolve().parents[2]/'README.md').read_text(encoding='utf8')
    for value in ['240.0','3.75','40.0','0.2793','(1, 64, 64, 64)']:assert value in text
