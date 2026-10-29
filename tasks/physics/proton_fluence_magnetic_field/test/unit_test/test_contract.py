from pathlib import Path
import re
import numpy as np
from src import physics_model as p
from src.solvers import linear_poisson
from src.preprocessing import load_observations

ROOT=Path(__file__).resolve().parents[2]

def test_readme_acquisition_constants_and_visible_contract():
    text=(ROOT/'README.md').read_text(encoding='utf-8')
    expected={'Proton speed v':p.V_CMS,'Source-to-volume entrance ri':p.RI_CM,
              'Volume extent li':p.LI_CM,'Volume-to-detector distance rs':p.RS_CM,
              'Proton mass mp':p.M_PROTON_G,'Proton charge q':p.ESU,'Light speed c':p.C_CMS}
    for name,value in expected.items():
        match=re.search(r'^\| '+re.escape(name)+r' \| ([^|]+)\|',text,re.M)
        assert match and float(match.group(1).strip())==value
    assert 'Negative laboratory Z' in text
    assert not re.search(r'github\.com|githubusercontent\.com|github\.io|git\.io',text,re.I)
    with np.load(ROOT/'data/raw_data.npz',allow_pickle=False) as d:
        assert set(d.files)=={'x','y','fluence'}
        for key in d.files:
            assert re.search(r'^\| '+key+r' \| \(1, 200, 200\) \| float64 \|',text,re.M)
            assert d[key].shape==(1,200,200) and d[key].dtype==np.float64

def test_half_pixel_grid_order_and_units():
    d=load_observations()
    assert np.all(d['x']==d['x'][:,0,None])
    assert np.all(d['y']==d['y'][0,None,:])
    np.testing.assert_allclose([d['x'][0,0],d['x'][-1,0]],[-.0995,.0995],atol=0,rtol=1e-15)
    np.testing.assert_allclose(np.diff(d['x'][:,0]),.001,atol=2e-17,rtol=0)
    assert np.sum(d['fluence'])==29354394.

def test_poisson_dct_eigenmode_factor_and_sign():
    nx,ny=18,15;dx=.002;dy=.003
    xv=(np.arange(nx)+.5)*dx;yv=(np.arange(ny)+.5)*dy
    x,y=np.meshgrid(xv,yv,indexing='ij')
    mode=np.cos(np.pi*(np.arange(nx)+.5)/nx)[:,None]*np.cos(2*np.pi*(np.arange(ny)+.5)/ny)[None,:]
    epsilon=.01
    f=700*(1-epsilon*mode)
    eigenvalue=-4*np.sin(np.pi/(2*nx))**2/dx**2-4*np.sin(2*np.pi/(2*ny))**2/dy**2
    u=epsilon*mode/eigenvalue
    ux,uy=np.gradient(u,dx,dy);ux[[0,-1],:]=0;uy[:,[0,-1]]=0
    multiplier=p.M_PROTON_G*p.C_CMS/p.ESU*p.MAGNIFICATION*p.V_CMS/p.RS_CM
    expected=np.stack([-multiplier*uy,multiplier*ux])
    np.testing.assert_allclose(linear_poisson(x,y,f),expected,rtol=1e-10,atol=2e-10)

def test_uniform_zero_field_ray_count_conservation():
    d=load_observations();f=p.ray_forward_fluence(d['x'],d['y'],np.zeros((2,200,200)),317.,subpixels=2)
    np.testing.assert_array_equal(f,np.full((200,200),317.))
