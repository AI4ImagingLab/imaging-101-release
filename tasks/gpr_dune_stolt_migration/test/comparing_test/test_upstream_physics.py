from pathlib import Path
import numpy as np
from src.preprocessing import load_data,prepare,taper
from src.physics_model import remap,remap_adjoint,dispersion
from src.solvers import reconstruct
ROOT=Path(__file__).resolve().parents[2]

def test_full_original_upstream_stage_fixtures():
    f=np.load(ROOT/'test/fixtures/upstream.npz',allow_pickle=False)
    raw,time,x=load_data(ROOT/'data/raw_data.npz')
    np.testing.assert_array_equal(raw,f['input_radar'])
    np.testing.assert_array_equal(time,f['input_time_ns'])
    np.testing.assert_array_equal(x,f['input_position_m'])
    shifted,dew,pre=prepare(raw,time)
    for actual,key in [(shifted,'output_shifted'),(dew,'output_dewow'),(pre,'output_preprocessed')]:np.testing.assert_allclose(actual,f[key],rtol=1e-10,atol=1e-8)
    image,diagnostics=reconstruct(raw,time,return_diagnostics=True)
    np.testing.assert_allclose(image,f['output_image'],rtol=1e-10,atol=1e-8)
    np.testing.assert_allclose(diagnostics['fk'],f['output_fk'],rtol=1e-10,atol=1e-5)
    np.testing.assert_allclose(diagnostics['kk'],f['output_kk'],rtol=1e-10,atol=1e-5)
    np.testing.assert_allclose(diagnostics['scaling'],f['output_scaling'],rtol=1e-13,atol=1e-14,equal_nan=True)
    for velocity,key in [(1.2e8,'output_low_velocity'),(1.6e8,'output_high_velocity')]:np.testing.assert_allclose(reconstruct(raw,time,velocity),f[key],rtol=1e-10,atol=1e-8)

def test_dispersion_and_linear_interpolation_adjoint():
    n=32;nx=18;dt=1e-9;dx=.02;v=1.4e8
    omega,kx,kz,target,scale=dispersion(n,nx,dt,dx,v)
    np.testing.assert_allclose((2*target/v)**2,kz[:,None]**2+kx[None,:]**2,rtol=1e-14,atol=1e-12)
    rng=np.random.default_rng(4);a=rng.normal(size=target.shape)+1j*rng.normal(size=target.shape);b=rng.normal(size=target.shape)+1j*rng.normal(size=target.shape)
    mapped=remap(a,omega,target)
    adjoint=remap_adjoint(b,omega,target)
    np.testing.assert_allclose(np.vdot(mapped,b),np.vdot(a,adjoint),rtol=1e-13,atol=1e-12)

def test_vertical_incidence_harmonic_and_real_linearity():
    # No taper: a horizontally invariant sinusoid is already at its correct
    # vertical position, apart from explicitly removed DC/last Nyquist bin.
    n=64;nx=20;t=np.arange(n);wave=np.cos(2*np.pi*5*t/n)
    data=np.broadcast_to(wave[:,None],(n,nx)).copy()
    from src.solvers import migrate
    migrated,_=migrate(data,1e-9,.02,1.4e8,horizontal_taper=1,vertical_taper=1,apply_taper=False)
    np.testing.assert_allclose(migrated,data,atol=1e-14)
    np.testing.assert_allclose(migrate(2*data,1e-9,.02,1.4e8,apply_taper=False)[0],2*migrated,atol=1e-13)

def test_preprocessing_background_and_calibration():
    raw,time,x=load_data(ROOT/'data/raw_data.npz')
    _,_,pre=prepare(raw,time)
    np.testing.assert_allclose(pre.mean(axis=0),0,atol=1e-11)
    np.testing.assert_allclose(pre.mean(axis=1),0,atol=1e-11)
    assert pre.shape==(458,1040)
    np.testing.assert_array_equal(time,np.linspace(0,48.,512))
    np.testing.assert_array_equal(x,np.arange(1040)/50.)
    zero=reconstruct(np.zeros_like(raw),time);np.testing.assert_array_equal(zero,0)

def test_independent_point_diffraction_focus():
    from src.physics_model import point_diffraction
    from src.solvers import migrate
    n=512;nx=128;dt=.1e-9;dx=.02;v=1.4e8
    x=(np.arange(nx)-nx//2)*dx;t=np.arange(n)*dt;depth=1.0
    data=point_diffraction(x,t,0.,depth,v,.3e-9)
    # Travel-time apex and migration must both respect two-way propagation.
    apex=np.argmax(data[:,nx//2])
    assert abs(t[apex]-2*depth/v)<=dt
    image,_=migrate(data,dt,dx,v,horizontal_taper=12,vertical_taper=12)
    zi,xi=np.unravel_index(np.argmax(abs(image)),image.shape)
    assert abs(x[xi])<=dx
    assert abs(zi*dt*v/2-depth)<=3*dt*v/2

def test_independent_survey_and_source_terrain_display():
    from src.preprocessing import load_topography
    from src.visualization import terrain_display
    f=np.load(ROOT/'test/fixtures/upstream.npz',allow_pickle=False)
    station,elevation=load_topography(ROOT/'data/raw_data.npz')
    np.testing.assert_array_equal(station,f['input_topo_position_m'])
    np.testing.assert_array_equal(elevation,f['input_surface_elevation_m'])
    display,highest,lowest=terrain_display(f['output_image'],f['input_position_m'],station,elevation,(48/511)*.14/2)
    np.testing.assert_array_equal(display,f['output_terrain'])
    assert abs(highest-f['output_max_elevation'])<1e-12 and abs(lowest-f['output_min_elevation'])<1e-12
