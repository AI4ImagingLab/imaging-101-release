from pathlib import Path
import importlib.util,json,subprocess,sys
import numpy as np
import pytest
from src.preprocessing import bracket_grid,load_observations
from src.physics_model import shell_paths,line_of_sight_tau,integrate_radii,K_B,G,AMU,TransmissionModel
from src.solvers import log_prior,GUESS,GAUSSIAN_STD
T=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('independent_judge',T/'evaluation/judge.py');judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)

def test_constant_extinction_chord_analytic():
    radii=np.array([3.,2.,1.]);alpha=np.full((3,2),.7)
    expected=2*.7*np.sqrt(9-np.array([2.,1.])**2)
    np.testing.assert_allclose(line_of_sight_tau(alpha,radii),np.tile(expected,(2,1)),atol=1e-14)

def test_constant_hydrostatics_and_prior():
    p=np.log(np.array([1e5,1e4,1e3]));mass=1e27;radius=1e8;temp=1000.;mu=2.
    constant=lambda x:np.full_like(x,temp);mean=lambda x:np.full_like(x,mu)
    result=integrate_radii(p,mass,radius,constant,mean)
    np.testing.assert_allclose(result,1/(1/radius+(p-p[0])*K_B*temp/(G*mass*mu*AMU)),rtol=1e-14)
    trial=GUESS.copy();trial[4]=.5
    expected=-np.log(np.sqrt(2*np.pi)*GAUSSIAN_STD).sum()
    np.testing.assert_allclose(log_prior(trial),expected,atol=1e-14)
    trial[4]=0;assert log_prior(trial)==-np.inf

def test_bracket_and_transparent_limit():
    np.testing.assert_array_equal(bracket_grid([2,3],np.array([1,2,3,4])),[False,True,True,False])
    model=TransmissionModel(load_observations(T/'data/raw_data.npz'))
    with np.load(T/'evaluation/baseline_reference.npz',allow_pickle=False) as z:p=z['best_fit'][0]
    d,info=model.forward(p,full_output=True,gas=False,rayleigh=False,collision=False)
    np.testing.assert_allclose(info['tau_los'],0);np.testing.assert_allclose(d,(info['radii'].min()/p[0])**2,atol=1e-16)

def test_independent_rayleigh_and_collision_scalings():
    # Small independently specified physical calibration, not benchmark truth.
    raw={'wavelength_grid':np.array([1e-6,2e-6]),'pressure_grid':np.array([1.,2.]),'temperature_grid':np.array([1000.,2000.]),'absorber_species':np.array([],dtype=int),'wavelength_bins':np.array([[.5e-6,3e-6]]),'species_polarizability':np.array([1e-30,2e-30]),'collision_species':np.array([[0,1]]),'collision_coefficients':np.ones((1,2,2))*1e-47}
    model=TransmissionModel(raw);abundance=np.stack([np.full((2,2),.7),np.full((2,2),.3)]);mask=np.ones(2,bool)
    _,r=model.absorption(abundance,mask,mask,2.,gas=False)
    np.testing.assert_allclose(r['rayleigh'][...,0]/r['rayleigh'][...,1],16.,rtol=1e-14)
    np.testing.assert_allclose(r['rayleigh'][:,1]/r['rayleigh'][:,0],2.,rtol=1e-14)
    np.testing.assert_allclose(r['collision'][:,1]/r['collision'][:,0],4.,rtol=1e-14)
    density=1/(K_B*1000.)
    expected=2*(128*np.pi**5/3)*density*(.7*1e-60+.3*4e-60)/(1e-6)**4
    np.testing.assert_allclose(r['rayleigh'][0,0,0],expected,rtol=1e-14)

def test_independent_judge_prior_normalization():
    p=GUESS.copy();p[4]=.5
    np.testing.assert_allclose(judge.prior(p),-np.log(np.sqrt(2*np.pi)*GAUSSIAN_STD).sum(),atol=1e-14)

@pytest.fixture
def good():
    with np.load(T/'evaluation/baseline_reference.npz',allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}

@pytest.mark.parametrize('kind',['missing','shape','dtype','complex','nan','inf','bounds'])
def test_malformed(tmp_path,good,kind):
    if kind=='missing':del good['best_fit']
    elif kind=='shape':good['best_fit']=good['best_fit'][0]
    elif kind=='dtype':good['best_fit']=good['best_fit'].astype(np.float32)
    elif kind=='complex':good['best_fit']=good['best_fit'].astype(np.complex128)
    elif kind in ['nan','inf']:good['best_fit'][0,0]=float(kind)
    else:good['best_fit'][0,7]=-1
    p=tmp_path/'bad.npz';np.savez(p,**good);r,c=judge.evaluate(p);assert c==2 and r['status']=='error' and r['metrics']=={} and r['errors']

def test_poor_valid_and_all_output_quantities(tmp_path,good):
    good['transit_depth']+=.001;p=tmp_path/'poor.npz';np.savez(p,**good);r,c=judge.evaluate(p);assert c==1 and r['metrics']['spectrum_consistency_ppm']>1
    good['transit_depth']-=.001;good['reconstruction'][0,3]=2000.;np.savez(p,**good);r,c=judge.evaluate(p);assert c==1 and r['metrics']['median_standardized_max']>.5

def test_judge_cwd_report_and_hidden_errors(tmp_path):
    p=subprocess.run([sys.executable,str(T/'evaluation/judge.py'),'--prediction',str(T/'evaluation/baseline_reference.npz'),'--report',str(tmp_path/'nested/report.json')],cwd=tmp_path,capture_output=True)
    assert p.returncode==0 and json.loads((tmp_path/'nested/report.json').read_text())['passed']
    r,c=judge.evaluate(tmp_path/'missing.npz');assert c==2 and not r['metrics']
    r,c=judge.evaluate(T/'evaluation/baseline_reference.npz',tmp_path);assert c==2 and not r['metrics']
    blocker=tmp_path/'file';blocker.write_text('file');p=subprocess.run([sys.executable,str(T/'evaluation/judge.py'),'--prediction',str(T/'evaluation/baseline_reference.npz'),'--report',str(blocker/'report.json')],capture_output=True);assert p.returncode==2 and b'Report write failed' in p.stderr

def test_truth_priority_and_invalid_metric(tmp_path,good):
    np.savez(tmp_path/'baseline_reference.npz',**good);np.savez(tmp_path/'ground_truth.npz',wrong=np.ones(2))
    r,c=judge.evaluate(T/'evaluation/baseline_reference.npz',tmp_path);assert c==2 and 'Missing required key' in r['errors'][0]

def test_main_backend_and_public_constants():
    import matplotlib
    before=matplotlib.get_backend();from src import main
    assert matplotlib.get_backend()==before
    text=(T/'README.md').read_text(encoding='utf8');assert '| Reference pressure | 100000 |' in text and '| Carbon/oxygen ratio | 0.53 |' in text
