from pathlib import Path
import importlib.util,json
import numpy as np
import pytest
from src.preprocessing import prepare
from src.physics_model import DopplerModel,triangular_cdf
from src.solvers import objective_derivatives
from src.generate_data import add_spectral_noise
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('independent_judge',ROOT/'evaluation/judge.py')
judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)

@pytest.fixture(scope='module')
def small():
 raw=dict(np.load(ROOT/'data/raw_data.npz',allow_pickle=False));p=prepare(raw);m=DopplerModel(p,nv=16)
 return raw,m,m.initialize_data(p['line_spectra'])

def test_triangular_cdf_known_values():
 np.testing.assert_array_equal(triangular_cdf(np.array([-1,0,.25,.5,.75,1,2])),[0,0,.125,.5,.875,1,1])

def test_adjoint_flux_and_default(small):
 _,m,_=small;r=np.random.default_rng(17);x=r.normal(size=m.n_pixels);y=r.normal(size=m.nph*m.nvp)
 np.testing.assert_allclose(np.dot(m.forward(x),y),np.dot(x,m.backward(y)),atol=1e-12,rtol=1e-12)
 np.testing.assert_allclose(np.asarray(m.P_csc.sum(axis=0)).ravel(),1,atol=2e-8,rtol=0)
 z=r.normal(size=m.n_pixels)
 np.testing.assert_allclose(np.dot(x,m.smear(z)),np.dot(m.smear(x),z),atol=1e-12,rtol=1e-12)
 assert np.all(m.smear(np.ones(m.n_pixels))>0)

def test_independent_objective_finite_difference(small):
 _,m,s=small;x=np.ones(m.n_pixels)*2;d=np.cos(np.arange(m.n_pixels))*.01
 H,S,gH,gS,*_=objective_derivatives(m,x,s['positive_data'],s['weights'])
 plus=objective_derivatives(m,x+1e-3*d,s['positive_data'],s['weights'])
 minus=objective_derivatives(m,x-1e-3*d,s['positive_data'],s['weights'])
 np.testing.assert_allclose((plus[0]-minus[0])/.002,np.dot(gH,d),rtol=1e-5,atol=1e-7)
 np.testing.assert_allclose((plus[1]-minus[1])/.002,np.dot(gS,d),rtol=1e-5,atol=1e-7)

def test_generator_seed_units_and_no_mutation(small):
 raw,_,_=small;a=add_spectral_noise(raw,.01,23);b=add_spectral_noise(raw,.01,23)
 for key in raw:np.testing.assert_array_equal(a[key],b[key])
 np.testing.assert_array_equal(a['wavelength'],raw['wavelength']);np.testing.assert_array_equal(a['orbital_phase'],raw['orbital_phase'])
 delta=a['spectrum']-raw['spectrum'];expected=.01*np.max(abs(raw['spectrum']))
 assert abs(np.std(delta)/expected-1)<.02
 assert np.max(abs(delta))>0
 for scale in [-1,True,float('nan')]:
  with pytest.raises(ValueError):add_spectral_noise(raw,scale)

def test_known_metrics():
 y=np.array([0.,1.,2.]);np.testing.assert_allclose(judge.reference_metrics(y,y),(1.,0.),atol=1e-15,rtol=0)
 cosine,error=judge.reference_metrics(np.zeros(3),y)
 assert cosine==0;np.testing.assert_allclose(error,np.sqrt(5/3)/2)
 with pytest.raises(ValueError):judge.reference_metrics(y,np.ones(3))

@pytest.mark.parametrize('value',[True,'0.1',-1,float('inf'),float('nan')])
def test_bad_error_configuration(tmp_path,value):
 c=dict(ncc_boundary=.9,nrmse_boundary=.1,forward_range_rms_boundary=.1);c['nrmse_boundary']=value
 p=tmp_path/'config.json';p.write_text(json.dumps(c),encoding='utf8')
 with pytest.raises(ValueError):judge.config(p)

@pytest.mark.parametrize('value',[-1.1,1.1,False,'0.9'])
def test_bad_cosine_configuration(tmp_path,value):
 p=tmp_path/'config.json';p.write_text(json.dumps(dict(ncc_boundary=value,nrmse_boundary=.1,forward_range_rms_boundary=.1)),encoding='utf8')
 with pytest.raises(ValueError):judge.config(p)

@pytest.mark.parametrize('kind',['missing','key','shape','complex','integer','nan','inf'])
def test_malformed_submission(tmp_path,kind):
 path=tmp_path/'submission.npz';a=np.zeros((1,144,144))
 if kind=='missing':pass
 elif kind=='key':np.savez(path,wrong=a)
 else:
  if kind=='shape':a=a[0]
  if kind=='complex':a=a.astype(complex)
  if kind=='integer':a=a.astype(int)
  if kind in ['nan','inf']:a.flat[0]=np.nan if kind=='nan' else np.inf
  np.savez(path,reconstruction=a)
 result=judge.evaluate(path);assert result['status']=='error' and result['metrics']=={} and result['errors']

def test_truth_priority_invalid_truth_and_overflow(tmp_path,monkeypatch):
 e=tmp_path/'evaluation';e.mkdir();c=dict(ncc_boundary=.99,nrmse_boundary=.01,forward_range_rms_boundary=.1)
 (e/'metrics.json').write_text(json.dumps(c),encoding='utf8');y=np.arange(144*144,dtype=float).reshape(1,144,144)
 np.savez(e/'baseline_reference.npz',reconstruction=-y);np.savez(e/'ground_truth.npz',reconstruction=y)
 p=tmp_path/'prediction.npz';np.savez(p,reconstruction=y)
 monkeypatch.setattr(judge,'forward_range_rms',lambda *args:0.)
 assert judge.evaluate(p,e)['status']=='pass'
 np.savez(e/'ground_truth.npz',wrong=y)
 assert judge.evaluate(p,e)['status']=='error'
 np.savez(e/'ground_truth.npz',reconstruction=y);np.savez(p,reconstruction=np.full_like(y,1e308))
 result=judge.evaluate(p,e);assert result['status']=='error' and result['metrics']=={}

@pytest.mark.parametrize('control',['zero','constant','shuffle','negative','transpose'])
def test_poor_whole_field_controls(control):
 y=np.load(ROOT/'evaluation/baseline_reference.npz')['reconstruction'];x=y.copy()
 if control=='zero':x*=0
 if control=='constant':x.fill(y.mean())
 if control=='shuffle':x=np.random.default_rng(18).permutation(x.ravel()).reshape(x.shape)
 if control=='negative':x=-x
 if control=='transpose':x=x.transpose(0,2,1)
 cosine,error=judge.reference_metrics(x,y)
 assert cosine<.999999 or error>7.691681073121665e-6

def test_cli_foreign_directory_and_report_failure(tmp_path):
 import subprocess,sys
 command=[sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(ROOT/'output/reconstruction.npz'),'--report',str(tmp_path/'nested/report.json')]
 p=subprocess.run(command,cwd=tmp_path,capture_output=True,text=True)
 assert p.returncode==0,p.stderr
 report=json.loads((tmp_path/'nested/report.json').read_text());assert report['passed'] and not report['errors']
 bad=tmp_path/'bad.npz';np.savez(bad,reconstruction=np.zeros((1,144,144)))
 command[3]=str(bad);p=subprocess.run(command,cwd=tmp_path,capture_output=True,text=True)
 assert p.returncode==1
 command[3]=str(tmp_path/'missing.npz');p=subprocess.run(command,cwd=tmp_path,capture_output=True,text=True)
 assert p.returncode==2
 command[5]=str(tmp_path);p=subprocess.run(command,cwd=tmp_path,capture_output=True,text=True)
 assert p.returncode==2 and 'Report write failure' in p.stderr
