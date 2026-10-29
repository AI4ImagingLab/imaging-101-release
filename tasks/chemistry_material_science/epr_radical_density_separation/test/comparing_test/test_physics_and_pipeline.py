import json,sys,subprocess,hashlib
import numpy as np

def test_gradient_divergence_and_constant():
 from src.physics_model import grad2d,div2d
 rng=np.random.default_rng(16);u=rng.normal(size=(17,23));p=rng.normal(size=(2,17,23));a=np.vdot(grad2d(u),p);b=-np.vdot(u,div2d(p))
 assert abs(a-b)/max(abs(a),abs(b))<1e-12
 assert np.array_equal(grad2d(np.ones((7,9))),np.zeros((2,7,9)))

def test_adjoint_and_directional_derivative(model,public):
 rng=np.random.default_rng(82);u=rng.normal(size=(2,100,100));v=rng.normal(size=(40,1200));a=np.vdot(model.project(u),v);b=np.vdot(u,model.adjoint(v));assert abs(a-b)/max(abs(a),abs(b))<1e-11
 direction=rng.normal(size=u.shape);e=1e-5;f=lambda x:.5*np.sum((model.project(x)-public['projections'])**2);fd=(f(u+e*direction)-f(u-e*direction))/(2*e);g=np.vdot(model.adjoint(model.project(u)-public['projections']),direction)
 assert abs(fd-g)/max(abs(fd),abs(g))<1e-5

def test_known_unit_reparameterization(model,public,original):
 from src.physics_model import EPRModel
 u=original['operator_test_density'];baseline=model.project(u)
 # G to mT changes field and gradient by the same factor; phase is invariant.
 mt=EPRModel(public['magnetic_field_G']/10,public['gradients_G_cm']/10,public['reference_spectra'])
 assert np.linalg.norm(mt.project(u)-baseline)/np.linalg.norm(baseline)<1e-10
 # cm to mm: area100x larger and density100x smaller, unchanged receiver signal.
 mm=EPRModel(public['magnetic_field_G'],public['gradients_G_cm']/10,public['reference_spectra'],delta=.3)
 assert np.linalg.norm(mm.project(u/100)-baseline)/np.linalg.norm(baseline)<1e-10

def test_local_generator_against_original(root,public):
 from src.generate_data import generate_author_observations
 generated=generate_author_observations(public)
 with np.load(root/'test/fixtures/original_generation_control.npz') as f:
  assert np.array_equal(generated['truth'],f['generation_truth'])
  assert np.linalg.norm(generated['clean']-f['clean_float64'])/np.linalg.norm(f['clean_float64'])<1e-9
  assert np.linalg.norm(generated['clean']-f['clean_original_float32'])/np.linalg.norm(f['clean_original_float32'])<1e-5
  regenerated=(f['clean_original_float32']+generated['noise']).astype(np.float32)
  assert np.array_equal(regenerated,f['observed_original_float32'])

def test_default_cli_and_foreign_cwd_judge(root,tmp_path):
 first=subprocess.run([sys.executable,'-m','src.main'],cwd=root,capture_output=True,text=True);assert first.returncode==0,first.stderr
 report=tmp_path/'nested/report.json';second=subprocess.run([sys.executable,str(root/'evaluation/judge.py'),'--prediction',str(root/'output/reconstruction.npz'),'--report',str(report)],cwd=tmp_path,capture_output=True,text=True);assert second.returncode==0,second.stderr
 result=json.loads(report.read_text());assert result['passed'] and result['status']=='pass'
 assert result['metrics']['TAM_mass_relative_error']<.15 and result['metrics']['TEMPO_mass_relative_error']<.15
