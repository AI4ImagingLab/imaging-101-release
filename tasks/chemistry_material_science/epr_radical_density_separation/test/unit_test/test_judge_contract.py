import json,sys,subprocess,shutil
import numpy as np
import pytest

def test_valid_original_prediction(judge,valid_prediction,isolated_assets):
 report,code=judge.evaluate(valid_prediction,isolated_assets);assert code==0 and report['status']=='pass' and report['passed']
 assert report['metrics']['TAM_mass_relative_error']<.15 and report['metrics']['TEMPO_mass_relative_error']<.15

@pytest.mark.parametrize('case',['missing','corrupt','missing_key','extra_key','shape','float32','integer','complex','nan','inf','object'])
def test_malformed_predictions(judge,tmp_path,isolated_assets,case):
 file=tmp_path/'bad.npz';x=np.zeros((1,2,100,100))
 if case=='corrupt':file.write_text('not a zip')
 elif case=='missing':pass
 elif case=='missing_key':np.savez(file,other=x)
 elif case=='extra_key':np.savez(file,reconstruction=x,privileged=x)
 else:
  if case=='shape':x=x[0]
  if case=='float32':x=x.astype(np.float32)
  if case=='integer':x=x.astype(np.int64)
  if case=='complex':x=x.astype(complex)
  if case=='nan':x.flat[0]=np.nan
  if case=='inf':x.flat[-1]=np.inf
  if case=='object':x=np.array([object()],dtype=object)
  np.savez(file,reconstruction=x)
 report,code=judge.evaluate(file,isolated_assets);assert code==2 and report['status']=='error' and not report['passed'] and report['metrics']=={} and report['errors']

@pytest.mark.parametrize('case',['zeros','negative','swapped'])
def test_poor_fields_are_valid_failures(judge,tmp_path,isolated_assets,original,case):
 x=original['original_reconstruction'].copy()
 if case=='zeros':x*=0
 if case=='negative':x=-x
 if case=='swapped':x=x[::-1]
 file=tmp_path/'poor.npz';np.savez(file,reconstruction=x[None]);report,code=judge.evaluate(file,isolated_assets)
 assert code==1 and report['status']=='fail' and report['metrics'] and not report['errors']

@pytest.mark.parametrize('key,value',[('ncc_boundary',True),('ncc_boundary','0.9'),('ncc_boundary',0),('ncc_boundary',2),('mass_relative_error_boundary',False),('mass_relative_error_boundary','0.15'),('mass_relative_error_boundary',-1),('nrmse_boundary_per_species',[.1,'0.1']),('nrmse_boundary_per_species',[.1,True]),('nrmse_boundary_per_species',[.1]),('forward_noise_rms_lower_boundary','0.9'),('forward_noise_rms_upper_boundary',True),('absolute_residual_lag1_boundary',0),('absolute_residual_lag1_boundary',float('nan'))])
def test_invalid_metric_config(judge,valid_prediction,isolated_assets,key,value):
 path=isolated_assets/'metrics.json';config=json.loads(path.read_text());config[key]=value;path.write_text(json.dumps(config));report,code=judge.evaluate(valid_prediction,isolated_assets)
 assert code==2 and report['metrics']=={} and report['errors']

def test_truth_takes_priority(judge,valid_prediction,isolated_assets):
 np.savez(isolated_assets/'baseline_reference.npz',reconstruction=np.zeros((1,2,100,100)))
 report,code=judge.evaluate(valid_prediction,isolated_assets);assert code==0

def test_malformed_truth_is_not_replaced(judge,valid_prediction,isolated_assets):
 shutil.copyfile(isolated_assets/'ground_truth.npz',isolated_assets/'baseline_reference.npz');np.savez(isolated_assets/'ground_truth.npz',bad=np.zeros(1))
 report,code=judge.evaluate(valid_prediction,isolated_assets);assert code==2 and report['metrics']=={}

def test_baseline_fallback_and_missing_assets(judge,valid_prediction,isolated_assets):
 (isolated_assets/'ground_truth.npz').rename(isolated_assets/'baseline_reference.npz')
 assert judge.evaluate(valid_prediction,isolated_assets)[1]==0
 (isolated_assets/'baseline_reference.npz').unlink();report,code=judge.evaluate(valid_prediction,isolated_assets);assert code==2 and report['metrics']=={}

def test_degenerate_reference_is_error(judge,valid_prediction,isolated_assets):
 np.savez(isolated_assets/'ground_truth.npz',reconstruction=np.ones((1,2,100,100)))
 assert judge.evaluate(valid_prediction,isolated_assets)[1]==2

def test_metric_definitions_on_known_scaled_truth(judge,tmp_path,isolated_assets):
 with np.load(isolated_assets/'ground_truth.npz') as archive:truth=archive['reconstruction']
 prediction=tmp_path/'double.npz';np.savez(prediction,reconstruction=2*truth)
 report,code=judge.evaluate(prediction,isolated_assets);assert code==1
 for j,label in enumerate(('TAM','TEMPO')):
  assert abs(report['metrics'][f'{label}_ncc']-1)<1e-12
  assert abs(report['metrics'][f'{label}_nrmse']-np.sqrt(np.mean(truth[0,j]**2)))<1e-12
  assert abs(report['metrics'][f'{label}_mass_relative_error']-1)<1e-12

def test_report_write_failure(root,valid_prediction,tmp_path):
 report=tmp_path/'report_directory';report.mkdir()
 r=subprocess.run([sys.executable,str(root/'evaluation/judge.py'),'--prediction',str(valid_prediction),'--report',str(report)],capture_output=True,text=True)
 assert r.returncode==2 and 'Report write failed' in r.stderr
