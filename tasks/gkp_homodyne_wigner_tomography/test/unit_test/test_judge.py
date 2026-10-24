from pathlib import Path
import importlib.util,json,shutil,subprocess,sys
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[2]

def reference():
    with np.load(ROOT/'evaluation/baseline_reference.npz',allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}

def run(tmp_path,payload,judge=None):
    prediction=tmp_path/'prediction.npz';report=tmp_path/'nested/report.json'
    if payload is not None:np.savez_compressed(prediction,**payload)
    child=subprocess.run([sys.executable,str(judge or ROOT/'evaluation/judge.py'),'--prediction',str(prediction),'--report',str(report)],cwd=tmp_path,capture_output=True,text=True)
    result=json.loads(report.read_text(encoding='utf-8'))
    assert set(result)=={'status','passed','metrics','thresholds','errors'}
    return child.returncode,result

@pytest.mark.parametrize('single_precision',[False,True])
def test_valid_foreign_cwd_method_reference_and_declared_precision(tmp_path,single_precision):
    r=reference()
    if single_precision:r={'density_matrix':r['density_matrix'].astype(np.complex64),'wigner':r['wigner'].astype(np.float32)}
    code,report=run(tmp_path,r)
    assert code==0 and report['passed'] and report['status']=='pass'

@pytest.mark.parametrize('kind',['zero','mixed_state','wrong_W_sign','density_scale','nonhermitian','bad_trace','component_phase'])
def test_valid_poor_physics_rejected_without_relaxation(tmp_path,kind):
    r=reference()
    if kind=='zero':r={k:np.zeros_like(v) for k,v in r.items()}
    elif kind=='mixed_state':r['density_matrix'][0]=np.eye(31)/31
    elif kind=='wrong_W_sign':r['wigner']*=-1
    elif kind=='density_scale':r['density_matrix']*=1.01
    elif kind=='nonhermitian':r['density_matrix'][0,0,1]+=.01j
    elif kind=='bad_trace':r['density_matrix'][0,0,0]+=.01
    else:r['density_matrix'][0,0,2]*=-1
    code,report=run(tmp_path,r)
    assert code==1 and report['status']=='fail' and not report['passed'] and report['metrics']

@pytest.mark.parametrize('kind',['missing_file','missing_key','shape','rho_real','wigner_complex','integer','object','nan','inf','corrupt'])
def test_malformed_inputs_exit2_and_empty_metrics(tmp_path,kind):
    r=reference()
    if kind=='missing_file':r=None
    elif kind=='missing_key':r.pop('density_matrix')
    elif kind=='shape':r['wigner']=r['wigner'][:,1:]
    elif kind=='rho_real':r['density_matrix']=r['density_matrix'].real
    elif kind=='wigner_complex':r['wigner']=r['wigner'].astype(complex)
    elif kind=='integer':r['wigner']=r['wigner'].astype(int)
    elif kind=='object':r['wigner']=r['wigner'].astype(object)
    elif kind=='nan':r['wigner'][0,0,0]=np.nan
    elif kind=='inf':r['density_matrix'][0,0,0]=np.inf
    elif kind=='corrupt':r=None;(tmp_path/'prediction.npz').write_bytes(b'invalid NPZ')
    code,report=run(tmp_path,r)
    assert code==2 and report['status']=='error' and not report['passed'] and report['metrics']=={} and report['errors']

@pytest.mark.parametrize('kind',['missing_reference','bad_reference','invalid_threshold'])
def test_evaluator_asset_errors(tmp_path,kind):
    e=tmp_path/'evaluator';shutil.copytree(ROOT/'evaluation',e)
    if kind=='missing_reference':(e/'baseline_reference.npz').unlink()
    elif kind=='bad_reference':np.savez_compressed(e/'baseline_reference.npz',**{k:np.zeros_like(v) for k,v in reference().items()})
    else:
        path=e/'metrics.json';m=json.loads(path.read_text());m['rho_relative_l2_boundary']='bad';path.write_text(json.dumps(m))
    code,report=run(tmp_path,reference(),e/'judge.py')
    assert code==2 and report['metrics']=={} and report['errors']

def test_report_write_failure_returns2(tmp_path):
    p=tmp_path/'prediction.npz';np.savez_compressed(p,**reference())
    blocked=tmp_path/'blocked';blocked.write_text('file')
    child=subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(p),'--report',str(blocked/'report.json')],cwd=tmp_path,capture_output=True,text=True)
    assert child.returncode==2 and 'Cannot write' in child.stderr


@pytest.mark.parametrize('kind',['truth_wins','invalid_truth_no_fallback','truth_without_baseline'])
def test_actual_truth_has_priority_over_method_reference(tmp_path,kind):
    e=tmp_path/'evaluator';shutil.copytree(ROOT/'evaluation',e)
    axis=np.linspace(-7,7,201);Q,P=np.meshgrid(axis,axis)
    rho=np.zeros((1,31,31),complex);rho[0,0,0]=1
    truth={'density_matrix':rho,'wigner':np.exp(-(Q*Q+P*P))[None]/np.pi}
    np.savez_compressed(e/'ground_truth.npz',**truth)
    if kind=='invalid_truth_no_fallback':
        (e/'ground_truth.npz').write_bytes(b'Invalid truth must be an evaluator error, never silently fallback')
        code,report=run(tmp_path,reference(),e/'judge.py')
        assert code==2 and report['metrics']=={} and report['errors']
    else:
        if kind=='truth_without_baseline':(e/'baseline_reference.npz').unlink()
        code,report=run(tmp_path,truth,e/'judge.py')
        assert code==0 and report['passed']
        if kind=='truth_wins':
            code,report=run(tmp_path,reference(),e/'judge.py')
            assert code==1 and not report['passed']

def test_independent_Judge_Wigner_matches_analytic_vacuum_and_signed_single_photon():
    spec=importlib.util.spec_from_file_location('independent_judge',ROOT/'evaluation/judge.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    axis=np.linspace(-7,7,201);Q,P=np.meshgrid(axis,axis);r2=Q*Q+P*P
    for n in (0,1):
        rho=np.zeros((31,31),complex);rho[n,n]=1
        expected=np.exp(-r2)/np.pi*(1 if n==0 else 2*r2-1)
        np.testing.assert_allclose(m.independent_wigner(rho),expected,atol=1e-14,rtol=1e-13)
