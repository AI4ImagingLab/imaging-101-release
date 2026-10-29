from pathlib import Path
import importlib.util
import numpy as np
import pytest
T=Path(__file__).resolve().parents[2]
s=importlib.util.spec_from_file_location('judge',T/'evaluation/judge.py');judge=importlib.util.module_from_spec(s);s.loader.exec_module(judge)
@pytest.mark.parametrize('kind',['missing','key','shape','dtype','nan','inf'])
def test_malformed(tmp_path,kind):
    path=tmp_path/'prediction.npz'
    if kind!='missing':
        with np.load(T/'evaluation/baseline_reference.npz',allow_pickle=False) as z:data={k:z[k].copy() for k in z.files}
        if kind=='key':data.pop('transition')
        if kind=='shape':data['transition']=np.zeros((1,3,3))
        if kind=='dtype':data['noise']=data['noise'].astype(np.float16)
        if kind in ('nan','inf'):data['noise'][0,0]=float(kind)
        np.savez_compressed(path,**data)
    r=judge.evaluate(path);assert r['status']=='error' and not r['metrics'] and r['errors']
def test_reference_and_poor(tmp_path):
    assert judge.evaluate(T/'evaluation/baseline_reference.npz')['passed']
    with np.load(T/'evaluation/baseline_reference.npz',allow_pickle=False) as z:data={k:z[k].copy() for k in z.files}
    data['posterior']*=0;data['noise']*=2;p=tmp_path/'poor.npz';np.savez_compressed(p,**data)
    assert judge.evaluate(p)['status']=='fail'
def test_corrected_early_emission_basis():
    from src.physics_model import ms2_loading_coeff,get_adjusted,compute_dynamic_F
    w=ms2_loading_coeff(1.5,19);reduction=np.array([w[0,t+1:].sum() for t in range(18)])[:,None]
    for state in [0,1,31,12413,2**19-1]:
        on,off=get_adjusted(state,2,19,w);f=compute_dynamic_F(state,84,19,2,w,reduction)
        for ti in [0,1,17,18,83]:np.testing.assert_allclose([np.exp(v[0,ti]) for v in f],[off,on],atol=1e-12)

def test_original_signed_log_normal_equation():
    from src.physics_model import v_log_solve
    with np.load(T/'test/fixtures/emission_normal_equation.npz',allow_pickle=False) as z:M=z['input_matrix'];b=z['input_rhs'];expected=z['output_emission']
    result=v_log_solve(np.log(abs(M)),np.sign(M).copy(),np.log(abs(b)).ravel(),np.sign(b).ravel().copy())
    np.testing.assert_allclose(np.exp(result[0]),expected,rtol=1e-6,atol=1e-7)
    np.testing.assert_allclose(M@np.exp(result[0]),b,rtol=1e-10)

def test_truth_precedence_and_error(tmp_path):
    import shutil
    assets=tmp_path/'evaluation';assets.mkdir();(tmp_path/'data').mkdir()
    for name in ('metrics.json','baseline_reference.npz'):shutil.copyfile(T/'evaluation'/name,assets/name)
    shutil.copyfile(T/'data/raw_data.npz',tmp_path/'data/raw_data.npz')
    baseline=assets/'baseline_reference.npz';assert judge.evaluate(baseline,assets)['passed']
    (assets/'ground_truth.npz').write_bytes(b'invalid high-priority asset')
    assert judge.evaluate(baseline,assets)['status']=='error'

def test_cli_report_failure(tmp_path):
    import subprocess,sys
    command=[sys.executable,str(T/'evaluation/judge.py'),'--prediction','absent.npz','--report',str(tmp_path/'nested/report.json')]
    p=subprocess.run(command,cwd=tmp_path,capture_output=True);assert p.returncode==2 and (tmp_path/'nested/report.json').is_file()
    command[-1]=str(tmp_path);p=subprocess.run(command,capture_output=True);assert p.returncode==2 and b'Report write failed' in p.stderr

def test_operator_memoization_is_exact_and_known_metric_cases():
    from src.physics_model import get_adjusted,_get_adjusted_uncached,compute_dynamic_F,_compute_dynamic_F_uncached,ms2_loading_coeff
    with pytest.raises(ValueError):ms2_loading_coeff(2.5,19)
    for kappa in (1.5,1.75):
        weights=ms2_loading_coeff(kappa,19)
        for state in (0,1,31,12413,2**19-1):
            np.testing.assert_array_equal(get_adjusted(state,2,19,weights),_get_adjusted_uncached(state,2,19,weights))
            for a,b in zip(compute_dynamic_F(state,84,19,2,weights,None),_compute_dynamic_F_uncached(state,84,19,2,weights,None)):np.testing.assert_array_equal(a,b)
    r={k:np.ones(shape) for k,shape in judge.SHAPES.items()};x={k:a.copy() for k,a in r.items()}
    x['posterior']*=.5;x['transition']*=1.1;x['initial']*=1.2;x['emission']*=2;x['noise']*=2
    result=judge.score(x,r,np.full(63,86))
    for k,v in {'posterior_rmse':.5,'transition_rmse':.1,'initial_rmse':.2,'emission_relative_l2':1.,'noise_relative_error':1.}.items():assert np.isclose(result[k],v)

def test_forward_mean_constant_off_history():
    from src.visualization import predicted_signal
    result={'posterior':np.zeros((1,2,5)),'emission':np.array([[2.,3.]])}
    np.testing.assert_allclose(predicted_signal(result,np.array([5,4]))[0],36.5,atol=1e-12)

def test_response_independent_integrated_ramp():
    from src.physics_model import ms2_loading_coeff
    for kappa in (1.5,1.75):
        expected=[]
        for j in range(19):
            if j>=kappa:value=1.
            elif j+1<=kappa:value=((j+1)**2-j*j)/(2*kappa)
            else:value=(kappa*kappa-j*j)/(2*kappa)+j+1-kappa
            expected.append(value)
        np.testing.assert_allclose(ms2_loading_coeff(kappa,19)[0],expected,atol=1e-15)
