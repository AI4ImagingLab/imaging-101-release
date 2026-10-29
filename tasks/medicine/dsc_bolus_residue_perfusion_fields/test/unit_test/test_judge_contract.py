"""Independent small evaluator fixtures; no solver-generated expected values."""
from pathlib import Path
import importlib.util,json,subprocess,sys,shutil
import numpy as np
import pytest
T=Path(__file__).resolve().parents[2]

@pytest.fixture
def oracle(tmp_path):
    root=tmp_path/'case';ev=root/'evaluation';ev.mkdir(parents=True);(root/'data').mkdir()
    spec=importlib.util.spec_from_file_location('contract_judge',T/'evaluation/judge.py');judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge);judge.HERE=ev
    n=80;a=np.zeros(n);a[14]=30.;col=np.zeros(160)
    for i in range(1,79):col[i]=(a[i-1]+4*a[i]+a[i+1])/6
    matrix=np.array([[col[(i-j)%160]*1.55 for j in range(160)] for i in range(80)])
    residue=np.stack([.1*np.exp(-np.arange(160)/8),.2*np.exp(-np.arange(160)/12)]).reshape(1,1,1,2,160)
    observations=residue.reshape(2,160)@matrix.T
    signal=np.rint(1000*np.exp(-.025*observations)).astype(np.int16).reshape(1,1,1,2,80)
    support=np.ones((1,1,1,2),bool)
    np.savez(root/'data/raw_data.npz',signal=signal,support=support,arterial_input=a[None])
    fields=dict(reconstruction=residue,relative_flow=np.array([.1,.2]).reshape(1,1,1,2),relative_volume=np.array([1.,2.4]).reshape(1,1,1,2),transit_time=np.array([10.,12.]).reshape(1,1,1,2))
    cfg=dict(ncc_boundary=.99,nrmse_boundary=.01,forward_relative_l2_boundary=.1,required_outputs={k:dict(shape=list(x.shape),dtype=str(x.dtype)) for k,x in fields.items()})
    (ev/'metrics.json').write_text(json.dumps(cfg),encoding='utf8');np.savez(ev/'baseline_reference.npz',**fields)
    pred=root/'prediction.npz';np.savez(pred,**fields)
    return judge,pred,fields,ev,cfg

def test_analytic_metrics(oracle):
    judge,*_=oracle
    x=np.array([0.,1.,2.]);r=np.array([0.,2.,4.]);ncc,nrmse=judge.score_arrays(x,r)
    assert abs(ncc-1)<1e-15 and abs(nrmse-np.sqrt(5/3)/4)<1e-15
    assert judge.score_arrays(np.zeros(3),r)[0]==0
    with pytest.raises(ValueError):judge.score_arrays(x,np.ones(3))

@pytest.mark.parametrize('scale',[1e150,1e-150])
def test_norm_product_overflow_and_underflow(oracle,scale):
    judge,*_=oracle
    ncc,nrmse=judge.score_arrays(np.array([1.,2.,3.])*scale,np.array([2.,3.,4.])*scale)
    assert abs(ncc-20/np.sqrt(406))<2e-15 and abs(nrmse-.5)<2e-15

def test_known_valid(oracle):
    judge,pred,*_=oracle;report,code=judge.evaluate(pred)
    assert code==0 and report['passed'] and report['status']=='pass',report
    assert len(report['metrics'])==9 and len(report['thresholds'])==9

@pytest.mark.parametrize('kind',['missing_file','missing_key','shape','float32','complex','object','nan','inf','overflow'])
def test_malformed_prediction(oracle,kind):
    judge,pred,fields,_,_=oracle
    if kind=='missing_file':pred=pred.with_name('absent.npz')
    else:
        fields={k:v.copy() for k,v in fields.items()}
        if kind=='missing_key':fields.pop('relative_flow')
        elif kind=='shape':fields['reconstruction']=fields['reconstruction'][0]
        elif kind=='float32':fields['reconstruction']=fields['reconstruction'].astype(np.float32)
        elif kind=='complex':fields['reconstruction']=fields['reconstruction'].astype(np.complex128)
        elif kind=='object':fields['reconstruction']=fields['reconstruction'].astype(object)
        else:fields['reconstruction'].flat[0]={'nan':np.nan,'inf':np.inf,'overflow':1e308}[kind]
        np.savez(pred,**fields)
    report,code=judge.evaluate(pred);assert code==2 and report['status']=='error' and report['metrics']=={} and report['errors']
    json.dumps(report,allow_nan=False)

@pytest.mark.parametrize('key',['reconstruction','relative_flow','relative_volume','transit_time'])
def test_every_required_quantity_can_fail(oracle,key):
    judge,pred,fields,_,_=oracle;fields[key]=np.zeros_like(fields[key]);np.savez(pred,**fields)
    report,code=judge.evaluate(pred);assert code==1 and not report['passed'] and report['status']=='fail'

@pytest.mark.parametrize('boundary',[True,None,'0.1',-1,float('nan')])
def test_bad_configuration(oracle,boundary):
    judge,pred,_,ev,cfg=oracle;cfg['nrmse_boundary']=boundary;(ev/'metrics.json').write_text(json.dumps(cfg),encoding='utf8')
    report,code=judge.evaluate(pred);assert code==2 and report['metrics']=={}

def test_actual_truth_priority_and_no_fallback(oracle):
    judge,pred,fields,ev,_=oracle
    np.savez(ev/'baseline_reference.npz',**{k:2*v for k,v in fields.items()});np.savez(ev/'ground_truth.npz',**fields)
    assert judge.evaluate(pred)[1]==0
    np.savez(ev/'ground_truth.npz',wrong=np.ones(3));assert judge.evaluate(pred)[1]==2

def test_unavailable_reference(oracle):
    judge,pred,_,ev,_=oracle;(ev/'baseline_reference.npz').unlink();report,code=judge.evaluate(pred);assert code==2 and report['metrics']=={}

@pytest.mark.parametrize('kind',['shape','dtype','nan','constant'])
def test_invalid_reference_is_evaluator_error(oracle,kind):
    judge,pred,fields,ev,_=oracle
    if kind=='shape':fields['relative_flow']=np.ones(3)
    elif kind=='dtype':fields['relative_flow']=fields['relative_flow'].astype(np.float32)
    elif kind=='nan':fields['relative_flow'].flat[0]=np.nan
    else:fields['relative_flow'][:]=1
    np.savez(ev/'ground_truth.npz',**fields);report,code=judge.evaluate(pred);assert code==2 and report['metrics']=={}

def test_outside_support_zero_contract(oracle):
    judge,pred,_,ev,_=oracle;p=ev.parent/'data/raw_data.npz'
    with np.load(p,allow_pickle=False) as z:raw={k:z[k] for k in z.files}
    raw['support'].flat[0]=False;np.savez(p,**raw)
    report,code=judge.evaluate(pred);assert code==2 and 'outside' in report['errors'][0] and report['metrics']=={}

def test_native_foreign_cwd_missing_and_report_error(tmp_path):
    script=T/'evaluation/judge.py';report=tmp_path/'nested/report.json'
    command=[sys.executable,str(script),'--prediction',str(tmp_path/'absent.npz'),'--report',str(report)]
    run=subprocess.run(command,cwd=tmp_path,capture_output=True);assert run.returncode==2
    data=json.loads(report.read_text(encoding='utf8'));assert data['status']=='error' and data['metrics']=={}
    run=subprocess.run(command[:-1]+[str(tmp_path)],cwd=tmp_path,capture_output=True);assert run.returncode==2 and b'Report-write failure' in run.stderr
