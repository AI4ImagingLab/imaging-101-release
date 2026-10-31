from pathlib import Path
import importlib.util,json,subprocess,sys
import numpy as np
import pytest
T=Path(__file__).resolve().parents[2]
s=importlib.util.spec_from_file_location('judge',T/'evaluation/judge.py');j=importlib.util.module_from_spec(s);s.loader.exec_module(j)

def test_known_metrics():
    r={'reconstruction':np.array([0.,3.,4.]),'contact_position':np.zeros(3)}
    p={'reconstruction':np.array([0.,6.,8.]),'contact_position':np.ones(3)}
    m=j.metrics(p,r);assert m['ncc']==1;assert m['modulus_max_abs_pa']==4
    np.testing.assert_allclose(m['nrmse'],np.sqrt(25/3)/4)
    assert m['contact_max_abs_m']==1

@pytest.mark.parametrize('bad',['missing','key','shape','dtype','complex','nan','inf','negative','overflow','poor'])
def test_bad_inputs(tmp_path,bad):
    z=j.read_arrays(T/'evaluation/baseline_reference.npz');p=tmp_path/'p.npz'
    if bad=='key':del z['contact_position']
    if bad=='shape':z['reconstruction']=np.zeros(454)
    if bad=='dtype':z['reconstruction']=np.zeros((1,454),dtype=int)
    if bad=='complex':z['reconstruction']=z['reconstruction'].astype(complex)
    if bad in ['nan','inf']:z['reconstruction'][0,0]=float(bad)
    if bad=='negative':z['reconstruction'][0,0]=-1
    if bad=='overflow':z['reconstruction'][:]=1e308
    if bad=='poor':z['reconstruction']*=0
    if bad!='missing':np.savez(p,**z)
    r,c=j.evaluate(p);assert c==(1 if bad=='poor' else 2)
    assert r['status']==('fail' if bad=='poor' else 'error')
    if c==2:assert r['metrics']=={} and r['errors']

def test_assets_and_cli(tmp_path):
    r,c=j.evaluate(T/'output/reconstruction.npz',assets=tmp_path);assert c==2
    path=tmp_path/'nested/report.json'
    cp=subprocess.run([sys.executable,str(T/'evaluation/judge.py'),'--prediction',str(T/'output/reconstruction.npz'),'--report',str(path)],cwd=tmp_path,capture_output=True)
    assert cp.returncode==0,cp.stderr
    assert json.loads(path.read_text())['passed']
    bad=subprocess.run([sys.executable,str(T/'evaluation/judge.py'),'--prediction',str(T/'output/reconstruction.npz'),'--report',str(tmp_path)],capture_output=True)
    assert bad.returncode==2 and b'Cannot write report' in bad.stderr

def test_contact_float32_and_bad_reference(tmp_path):
    z=j.read_arrays(T/'evaluation/baseline_reference.npz');p=tmp_path/'p.npz'
    np.savez(p,**{k:v.astype(np.float32) for k,v in z.items()})
    assert j.evaluate(p)[1]==0
    z['contact_position']+=1e-6;np.savez(p,**z)
    assert j.evaluate(p)[1]==1
    (tmp_path/'metrics.json').write_text((T/'evaluation/metrics.json').read_text(),encoding='utf8')
    np.savez(tmp_path/'baseline_reference.npz',reconstruction=np.ones((1,454)),contact_position=np.ones((1,454)))
    assert j.evaluate(p,assets=tmp_path)[1]==2
    cfg=json.loads((tmp_path/'metrics.json').read_text());cfg['maximum_curve_loss_excess_boundary']=-1
    (tmp_path/'metrics.json').write_text(json.dumps(cfg),encoding='utf8')
    assert j.evaluate(p,assets=tmp_path)[1]==2


def test_known_contact_law_and_profiled_force_baseline():
    x=np.linspace(4e-6,-8e-6,200);depth=np.maximum(-x,0);radius=18.64e-6;q=depth/radius
    f=(4/3)*1000/(1-.5**2)*np.sqrt(radius)*depth**1.5*(1-q/10-q*q/840+11*q**3/15120+1357*q**4/6652800)+2e-9
    height=x-f/.1;segment=np.zeros(200,dtype=np.uint8)
    assert j.curve_loss(1000,0,f,height,segment,.1)<1e-28
    assert j.curve_loss(0,0,f,height,segment,.1)>1e-3


def test_different_independent_linear_modulus_fit_passes(tmp_path):
    arrays=j.read_arrays(T/'evaluation/baseline_reference.npz')
    with np.load(T/'data/raw_data.npz',allow_pickle=False) as raw:data={k:raw[k][0] for k in raw.files}
    improvements=[]
    for i in [0,50,100,200,400]:
        lo,hi=data['offsets'][i:i+2];f=data['force'][lo:hi];h=data['height'][lo:hi];segment=data['segment'][lo:hi];spring=data['spring_constant'][i]
        keep=segment==0;x=h[keep]+f[keep]/spring;c=arrays['contact_position'][0,i];depth=np.maximum(c-x,0);q=depth/18.64e-6
        basis=(4/3)*1000/(1-.5**2)*np.sqrt(18.64e-6)*depth**1.5*(1-q/10-q*q/840+11*q**3/15120+1357*q**4/6652800)
        w=np.minimum(abs(x-c)/2e-6,1)
        coefficients=np.linalg.lstsq(np.column_stack([basis*1e9,np.ones(len(x))])*w[:,None],f[keep]*1e9*w,rcond=None)[0]
        modulus=coefficients[0]*1000
        before=j.curve_loss(arrays['reconstruction'][0,i],c,f,h,segment,spring)
        after=j.curve_loss(modulus,c,f,h,segment,spring)
        if modulus>=0 and after<before-1e-12:
            arrays['reconstruction'][0,i]=modulus;improvements.append(before-after)
    assert improvements
    path=tmp_path/'improved.npz';np.savez(path,**arrays)
    report,code=j.evaluate(path);assert code==0,report
