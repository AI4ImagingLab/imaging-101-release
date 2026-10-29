from pathlib import Path
import importlib.util
import numpy as np
import pytest
from src.preprocessing import load_data,parent_input
from src.physics_model import cumulative_integral,transformed_data,linear_relative_residual
from src.solvers import fit_tacs,reconstruct
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location("independent_pet_judge",ROOT/"evaluation/judge.py");judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)

def test_data_timing_calibration_and_readme():
    a,m,cp,s,d,raw=load_data();assert a.shape==(1,200,200,27) and m.sum()==5029
    assert s[0]==0 and s[-1]+d[-1]==90 and np.all(cp>0)
    expected=(a[...,-10:].mean(-1)>=1000)&np.all(a[...,s>=30]>0,axis=-1);assert np.array_equal(m,expected)
    assert raw["activity"].dtype==np.float32 and a.dtype==np.float64
    assert raw["voxel_to_world"][0,0,0]==-2.0364201068878174
    readme=(ROOT/"README.md").read_text(encoding="utf-8");assert "github" not in readme.lower()
    for key in raw:assert f"| {key} |" in readme

def test_linear_plasma_interpolation():
    cp=parent_input(np.array([60.,120.]),np.array([10.,20.]),np.array([.5,.5]),np.array([0.,60.]),np.array([60.,60.]))
    assert np.array_equal(cp,np.array([2.5,7.5]))

def test_quadrature_known_linear_curve():
    s=np.array([0.,1.,2.]);d=np.ones(3);mid=s+.5;a=2*mid+3
    expected=mid**2+3*mid-(.5**2+3*.5)
    assert np.array_equal(cumulative_integral(a,s,d),expected)
    assert np.array_equal(cumulative_integral(a,s,d,"rect"),np.cumsum(a*d))
    with pytest.raises(ValueError):cumulative_integral(a,s,d,"bad")

def test_reversible_one_tissue_physics_and_units():
    # Analytic ODE: dCt/dt=K1 Cp-k2 Ct; true distribution volume=K1/k2.
    dt=.0005;s=np.arange(0.,6.,dt);d=np.full(s.shape,dt);t=s+.5*dt
    k1=2.;k2=1.;kp=.3;cp=1e4*np.exp(-kp*t);ct=k1*1e4/(k2-kp)*(np.exp(-kp*t)-np.exp(-k2*t))
    v,_=fit_tacs(ct[None],cp,s,d,tstar=2.)
    assert v[0]==pytest.approx(k1/k2,rel=1e-4)
    scaled,_=fit_tacs(ct[None]/1000,cp/1000,s,d,tstar=2.)
    assert scaled[0]==pytest.approx(v[0],rel=1e-11)

def test_weighted_normal_equation_orthogonality():
    z=np.load(ROOT/"test/fixtures/upstream_stages.npz");a=z["input_activity"];cp=z["input_cp"];s=z["input_start"];d=z["input_duration"]
    v,b=fit_tacs(a,cp,s,d);x,y,w=transformed_data(a,cp,s,d)
    residual=y-b[:,None]-v[:,None]*x[:,:,1]
    grad=np.einsum("nji,nj,j->ni",x,residual,w)
    scale=np.maximum(1.,np.linalg.norm(x,axis=(1,2))*np.linalg.norm(y,axis=1)*np.max(w))
    assert np.max(np.linalg.norm(grad,axis=1)/scale)<1e-12

def test_original_equal_input_special_case():
    s=np.arange(6,dtype=float);d=np.ones(6);cp=np.array([1.,2.,3.,3.,2.,1.]);v,b=fit_tacs(cp[None],cp,s,d,tstar=0)
    assert v[0]==1 and b[0]==0

def test_reject_invalid_model_cases():
    s=np.arange(6,dtype=float);d=np.ones(6);cp=np.ones(6)
    with pytest.raises(ValueError):fit_tacs(np.zeros((1,6)),cp,s,d,tstar=0)
    with pytest.raises(ValueError):fit_tacs(np.ones((1,6)),cp,s,d,tstar=5)
    with pytest.raises(ValueError):fit_tacs(np.ones((1,6)),cp,s,d,tstar=0,weight_by="unknown")

def test_loading_invalid(tmp_path):
    p=tmp_path/"bad.npz";np.savez(p,activity=np.zeros((1,27,1,200,200),np.float32))
    with pytest.raises(ValueError):load_data(p)

def test_independent_metric_cases():
    a=np.array([1.,3.]);b=np.array([2.,4.]);m=judge.metrics(a,b)
    assert m["nrmse"]==.5 and m["ncc"]==pytest.approx(14/np.sqrt(200))
    assert judge.metrics(np.zeros(2),b)["ncc"]==0
    with pytest.raises(ValueError):judge.metrics(a,np.ones(2))

def test_independent_observation_metric():
    a,m,cp,s,d,raw=load_data();v,_=reconstruct(a,m,cp,s,d)
    expected,_,_=linear_relative_residual(v,a,m,cp,s,d)
    assert judge.observation_residual(v[None],raw)==pytest.approx(expected,rel=1e-13)

@pytest.mark.parametrize("kind",["missing","key","extra","shape","dtype","complex","nan","inf","exterior","object","broken"])
def test_judge_invalid(tmp_path,kind):
    p=tmp_path/"submission.npz"
    if kind=="missing":pass
    elif kind=="key":np.savez(p,wrong=np.zeros(judge.SHAPE))
    elif kind=="extra":np.savez(p,reconstruction=np.zeros(judge.SHAPE),extra=np.array([1]))
    elif kind=="shape":np.savez(p,reconstruction=np.zeros((200,200)))
    elif kind=="dtype":np.savez(p,reconstruction=np.zeros(judge.SHAPE,np.float32))
    elif kind=="complex":np.savez(p,reconstruction=np.zeros(judge.SHAPE,np.complex128))
    elif kind=="object":np.savez(p,reconstruction=np.array([{}],object))
    elif kind=="broken":p.write_bytes(b"broken archive")
    else:
        a=np.zeros(judge.SHAPE);a[0,0,0,0]=np.nan if kind=="nan" else np.inf if kind=="inf" else 1e-12;np.savez(p,reconstruction=a)
    r=judge.evaluate(p);assert r["status"]=="error" and not r["passed"] and r["metrics"]=={} and r["errors"]

def test_judge_zero_poor_quality(tmp_path):
    p=tmp_path/"poor.npz";np.savez(p,reconstruction=np.zeros(judge.SHAPE));r=judge.evaluate(p)
    assert r["status"]=="fail" and r["metrics"]["ncc"]==0

def test_judge_missing_assets_and_invalid_config(tmp_path):
    assert judge.evaluate(ROOT/"output/reconstruction.npz",tmp_path)["status"]=="error"
    (tmp_path/"metrics.json").write_text('{"ncc_boundary":NaN,"nrmse_boundary":0.1,"late_linear_relative_weighted_residual_boundary":0.1}')
    assert judge.evaluate(ROOT/"output/reconstruction.npz",tmp_path)["status"]=="error"

def test_foreign_cwd_cli_and_report_failure(tmp_path):
    import subprocess,sys,json
    def run(p,r):return subprocess.run([sys.executable,str(ROOT/"evaluation/judge.py"),"--prediction",str(p),"--report",str(r)],cwd=tmp_path,capture_output=True)
    report=tmp_path/"nested/report.json";assert run(ROOT/"output/reconstruction.npz",report).returncode==0
    assert json.loads(report.read_text())["status"]=="pass"
    assert run(tmp_path/"missing.npz",report).returncode==2
    poor=tmp_path/"poor.npz";np.savez(poor,reconstruction=np.zeros(judge.SHAPE));assert run(poor,report).returncode==1
    blocker=tmp_path/"file";blocker.write_text("x");assert run(poor,blocker/"report.json").returncode==2

def test_main_import_backend():
    import matplotlib
    before=matplotlib.get_backend()
    import src.main
    assert matplotlib.get_backend()==before



def test_visual_metric_and_diagnostic_quadrature():
    from src.visualization import metrics as visual_metrics
    a=np.array([1.,3.]);b=np.array([2.,4.])
    assert visual_metrics(a,b)==judge.metrics(a,b)
    s=np.array([0.,1.,2.]);d=np.ones(3);curve=np.array([4.,5.,6.])
    assert np.array_equal(cumulative_integral(curve,s,d,"trapz_injection")-cumulative_integral(curve,s,d),np.ones(3))
