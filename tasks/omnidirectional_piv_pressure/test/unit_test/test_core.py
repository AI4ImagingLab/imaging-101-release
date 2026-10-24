import importlib.util
from pathlib import Path
import numpy as np
import pytest
from src.preprocessing import load_data
from src.physics_model import spacing,weights,matrix_apply,build_rhs,edge_residual
from src.solvers import sequential_dot,solve
from src.generate_data import generate
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location("independent_judge",ROOT/"evaluation/judge.py");judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)

def test_loading_and_constants():
    g,q=load_data();assert g.shape==(2,1001,1001) and g.dtype==np.float32
    assert q[0]==-3 and q[-1]==3 and spacing(q)==np.float32(q[1]-q[0])
    assert weights(spacing(q))==(np.float32(.25),np.float32(.25))
    readme=(ROOT/"README.md").read_text(encoding="utf-8")
    for text in ["1001","0.001","0.006","(1, 2, 1001, 1001)","(1, 1001)"]:assert text in readme
    assert "github" not in readme.lower()

def test_loading_rejects_invalid(tmp_path):
    p=tmp_path/"bad.npz";np.savez(p,pressure_gradient=np.zeros((1,2,3,3)),coordinate=np.zeros((1,3)))
    with pytest.raises(ValueError):load_data(p)

def test_constant_nullspace_and_energy():
    p=np.ones((13,17),np.float32);assert np.array_equal(matrix_apply(p,.1),np.zeros_like(p))
    rng=np.random.default_rng(15);p=rng.normal(size=p.shape).astype(np.float32);q=rng.normal(size=p.shape).astype(np.float32)
    ap=matrix_apply(p,.1);aq=matrix_apply(q,.1)
    assert abs(np.sum(p.astype(float)*aq)-np.sum(q.astype(float)*ap))<2e-6
    energy=.25*(np.sum(np.diff(p.astype(float),axis=1)**2)+np.sum(np.diff(p.astype(float),axis=0)**2))
    assert np.sum(p.astype(float)*ap)==pytest.approx(energy,rel=2e-7)

def test_affine_gradient_boundary_and_residual():
    y,x=np.mgrid[:11,:13].astype(np.float32);h=np.float32(.125);p=.5*x*h-.25*y*h
    g=np.stack([np.full_like(x,.5),np.full_like(y,-.25)])
    assert np.allclose(matrix_apply(p,h),build_rhs(g,h),atol=1e-7)
    rx,ry=edge_residual(p,g,h);assert np.max(abs(rx))==0 and np.max(abs(ry))==0
    assert abs(float(np.sum(build_rhs(g,h),dtype=float)))<1e-12

def test_zero_input():
    p,history=solve(np.zeros((2,9,9),np.float32),np.linspace(-1,1,9,dtype=np.float32))
    assert not p.any() and history.shape==(0,3)

def test_metrics_known_case():
    a=np.array([1.,3.]);b=np.array([2.,4.]);m=judge.metrics(a,b)
    assert m["nrmse"]==.5 and m["ncc"]==pytest.approx(14/np.sqrt(200))
    assert judge.metrics(np.zeros(2),b)["ncc"]==0
    with pytest.raises(ValueError):judge.metrics(a,np.ones(2))

def test_generator_preserves_observations():
    g,q,p=generate();a,b=load_data()
    assert np.array_equal(g,a) and np.array_equal(q,b)
    with np.load(ROOT/"evaluation/ground_truth.npz") as z:assert np.array_equal(p,z["reconstruction"][0])

def test_analytic_gradient_grid_convergence():
    errors=[]
    for n in [129,257,1001]:
        g,q,p=generate(n);x,y=np.meshgrid(q.astype(float)*.001,q.astype(float)*.001)
        t=.3;nu=1e-6;C=1000*1e-12/(64*np.pi**2*nu*t**3)
        e=np.exp(-(x*x+y*y)/(2*nu*t));a=np.stack([C*e*x/(nu*t),C*e*y/(nu*t)])
        errors.append(np.linalg.norm(g-a)/np.linalg.norm(a))
    assert errors[1]<errors[0]/3.9 and errors[2]<errors[1]/14
    assert errors[2]<2.5e-5

@pytest.mark.parametrize("kind",["missing","key","shape","dtype","nan","inf","complex","gauge","pickle","broken"])
def test_judge_invalid_inputs(tmp_path,kind):
    path=tmp_path/"input.npz"
    if kind=="missing":pass
    elif kind=="key":np.savez(path,wrong=np.zeros(judge.EXPECTED_SHAPE))
    elif kind=="shape":np.savez(path,reconstruction=np.zeros((1001,1001)))
    elif kind=="dtype":np.savez(path,reconstruction=np.zeros(judge.EXPECTED_SHAPE,np.float32))
    elif kind=="complex":np.savez(path,reconstruction=np.zeros(judge.EXPECTED_SHAPE,np.complex128))
    elif kind=="pickle":np.savez(path,reconstruction=np.array([{}],object))
    elif kind=="broken":path.write_bytes(b"not NPZ")
    else:
        p=np.zeros(judge.EXPECTED_SHAPE);p[0,0,0]=np.nan if kind=="nan" else np.inf if kind=="inf" else 0
        if kind=="gauge":p[0,1,1]=.001
        np.savez_compressed(path,reconstruction=p)
    r=judge.evaluate(path);assert r["status"]=="error" and not r["passed"] and r["metrics"]=={} and r["errors"]

def test_judge_poor_quality(tmp_path):
    p=tmp_path/"poor.npz";np.savez_compressed(p,reconstruction=np.zeros(judge.EXPECTED_SHAPE));r=judge.evaluate(p)
    assert r["status"]=="fail" and not r["passed"] and r["metrics"]["ncc"]==0

def test_judge_evaluator_assets(tmp_path):
    r=judge.evaluate(ROOT/"output/reconstruction.npz",tmp_path);assert r["status"]=="error" and not r["metrics"]
    (tmp_path/"metrics.json").write_text('{"ncc_boundary":NaN,"nrmse_boundary":0.1}')
    r=judge.evaluate(ROOT/"output/reconstruction.npz",tmp_path);assert r["status"]=="error"

def test_cli_exit_codes_and_report_failure(tmp_path):
    import subprocess,sys,json
    def run(p,r):return subprocess.run([sys.executable,str(ROOT/"evaluation/judge.py"),"--prediction",str(p),"--report",str(r)],capture_output=True,cwd=tmp_path)
    report=tmp_path/"nested/report.json";assert run(ROOT/"output/reconstruction.npz",report).returncode==0
    assert json.loads(report.read_text())["status"]=="pass"
    assert run(tmp_path/"missing.npz",report).returncode==2
    poor=tmp_path/"poor.npz";np.savez_compressed(poor,reconstruction=np.zeros(judge.EXPECTED_SHAPE));assert run(poor,report).returncode==1
    blocker=tmp_path/"file";blocker.write_text("x");assert run(poor,blocker/"report.json").returncode==2

def test_visual_metrics_match_known_independent_case():
    from src.visualization import metrics
    a=np.array([1.,3.]);b=np.array([2.,4.])
    assert metrics(a,b)==judge.metrics(a,b)
