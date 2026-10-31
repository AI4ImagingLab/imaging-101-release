from pathlib import Path
import importlib.util,json,subprocess,sys
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location("judge",ROOT/"evaluation/judge.py")
judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)

@pytest.mark.parametrize("bad",[True,False,"1",[],{},None,float("nan"),float("inf"),-1])
def test_malformed_boundary(bad):
    config=json.loads((ROOT/"evaluation/metrics.json").read_text())
    config["objective_counts_boundary"]=bad
    with pytest.raises(ValueError):judge.thresholds(config)

@pytest.mark.parametrize("bad",[{},[],{"unknown_boundary":1}])
def test_incomplete_configuration(bad):
    with pytest.raises(ValueError):judge.thresholds(bad)

@pytest.mark.parametrize('mode',['extra','direction','empty_comparisons','extra_comparison'])
def test_exact_metric_set_and_comparisons(mode):
    config=json.loads((ROOT/'evaluation/metrics.json').read_text(encoding='utf8'))
    if mode=='extra':config['unknown_boundary']=123
    else:
        config['comparisons']={name:'<=' for name in judge.NAMES}
        if mode=='direction':config['comparisons']['objective_counts']='>='
        elif mode=='empty_comparisons':config['comparisons']={}
        else:config['comparisons']['unknown']='<='
    with pytest.raises(ValueError):judge.thresholds(config)

@pytest.mark.parametrize("mode",["missing_key","shape","integer","complex","float16","nan","inf"])
def test_malformed_prediction(tmp_path,mode):
    arrays=dict(reconstruction=np.ones((1,128,128)),baseline=np.ones((1,100)))
    if mode=="missing_key":del arrays["baseline"]
    elif mode=="shape":arrays["baseline"]=np.ones((100,))
    elif mode=="integer":arrays["baseline"]=np.ones((1,100),dtype=int)
    elif mode=="complex":arrays["baseline"]=np.ones((1,100),dtype=complex)
    elif mode=="float16":arrays["baseline"]=np.ones((1,100),dtype=np.float16)
    else:arrays["baseline"][0,0]=np.nan if mode=="nan" else np.inf
    p=tmp_path/"bad.npz";np.savez(p,**arrays)
    with pytest.raises(ValueError):judge.read_prediction(p)

def test_missing_prediction(tmp_path):
    with pytest.raises(ValueError):judge.read_prediction(tmp_path/"absent.npz")

def test_nonpositive_physical_output():
    values,errors=judge.score(dict(reconstruction=np.zeros((128,128)),baseline=np.ones(100)),None)
    assert not values and errors

def test_finite_extreme_outputs_are_physical_failures_before_overflow():
    for s,b in [(np.full((128,128),1e300),np.ones(100)),(np.ones((128,128)),np.full(100,1e300))]:
        values,errors=judge.score(dict(reconstruction=s,baseline=b),None)
        assert values=={} and errors

def test_true_metric_analytic_case():
    n=np.arange(128);x=np.outer(np.cos(np.pi*(n+.5)*3/128),np.ones(128))
    np.testing.assert_allclose(np.sum(judge.resampling_matrix(1040).toarray(),axis=1),1,atol=3e-7)

def test_foreign_cwd_missing_cli_and_report_failure(tmp_path):
    p=tmp_path/"nested/report.json"
    args=[sys.executable,str(ROOT/"evaluation/judge.py"),"--prediction",str(tmp_path/"absent.npz"),"--report",str(p)]
    completed=subprocess.run(args,cwd=tmp_path,capture_output=True,text=True,encoding="utf-8")
    assert completed.returncode==2
    report=json.loads(p.read_text());assert report["status"]=="error" and report["metrics"]=={}
    blocked=tmp_path/"blocked";blocked.write_text("file")
    args[-1]=str(blocked/"report.json")
    completed=subprocess.run(args,cwd=tmp_path,capture_output=True,text=True,encoding="utf-8")
    assert completed.returncode==2 and "Cannot write report" in completed.stderr

@pytest.mark.parametrize("mode",["identity","constant_baseline","reverse_baseline"])
def test_inadequate_valid_outputs_fail(tmp_path,mode):
    with np.load(ROOT/"evaluation/baseline_reference.npz",allow_pickle=False) as z:
        s=z["reconstruction"].copy();b=z["baseline"].copy()
    if mode=="identity":s[:]=1
    elif mode=="constant_baseline":b[:]=np.mean(b)
    else:b=b[:,::-1]
    p=tmp_path/"poor.npz";np.savez(p,reconstruction=s,baseline=b)
    report=judge.evaluate(p)
    assert report["status"]=="fail" and not report["passed"]
