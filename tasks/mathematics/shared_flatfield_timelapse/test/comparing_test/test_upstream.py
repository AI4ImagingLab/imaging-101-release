from pathlib import Path
import json,importlib.util
import numpy as np
from src.preprocessing import load_observations,resize_stack
from src.solvers import step
from src.main import run
ROOT=Path(__file__).resolve().parents[2]

def test_original_all100_observation_resampling():
    raw=load_observations(ROOT/"data/raw_data.npz")
    with np.load(ROOT/"test/fixtures/upstream_first_update.npz",allow_pickle=False) as z:
        expected=z["input_Im"][:,0]
    actual=resize_stack(raw,(128,128))
    np.testing.assert_allclose(actual,expected,rtol=2e-5,atol=.001)

def check_update(name):
    with np.load(ROOT/"test/fixtures"/name,allow_pickle=False) as z:
        args=[z[f"input_{i}"] for i in range(12)]
        actual=step(z["input_Im"][:,0],z["input_W"][:,0],args[2][0],
                    args[6][:,0],args[7],args[8][:,0],args[9],
                    np.float32(args[9]*1e7),5.)
        for i,j,convert in [(0,1,lambda a:a[0]),(1,2,lambda a:a[0]),
                            (2,5,lambda a:a[:,0]),(3,6,lambda a:a[:,0]),
                            (4,7,lambda a:a),(5,8,lambda a:a[:,0]),
                            (6,9,lambda a:a),(7,10,lambda a:a[:,0])]:
            np.testing.assert_allclose(actual[i],convert(z[f"output_{j}"]),rtol=2e-5,atol=.001)

def test_original_first_proximal_update():check_update("upstream_first_update.npz")
def test_original_nondegenerate_late_proximal_update():check_update("upstream_late_update.npz")

def test_complete_original_automatic_selection_and_final100frame_fit():
    result=run(ROOT)
    expected=json.loads((ROOT/"test/fixtures/upstream_search_costs.json").read_text(encoding="utf-8"))
    assert len(result["search"])==len(expected)==13
    assert result["smoothness"]==5
    with np.load(ROOT/"test/fixtures/upstream_search_fields.npz",allow_pickle=False) as z:
        for i,(actual,source) in enumerate(zip(result["search"],expected)):
            assert actual["smoothness_flatfield"]==source["smoothness_flatfield"]
            assert abs(actual["cost"]-source["cost"])<=.004
            ref=z[f"flatfield_{i}"]
            assert np.sqrt(np.mean((result["search_fields"][i]-ref)**2))/np.ptp(ref)<.002
            np.testing.assert_allclose(result["search_mainfit_baselines"][i],z[f"baseline_mainfit_{i}"],rtol=2e-5,atol=.001)
    with np.load(ROOT/"evaluation/baseline_reference.npz",allow_pickle=False) as z:
        assert np.sqrt(np.mean((result["flatfield"]-z["reconstruction"][0])**2))/np.ptp(z["reconstruction"])<.002
        assert np.sqrt(np.mean((result["baseline"]-z["baseline"][0])**2))<.01
    spec=importlib.util.spec_from_file_location("independent_judge",ROOT/"evaluation/judge.py")
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    report=module.evaluate(ROOT/"output/reconstruction.npz")
    assert report["passed"],report
