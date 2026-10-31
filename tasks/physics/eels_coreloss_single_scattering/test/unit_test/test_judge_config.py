from pathlib import Path
import importlib.util,json,shutil
import pytest
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('config_guard_eels',ROOT/'evaluation/judge.py')
judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)

@pytest.mark.parametrize('case',['bool_true','bool_false','string','array','empty_array','nan','inf','negative','empty','missing','extra_metric','empty_comparisons','wrong_comparison'])
def test_malformed_evaluator_configuration(case,tmp_path):
    folder=tmp_path/'evaluation';folder.mkdir()
    shutil.copyfile(ROOT/'evaluation/baseline_reference.npz',folder/'baseline_reference.npz')
    config=json.loads((ROOT/'evaluation/metrics.json').read_text(encoding='utf-8'))
    path=folder/'metrics.json';path.write_text(json.dumps(config),encoding='utf-8')
    prediction=ROOT/'output/reconstruction.npz'
    original,code=judge.evaluate(prediction,asset_root=folder)
    assert code==0 and original['passed'],'The copied evaluator assets must pass before corruption'
    if case=='empty':config={}
    elif case=='missing':del config['nrmse_boundary']
    elif case=='extra_metric':config['unscored_boundary']=1.
    elif case=='empty_comparisons':config['comparisons']={}
    elif case=='wrong_comparison':config['comparisons']={'nrmse':'>=','relative_integral_error':'<='}
    else:
        config['nrmse_boundary']={'bool_true':True,'bool_false':False,'string':'0.00001','array':[.00001],'empty_array':[],'nan':float('nan'),'inf':float('inf'),'negative':-1.}[case]
    path.write_text(json.dumps(config),encoding='utf-8')
    report,code=judge.evaluate(prediction,asset_root=folder)
    assert code==2 and report['status']=='error' and not report['passed']
    assert report['metrics']=={} and report['errors']
    json.dumps(report,allow_nan=False)
