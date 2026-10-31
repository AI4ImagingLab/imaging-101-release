"""Independent numeric metric and strict evaluator contract cases."""
from pathlib import Path
import importlib.util,json,subprocess,sys,shutil
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('standalone_judge',ROOT/'evaluation/judge.py');judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)


def test_known_relative_l2():
    assert judge.relative_l2(np.array([3.,4.]),np.array([0.,5.]))==np.sqrt(10)/5


def test_prediction_and_evaluator_contracts(tmp_path):
    with np.load(ROOT/'evaluation/baseline_reference.npz',allow_pickle=False) as f:good={k:f[k] for k in f.files}
    path=tmp_path/'input.npz';np.savez(path,**good)
    assert judge.evaluate(path)[1]==0
    poor={k:v.copy() for k,v in good.items()};poor['translation']+=.001
    np.savez(path,**poor);report,code=judge.evaluate(path);assert code==1 and report['metrics'] and not report['passed']
    for change in ['missing','shape','dtype','nan','inf','complex','zero_scale']:
        bad={k:v.copy() for k,v in good.items()}
        if change=='missing':del bad['rotation']
        elif change=='shape':bad['reconstruction']=bad['reconstruction'][0]
        elif change=='dtype':bad['rotation']=bad['rotation'].astype('float32')
        elif change=='complex':bad['rotation']=bad['rotation'].astype('complex128')
        elif change=='zero_scale':bad['scale'][:]=0
        else:bad['rotation'][0,0,0]=np.nan if change=='nan' else np.inf
        np.savez(path,**bad);report,code=judge.evaluate(path)
        assert code==2 and report['metrics']=={} and report['errors'],change
    assert judge.evaluate(tmp_path/'missing.npz')[1]==2
    path.write_text('invalid NPZ',encoding='utf8');assert judge.evaluate(path)[1]==2
    assets=tmp_path/'assets';shutil.copytree(ROOT/'evaluation',assets)
    (assets/'baseline_reference.npz').unlink();assert judge.evaluate(path,assets)[1]==2
    # Actual truth, when present, takes precedence over a corrupted baseline.
    np.savez(path,**good);np.savez(assets/'ground_truth.npz',**good)
    (assets/'baseline_reference.npz').write_bytes(b'bad');assert judge.evaluate(path,assets)[1]==0
    (assets/'metrics.json').write_text('{}',encoding='utf8');assert judge.evaluate(path,assets)[1]==2


def test_foreign_cwd_and_write_failure(tmp_path):
    prediction=ROOT/'evaluation/baseline_reference.npz';report=tmp_path/'nested/report.json'
    command=[sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(prediction),'--report',str(report)]
    result=subprocess.run(command,cwd=tmp_path,capture_output=True,text=True)
    assert result.returncode==0 and json.loads(report.read_text(encoding='utf8'))['status']=='pass'
    block=tmp_path/'file';block.write_text('block',encoding='utf8');command[-1]=str(block/'report.json')
    result=subprocess.run(command,cwd=tmp_path,capture_output=True,text=True);assert result.returncode==2 and 'Cannot write report' in result.stderr
