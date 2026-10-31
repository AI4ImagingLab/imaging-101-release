from pathlib import Path
import importlib.util
import json
import subprocess
import sys
import shutil
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('standalone',ROOT/'evaluation/judge.py')
judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)


def reference():
    with np.load(ROOT/'evaluation/baseline_reference.npz',allow_pickle=False) as f:return {k:f[k].copy() for k in f.files}


@pytest.mark.parametrize('case',['missing','key','shape','float32','complex','nan','inf','negative_amp','zero_width','phase_range','object'])
def test_reject_malformed(case,tmp_path):
    a=reference();p=tmp_path/'p.npz'
    if case=='missing':pass
    else:
        if case=='key':a.pop('linewidth_hz')
        elif case=='shape':a['phase_rad']=np.zeros((1,16))
        elif case=='float32':a['frequency_hz']=a['frequency_hz'].astype('float32')
        elif case=='complex':a['reconstruction']=a['reconstruction'].astype('complex128')
        elif case in ['nan','inf']:a['reconstruction'][0,0]=float(case)
        elif case=='negative_amp':a['reconstruction'][0,0]=-1
        elif case=='zero_width':a['linewidth_hz'][0,0]=0
        elif case=='phase_range':a['phase_rad'][0,0]=10
        elif case=='object':a['phase_rad']=a['phase_rad'].astype(object)
        np.savez(p,**a)
    result,code=judge.evaluate(p)
    assert code==2 and result['status']=='error' and result['metrics']=={} and result['errors']


def test_all_quantities_and_observations_are_evaluated(tmp_path):
    p=tmp_path/'p.npz';a=reference();np.savez(p,**a)
    result,code=judge.evaluate(p);assert code==0 and result['passed']
    for key,change in [('reconstruction',.1),('frequency_hz',.01),('linewidth_hz',.1),('phase_rad',.01)]:
        b=reference();b[key]+=change;np.savez(p,**b);result,code=judge.evaluate(p)
        assert code==1 and not result['passed']
    b=reference();b['reconstruction']*=0;np.savez(p,**b);result,code=judge.evaluate(p)
    assert code==1 and result['metrics']['observation_relative_l2']==pytest.approx(1.)


def test_evaluator_failure_and_caller_relative_cli(tmp_path):
    p=tmp_path/'p.npz';np.savez(p,**reference())
    result,code=judge.evaluate(p,tmp_path);assert code==2 and result['metrics']=={}
    command=[sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction','p.npz','--report']
    process=subprocess.run(command+['sub/report.json'],cwd=tmp_path,capture_output=True,text=True)
    assert process.returncode==0,process.stderr
    report=json.loads((tmp_path/'sub/report.json').read_text());assert report['passed']
    process=subprocess.run(command+['.'],cwd=tmp_path,capture_output=True,text=True)
    assert process.returncode==2 and 'Cannot write' in process.stderr
