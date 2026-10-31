import importlib.util,subprocess,sys,shutil
from pathlib import Path
import numpy as np
import pytest

TASK=Path(__file__).parents[2]
spec=importlib.util.spec_from_file_location('judge',TASK/'evaluation/judge.py');judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)

def test_physical_ratios_and_bad_domain():
    ref=judge.validate(TASK/'output/reconstruction.npz')
    assert judge.physical_ratios(ref).max()<.001
    assert judge.physical_ratios(np.zeros_like(ref)).min()==1e100

@pytest.mark.parametrize('case',['file','key','shape','dtype','complex','nan','inf'])
def test_invalid(tmp_path,case):
    path=tmp_path/'invalid.npz';x=np.ones((1,4,2))
    if case=='file':pass
    elif case=='key':np.savez(path,wrong=x)
    else:
        if case=='shape':x=x[0]
        if case=='dtype':x=x.astype(np.float32)
        if case=='complex':x=x.astype(complex)
        if case in ['nan','inf']:x[0,0,0]=np.nan if case=='nan' else np.inf
        np.savez(path,reconstruction=x)
    report,code=judge.evaluate(path)
    assert code==2 and report['metrics']=={} and report['errors']

def test_all_rows_and_quantities(tmp_path):
    ref=judge.validate(TASK/'evaluation/baseline_reference.npz')
    for row in range(4):
        for column in range(2):
            x=ref.copy();x[0,row,column]*=1.5
            np.savez(tmp_path/'poor.npz',reconstruction=x)
            report,code=judge.evaluate(tmp_path/'poor.npz')
            assert code==1 and report['status']=='fail'

def test_cwd_assets_write(tmp_path):
    script=TASK/'evaluation/judge.py';prediction=TASK/'output/reconstruction.npz'
    command=[sys.executable,str(script),'--prediction',str(prediction),'--report',str(tmp_path/'report.json')]
    assert subprocess.run(command,cwd=tmp_path,capture_output=True).returncode==0
    command[-1]=str(tmp_path)
    assert subprocess.run(command,cwd=tmp_path,capture_output=True).returncode==2
    shutil.copyfile(script,tmp_path/'judge.py');command[1]=str(tmp_path/'judge.py');command[-1]=str(tmp_path/'missing.json')
    assert subprocess.run(command,cwd=tmp_path,capture_output=True).returncode==2
