from pathlib import Path
import importlib.util,subprocess,sys
import numpy as np
T=Path(__file__).parents[2]
def test_standard_cli_and_independent_judge(tmp_path):
    output=tmp_path/'reconstruction.npz'
    r=subprocess.run([sys.executable,'-m','src.main','--output',str(output)],cwd=T,capture_output=True)
    assert r.returncode==0,r.stderr.decode('utf8',errors='replace')
    spec=importlib.util.spec_from_file_location('j',T/'evaluation/judge.py');j=importlib.util.module_from_spec(spec);spec.loader.exec_module(j)
    report,code=j.evaluate(output)
    assert code==0,report
