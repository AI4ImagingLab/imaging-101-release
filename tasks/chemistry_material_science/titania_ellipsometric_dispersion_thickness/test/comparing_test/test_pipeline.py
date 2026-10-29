from pathlib import Path
import subprocess,sys,os,json

ROOT=Path(__file__).resolve().parents[2]

def test_standard_cli_then_independent_foreign_cwd_judge(tmp_path):
    environment=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    run=subprocess.run([sys.executable,'-m','src.main'],cwd=ROOT,env=environment,capture_output=True,text=True)
    assert run.returncode==0,run.stderr
    report=ROOT/'output/judge_report.json'
    judge=subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),
        '--prediction',str(ROOT/'output/reconstruction.npz'),'--report',str(report)],
        cwd=tmp_path,env=environment,capture_output=True,text=True)
    assert judge.returncode==0,judge.stderr
    measured=json.loads(report.read_text())
    assert measured['passed'] and measured['status']=='pass' and len(measured['metrics'])>=7

