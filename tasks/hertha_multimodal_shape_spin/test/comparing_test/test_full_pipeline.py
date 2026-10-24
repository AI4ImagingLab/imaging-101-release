from pathlib import Path
import subprocess,sys,os,json

ROOT=Path(__file__).resolve().parents[2]

def test_complete_standard_cli_and_independent_foreign_directory_judge(tmp_path):
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONIOENCODING='utf-8')
    result=subprocess.run([sys.executable,'-X','utf8','-m','src.main'],cwd=ROOT,env=env,capture_output=True,text=True,encoding='utf-8',timeout=1200)
    assert result.returncode==0,result.stderr
    prediction=ROOT/'output/reconstruction.npz';assert prediction.is_file()
    report=ROOT/'output/judge_report.json'
    result=subprocess.run([sys.executable,'-X','utf8',str(ROOT/'evaluation/judge.py'),'--prediction',str(prediction),'--report',str(report)],cwd=tmp_path,env=env,capture_output=True,text=True,encoding='utf-8',timeout=120)
    assert result.returncode==0,result.stderr
    measured=json.loads(report.read_text(encoding='utf8'));assert measured['passed'] and len(measured['metrics'])==10
