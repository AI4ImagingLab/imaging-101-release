from pathlib import Path
import subprocess,sys,json
T=Path(__file__).resolve().parents[2]
def test_native_cli_and_standalone_judge(tmp_path):
 p=subprocess.run([sys.executable,'-m','src.main'],cwd=T,capture_output=True,text=True)
 assert p.returncode==0,p.stdout+p.stderr
 report=tmp_path/'nested/report.json'
 p=subprocess.run([sys.executable,str(T/'evaluation/judge.py'),'--prediction',str(T/'output/reconstruction.npz'),'--report',str(report)],cwd=tmp_path,capture_output=True,text=True)
 assert p.returncode==0,p.stdout+p.stderr
 assert json.loads(report.read_text(encoding='utf8'))['passed'] is True
