from pathlib import Path
import subprocess,sys,json
T=Path(__file__).resolve().parents[2]
def test_complete_standard_cli(tmp_path):
 p=subprocess.run([sys.executable,'-m','src.main'],cwd=T,capture_output=True,text=True);assert p.returncode==0,p.stdout+p.stderr
 p=subprocess.run([sys.executable,str(T/'evaluation/judge.py'),'--prediction',str(T/'output/reconstruction.npz'),'--report',str(tmp_path/'report.json')],cwd=tmp_path,capture_output=True,text=True)
 assert p.returncode==0 and json.loads((tmp_path/'report.json').read_text())['passed']
