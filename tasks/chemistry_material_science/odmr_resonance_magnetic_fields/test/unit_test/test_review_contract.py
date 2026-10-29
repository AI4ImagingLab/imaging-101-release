from pathlib import Path
import json,subprocess,sys,importlib.util
import numpy as np
import pytest
from src.generate_data import perturb_spectra
from src.physics_model import initial_guess
from src.solvers import fit
T=Path(__file__).resolve().parents[2]
def test_maintainer_noise_and_rng_isolation():
 data=np.full((100000,),.99);before=data.copy()
 np.random.seed(83);expected=np.random.random(4)
 np.random.seed(83);noise=perturb_spectra(data,.001,20261023)-data;after=np.random.random(4)
 np.testing.assert_array_equal(data,before);np.testing.assert_array_equal(after,expected)
 assert abs(noise.mean())<1e-5 and abs(noise.std()-.001)<1e-5
def test_chunk_partition_invariance():
 with np.load(T/'test/fixtures/upstream_stages.npz',allow_pickle=False) as z:
  data=z['spectra'][0,0];freq=z['frequency_ghz'][0];initial=initial_guess(data,freq)
  full=fit(freq,data,initial,10000)[0]
  parts=np.concatenate([fit(freq,data[sl],initial[sl],10000)[0] for sl in [slice(0,7),slice(7,23),slice(23,None)]])
  np.testing.assert_array_equal(full,parts)
@pytest.mark.parametrize('case,expected',[('good',0),('bad',1),('malformed',2),('huge_finite',1)])
def test_native_judge_exit_codes(tmp_path,case,expected):
 with np.load(T/'evaluation/baseline_reference.npz',allow_pickle=False) as z:a=z['reconstruction']
 if case=='bad':a=a+1
 if case=='huge_finite':a=np.full_like(a,1e308)
 p=tmp_path/'p.npz';np.savez(p,**({'wrong':a} if case=='malformed' else {'reconstruction':a}))
 r=tmp_path/'r.json'
 child=subprocess.run([sys.executable,str(T/'evaluation/judge.py'),'--prediction',str(p),'--report',str(r)],cwd=tmp_path,capture_output=True)
 assert child.returncode==expected
 record=json.loads(r.read_text());assert record['status']=={0:'pass',1:'fail',2:'error'}[expected]
 assert 'NaN' not in r.read_text() and 'Infinity' not in r.read_text()
def test_visible_metadata_and_units():
 text=(T/'README.md').read_text(encoding='utf8')
 for literal in ['0.002158','0.000028024','(1, 4, 300, 480)','configuration 0: −1','configuration 1: +1']:assert literal in text
 assert not any(s in text for s in ['github.com','github.io','ground_truth.npz','baseline_reference.npz'])

