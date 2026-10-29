import subprocess
import sys
from pathlib import Path
import numpy as np
import os

ROOT=Path(__file__).resolve().parents[2]


def test_cli_and_foreign_cwd_judge(tmp_path):
    environment=dict(os.environ,PYTHONIOENCODING='utf-8')
    subprocess.run([sys.executable,'-m','src.main'],cwd=ROOT,check=True,capture_output=True,
                   text=True,encoding='utf-8',env=environment)
    subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(ROOT/'output/reconstruction.npz'),
                    '--report',str(tmp_path/'nested/report.json')],cwd=tmp_path,check=True,capture_output=True,
                   text=True,encoding='utf-8',env=environment)


def test_readme_public_contract():
    text=(ROOT/'README.md').read_text(encoding='utf-8')
    for key in ['reference_image','deformed_image','nodes','elements']:
        assert '| '+key+' |' in text
    assert 'github' not in text.lower()
    assert '420' in text and '260' in text
    assert 'x right' in text and 'y up' in text
    with np.load(ROOT/'data/raw_data.npz',allow_pickle=False) as data:
        assert data['reference_image'].shape==(1,420,260)
        assert data['deformed_image'].dtype==np.float64
        assert data['elements'].dtype==np.int64
        np.testing.assert_array_equal(data['nodes'][0].min(axis=0),[50,50])
        np.testing.assert_array_equal(data['nodes'][0].max(axis=0),[210,370])
        for element in data['elements'][0]:
            corners=data['nodes'][0,element]
            np.testing.assert_array_equal(corners[1]-corners[0],[80,0])
            np.testing.assert_array_equal(corners[3]-corners[0],[0,80])
