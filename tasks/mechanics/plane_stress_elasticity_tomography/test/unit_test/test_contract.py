from pathlib import Path
import re
import subprocess
import sys
import numpy as np
from src import preprocessing as constants

ROOT=Path(__file__).resolve().parents[2]

def test_public_schema_and_acquisition_constants_match_README():
    text=(ROOT/'README.md').read_text(encoding='utf8')
    for heading in ('Background','Problem Description','Data Description','Method Hints','References'):
        assert f'## {heading}' in text
    table=text.split('**Acquisition metadata**')[0].split('## Data Description')[1]
    with np.load(ROOT/'data/raw_data.npz',allow_pickle=False) as data:
        assert set(data.files)=={'nodes','triangles','displacement','nodal_force','fixed_dofs'}
        for key in data.files:
            row=next(line for line in table.splitlines() if line.startswith(f'| {key} |'))
            shape=row.split('|')[2].strip().replace(' ','')
            assert shape==str(data[key].shape).replace(' ','')
            assert row.split('|')[3].strip()==str(data[key].dtype)
        assert data['nodal_force'].max()==.00275
    for label,value in [('Plate thickness h',constants.THICKNESS),('Poisson ratio nu',constants.POISSON_RATIO),('Boundary multiplier c',constants.BOUNDARY_MULTIPLIER),('Relative noise coefficient',constants.RELATIVE_NOISE),('Global noise coefficient',constants.GLOBAL_NOISE)]:
        row=next(line for line in text.splitlines() if line.startswith(f'| {label} |'))
        assert float(row.split('|')[2].strip())==value
    assert not re.search(r'github\.com|githubusercontent\.com|github\.io|git\.io|codeload\.github',text,re.I)

def test_importing_main_does_not_change_matplotlib_backend():
    result=subprocess.run([sys.executable,'-c','import matplotlib; before=matplotlib.get_backend(); import src.main; assert matplotlib.get_backend()==before'],cwd=ROOT,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
