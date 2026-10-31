import ast,importlib,inspect
from pathlib import Path
import numpy as np
from src.preprocessing import get_patch_edges
from src.solvers import patches,constrained_oasisAR1
import pytest

ROOT=Path(__file__).resolve().parents[2]

def test_metadata_grid_and_patch_count_match_runtime():
    text=(ROOT/'README.md').read_text(encoding='utf-8')
    assert '| Frame rate | 30 | Hz |' in text
    assert '| Frame dimensions | 170 by 170 | pixel |' in text
    assert '| Background rank | 2 | component |' in text
    assert '(1, 3000, 170, 170)' in text
    edges=get_patch_edges((170,170),(24,24),(50,50))
    assert [list(x) for x in edges]==[[0,50,96],[0,50,96]]
    assert len(list(patches((170,170))))==49

def test_no_forbidden_runtime_import_or_backend_mutation():
    forbidden={'caiman','oasis','sklearn','skimage','peakutils','tifffile','h5py'}
    for file in [*(ROOT/'src').glob('*.py'),ROOT/'evaluation/judge.py']:
        tree=ast.parse(file.read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):assert all(n.name.split('.')[0] not in forbidden for n in node.names)
            if isinstance(node,ast.ImportFrom) and node.module:assert node.module.split('.')[0] not in forbidden
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):assert not (node.func.attr=='use' and isinstance(node.func.value,ast.Name) and node.func.value.id=='matplotlib')

def test_selected_oasis_scope_is_enforced():
    for options in [{'penalty':0},{'decimate':2},{'optimize_g':2}]:
        with pytest.raises(ValueError):constrained_oasisAR1(np.ones(10,np.float32),.9,.1,**options)
