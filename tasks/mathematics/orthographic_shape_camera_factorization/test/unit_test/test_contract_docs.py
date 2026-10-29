from pathlib import Path
import importlib.util
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('judge_contract',ROOT/'evaluation/judge.py')
j=importlib.util.module_from_spec(spec);spec.loader.exec_module(j)

def test_readme_visible_contract():
    text=(ROOT/'README.md').read_text(encoding='utf-8')
    assert all(h in text for h in ['## Background','## Problem Description','## Data Description','## Method Hints','## References'])
    for token in ['(1, 51, 491, 2)','(1, 491, 3)','(1, 51, 2, 3)','(1, 51, 2)','pixel','No additional scale']:
        assert token in text
    for token in ['github.com','github.io','githubusercontent.com','git.io','ground_truth.npz','baseline_reference.npz']:
        assert token not in text
    assert not any(ord(c)<32 and c not in '\n\r\t' for c in text)

def test_judge_metrics_and_projection_known_cases():
    ncc,error=j.compare(np.array([.5,1.]),4.,np.array([1.,2.]))
    assert ncc==pytest.approx(1)
    assert error==pytest.approx(np.sqrt(2.5))
    shape=np.array([[1.,2.,3.]])
    camera=np.array([[[1.,0.,0.],[0.,1.,0.]]])
    translation=np.array([[4.,5.]])
    assert j.projection_rmse(shape,camera,translation,np.array([[[5.,7.]]]))==pytest.approx(0.,abs=1e-14)
