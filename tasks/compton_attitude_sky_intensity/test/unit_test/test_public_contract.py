from pathlib import Path
import importlib.util,json,zipfile,re
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[2]

def test_visible_arrays_and_acquisition_metadata():
    from src.physics_model import NSIDE,NPIX,PIXEL_AREA,NUMERICAL_ZERO
    text=(ROOT/'README.md').read_text(encoding='utf-8')
    assert NSIDE==16 and NPIX==3072 and PIXEL_AREA==4*np.pi/3072 and NUMERICAL_ZERO==1e-12
    assert 'HEALPix nside 16, RING' in text and '| Attitude bins | 278 |' in text
    assert '| Incident-energy band | 509–513 |' in text and '| Measured-energy band | 509–513 |' in text
    assert '60 equal-width bins' in text and 'Approximately 7,979,955' in text
    with zipfile.ZipFile(ROOT/'data/raw_data.npz') as archive:
        for name in archive.namelist():
            key=name.removesuffix('.npy')
            assert f'| `{key}` |' in text
            with archive.open(name) as handle:
                version=np.lib.format.read_magic(handle)
                reader=np.lib.format.read_array_header_1_0 if version==(1,0) else np.lib.format.read_array_header_2_0
                shape,fortran,dtype=reader(handle)
                assert shape[0]==1 and not fortran and not dtype.hasobject
    assert not re.search(r'github\.com|githubusercontent\.com|github\.io|git\.io|codeload\.',text,re.I)
    assert 'ground_truth.npz' not in text and 'baseline_reference.npz' not in text

def test_independent_nontrivial_metrics_match_visualization_and_judge(tmp_path):
    from src.visualization import image_metrics
    spec=importlib.util.spec_from_file_location('contract_judge',ROOT/'evaluation/judge.py')
    judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)
    # Known nonconstant arrays with a shape-contract compatible repeated case.
    ref=np.tile([1.,2.,3.],1024)[None].astype(np.float64)
    est=np.tile([2.,1.,4.],1024)[None].astype(np.float64)
    expected_ncc=16/np.sqrt(14*21)
    expected_nrmse=.5
    values=image_metrics(est,ref)
    assert values['ncc']==pytest.approx(expected_ncc)
    assert values['nrmse']==pytest.approx(expected_nrmse)
    directory=tmp_path/'evaluation';directory.mkdir()
    (directory/'metrics.json').write_bytes((ROOT/'evaluation/metrics.json').read_bytes())
    np.savez_compressed(directory/'baseline_reference.npz',reconstruction=ref)
    pred=tmp_path/'prediction.npz';np.savez_compressed(pred,reconstruction=est)
    report,code=judge.evaluate(pred,directory,ROOT/'data/raw_data.npz')
    assert code==1
    assert report['metrics']['ncc']==pytest.approx(expected_ncc)
    assert report['metrics']['nrmse']==pytest.approx(expected_nrmse)
