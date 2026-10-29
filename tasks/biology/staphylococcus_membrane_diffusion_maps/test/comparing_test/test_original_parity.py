"""Frozen original numeric fixtures captured before local extraction."""
from pathlib import Path
import numpy as np
from numpy.testing import assert_allclose
from src.preprocessing import correct_bleaching, correlate_trace, prepare_stack, bin_labels
from src.physics_model import correlation_model
from src.solvers import fit_curve

FIX=Path(__file__).resolve().parents[1]/'fixtures'/'original_stages.npz'

def test_four_nondegenerate_original_stages():
    with np.load(FIX,allow_pickle=False) as a:
        for i in range(4):
            p=f'case{i}'
            corrected,details=correct_bleaching(a[p+'_trace'],return_details=True)
            assert_allclose(details['parameters'],a[p+'_popt'],rtol=1e-10,atol=1e-12)
            assert_allclose(corrected,a[p+'_corrected'],rtol=1e-10,atol=1e-12)
            curve=correlate_trace(corrected)
            assert_allclose(curve,a[p+'_curves'],rtol=1e-10,atol=1e-12)
            parameters,prediction=fit_curve(curve)
            assert_allclose(parameters,a[p+'_parameters'],rtol=1e-8,atol=1e-10)
            assert_allclose(prediction,a[p+'_fit'],rtol=1e-9,atol=1e-12)
            assert_allclose(correlation_model(curve[:,0],*parameters),prediction,rtol=0,atol=0)

def test_full_native_pipeline_parity():
    from src.solvers import reconstruct
    T=Path(__file__).resolve().parents[2]
    with np.load(T/'data/raw_data.npz',allow_pickle=False) as a:
        result=reconstruct(a['frames'],a['cell_labels'])
    with np.load(T/'notebooks/reference_outputs/upstream_measurements.npz',allow_pickle=False) as a:
        mask=a['valid_mask']
        for key in ('reconstruction','molecule_number','correlation_offset'):
            assert_allclose(result[key][0][mask],a[key][0][mask],rtol=1e-7,atol=1e-10)
            assert np.all(result[key][0][~mask]==0)

