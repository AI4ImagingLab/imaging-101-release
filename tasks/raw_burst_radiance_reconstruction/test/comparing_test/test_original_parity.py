from pathlib import Path
import numpy as np
import torch
from src.preprocessing import load_data
from src.solvers import reconstruct,patchesRMS,mergeWienerDFTPatches,spatialDenoisePatches,NOISE
ROOT=Path(__file__).resolve().parents[2]
def test_corrected_original_all36_levels_motion_tiles_and_reconstruction():
    torch.set_num_threads(1);out,stages=reconstruct(load_data(ROOT/'data/raw_data.npz'))
    with np.load(ROOT/'evaluation/baseline_reference.npz',allow_pickle=False) as f:np.testing.assert_array_equal(out['reconstruction'],f['reconstruction'])
    with np.load(ROOT/'test/fixtures/oracle_stages.npz',allow_pickle=False) as f:
        for k in ['motion','aligned_sample','bayer_average']:np.testing.assert_array_equal(stages[k],f[k])
        assert len(stages['levels'])==36
        for i,v in enumerate(stages['levels']):np.testing.assert_array_equal(v,f['level_'+str(i)])
        for i,v in enumerate(stages['reference_pyramid']):np.testing.assert_array_equal(v,f['reference_level_'+str(i)])
        ref=f['aligned_sample'][0,...,0::2,0::2];alt=f['aligned_sample'][1,...,0::2,0::2]
        rms=patchesRMS(ref);np.testing.assert_array_equal(rms,f['rms_sample'])
        var=(NOISE[0]*rms[...,None,None]+NOISE[1]).repeat(16,-2).repeat(16,-1)
        fft=torch.fft.fftn(torch.from_numpy(ref.astype('float32')),dim=(2,3)).numpy();fa=torch.fft.fftn(torch.from_numpy(alt.astype('float32')),dim=(2,3)).numpy()
        np.testing.assert_array_equal(fft,f['ref_fft_sample'])
        np.testing.assert_allclose(mergeWienerDFTPatches(fft,fa,var,75),f['pair_fft_sample'],rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(spatialDenoisePatches(fft,var,.1),f['spatial_fft_sample'],rtol=1e-12,atol=1e-12)
