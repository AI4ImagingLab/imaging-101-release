from pathlib import Path
import numpy as np
from src.preprocessing import load_data
from src.physics_model import coordinates,green_matrices,incident_field,prepare_fft,fft_A,fft_AH,cg_fft
from src.solvers import linear_kernel,tikhonov,reconstruct
ROOT=Path(__file__).resolve().parents[2]
def test_original_geometry_greens_incident_and_three_CG_steps():
    d=load_data(ROOT/'data/raw_data.npz');gs,gd=green_matrices(d['receiver_angles']);ei=incident_field(d['illumination_angles']);cache=prepare_fft();x,y=coordinates()
    with np.load(ROOT/'test/fixtures/oracle_stages.npz',allow_pickle=False) as f:
        for value,key in [(x,'x'),(y,'y'),(gs,'GS'),(gd,'GD'),(ei,'incident'),(d['measurements'],'measurement'),(np.cos(d['receiver_angles']),'receiver_x'),(np.sin(d['receiver_angles']),'receiver_y')]:np.testing.assert_array_equal(value,f['output_'+key])
        chi_expected=f['output_complex_contrasts'][0]
    with np.load(ROOT/'test/fixtures/extra_oracle_stages.npz',allow_pickle=False) as f:
        np.testing.assert_array_equal(cache['G'],f['output_extended_kernel'])
        np.testing.assert_allclose(fft_A(f['input_field'],f['input_contrast'],cache),f['output_A'],rtol=2e-8,atol=2e-9)
        np.testing.assert_allclose(fft_AH(f['input_field'],f['input_contrast'],cache),f['output_AH'],rtol=2e-8,atol=2e-9)
        field,errors=cg_fft(f['input_rhs'],f['input_contrast'],cache,maximum_iterations=3)
        np.testing.assert_allclose(field,f['output_CG3'],rtol=2e-8,atol=2e-9);np.testing.assert_allclose(errors,f['output_CG3_error'],rtol=2e-8,atol=2e-9)
        kernel=linear_kernel(gs,ei);np.testing.assert_allclose(kernel,f['output_first_kernel'],rtol=2e-8,atol=2e-9)
        np.testing.assert_allclose(tikhonov(kernel,d['measurements']),chi_expected,rtol=2e-8,atol=2e-9)
def test_fifteen_original_BIM_updates_and_full_native_result():
    out,stage=reconstruct(load_data(ROOT/'data/raw_data.npz'))
    with np.load(ROOT/'test/fixtures/oracle_stages.npz',allow_pickle=False) as f:
        for key in ['complex_contrasts','total_field_inputs','final_total_field']:np.testing.assert_allclose(stage[key],f['output_'+key],rtol=2e-8,atol=2e-9)
        np.testing.assert_allclose(stage['legacy_objective'],f['output_original_objective'],rtol=2e-8,atol=2e-9)
    with np.load(ROOT/'notebooks/reference_outputs/upstream.npz',allow_pickle=False) as f:np.testing.assert_allclose(out['reconstruction'],f['reconstruction'],rtol=1e-6,atol=1e-7)
    assert stage['state_residual'][-1]<2e-8
    assert stage['physical_residual'][-1]<.0255218106788596
