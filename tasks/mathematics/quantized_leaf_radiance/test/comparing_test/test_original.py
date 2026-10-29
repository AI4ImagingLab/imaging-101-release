import numpy as np,pytest
from src.solvers import reconstruct,reconstruct_tensor

@pytest.mark.parametrize('method',['tv','tgv'])
@pytest.mark.parametrize('iterations',[0,1,2,5,30])
def test_all_original_states_cpu(oracle,method,iterations):
    _,_,state=reconstruct(oracle['input_lower'],oracle['input_upper'],method=method,iterations=iterations,return_state=True)
    for key,value in state.items():np.testing.assert_allclose(value,oracle[f'output_{method}_{iterations}_{key}'],rtol=2e-5,atol=3e-6)

@pytest.mark.parametrize('method',['tv','tgv'])
def test_original_states_tensor_cpu(oracle,method):
    _,state=reconstruct_tensor(oracle['input_lower'],oracle['input_upper'],method=method,iterations=30,device='cpu',return_state=True)
    for key,value in state.items():np.testing.assert_allclose(value,oracle[f'output_{method}_30_{key}'],rtol=2e-5,atol=3e-6)

def test_fixture_is_informative(oracle):
    assert len(np.unique(oracle['input_lower']))>=5
    assert np.abs(oracle['output_tgv_30_p']).max()>.1 and np.abs(oracle['output_tgv_30_q']).max()>.01

def test_full_cli_source_parity(root):
    # The standard full pipeline is independently executed by final CLI evidence;
    # test its actual native output, not a self-generated expected array.
    with np.load(root/'output/reconstruction.npz',allow_pickle=False) as f:actual=f['reconstruction'][0]
    with np.load(root/'notebooks/reference_outputs/upstream_outputs.npz',allow_pickle=False) as f:source=f['tgv']
    np.testing.assert_allclose(actual,source,rtol=2e-5,atol=3e-6)
    with np.load(root/'data/raw_data.npz',allow_pickle=False) as f:
        assert np.all(actual>=f['lower'][0]) and np.all(actual<=f['upper'][0])
