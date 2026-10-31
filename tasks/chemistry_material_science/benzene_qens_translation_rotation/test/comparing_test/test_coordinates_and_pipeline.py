from pathlib import Path
import ast,importlib.util
import numpy as np
from src.preprocessing import load_raw,prepare
from src.physics_model import predict
from src.main import run

ROOT=Path(__file__).resolve().parents[2]

def test_twenty_original_physical_adapter_cases():
    raw=load_raw(ROOT/'data/raw_data.npz');data=prepare(raw)
    with np.load(ROOT/'test/fixtures/qens_native_energy_affine_adapter_cases.npz') as f:
        native=f['parameters_native'];physical=f['parameters_physical']
        np.testing.assert_array_equal(physical[:,:2],native[:,:2]/1.5)
        np.testing.assert_array_equal(physical[:,[3,16]],native[:,[3,16]]*2.25)
        for n in data:
            edges=raw['energy_edges_'+n][0];true=(edges[:-1]+edges[1:])/2
            np.testing.assert_array_equal(edges,f[n+'_physical_energy_edges'])
            # Source sc.arange coordinates and original HDF5 edges have distinct
            # accumulation roundoff: prelocal independent capture measured
            # 8.526512829121202e-14 meV across the full 320-center axis.
            np.testing.assert_allclose(f[n+'_source_wrong_energy_centers'],1.5*true-np.diff(edges)[0]/4,rtol=0,atol=1e-13)
            np.testing.assert_allclose(f[n+'_native_predictions'],f[n+'_physical_predictions'],rtol=2e-11,atol=2e-10)
        for i,p in enumerate(physical):
            curves=predict(p,data)
            for n in data:np.testing.assert_allclose(curves[n],f[n+'_physical_predictions'][i],rtol=2e-11,atol=2e-10)

def test_exact_native_missingness_and_public_metadata():
    raw=load_raw(ROOT/'data/raw_data.npz');text=(ROOT/'README.md').read_text(encoding='utf8')
    for key in raw:assert '| '+key+' |' in text
    assert 'github' not in text.lower() and 'zenodo.org' not in text.lower()
    for n,count,qrows in [('360',45330,np.arange(6,18)),('197',45393,np.arange(7,21))]:
        for key in ('signal','standard_error'):
            a=raw[key+'_'+n];assert np.isnan(a).sum()==count and not np.isinf(a).any()
            np.testing.assert_array_equal(np.isfinite(a),raw['acquisition_mask_'+n])
        np.testing.assert_array_equal(prepare(raw)[n]['q_index'],qrows)
    assert '6 through 17; 7 through 20' in text

def test_active_fresh_full_pipeline_and_judge(tmp_path):
    output=tmp_path/'full/reconstruction.npz';result,diagnostics=run(ROOT/'data/raw_data.npz',output)
    assert diagnostics['weighted_sse']<=5661.294967552445 and diagnostics['grid_points']==27
    spec=importlib.util.spec_from_file_location('pipeline_judge',ROOT/'evaluation/judge.py');judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)
    report,code=judge.evaluate(output);assert code==0 and report['passed']
    assert set(result)=={'reconstruction','predicted_360','predicted_197'}

def test_runtime_call_roles_and_no_hidden_initialization():
    for path in (ROOT/'src').glob('*.py'):
        tree=ast.parse(path.read_text(encoding='utf8'))
        for node in ast.walk(tree):
            if isinstance(node,(ast.Import,ast.ImportFrom)):
                names=[a.name for a in node.names] if isinstance(node,ast.Import) else [node.module or '']
                assert not any(n.startswith(('scipp','dynesty','h5py','plet_data')) for n in names)
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):assert not (node.func.attr=='use' and isinstance(node.func.value,ast.Name) and node.func.value.id=='matplotlib')
    source=(ROOT/'src/solvers.py').read_text(encoding='utf8')
    for forbidden in ['np.load','baseline_reference','source_best','fixtures','pickle']:assert forbidden not in source
