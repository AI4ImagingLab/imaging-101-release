from pathlib import Path
import numpy as np
from src.physics_model import FourierGrid,SHGFrog,abs2
from src.preprocessing import load_observations
from src.solvers import Retriever
T=Path(__file__).resolve().parents[2]

def relative(a,b,tolerance=1e-9):
    assert np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-300)<tolerance

def test_original_forward_projection_gradient():
    f=dict(np.load(T/'test/fixtures/original_stages.npz',allow_pickle=False));d,m=load_observations(T/'data/raw_data.npz');g=m.grid;s,A,E,H=m.signal(f['input_case'],f['input_delay_indices']);p=m.project(d[f['input_delay_indices']]/f['case_mu'],s)
    for a,b in [(E,f['output_field']),(g.forward(E),f['output_roundtrip']),(H,f['output_delay_mask']),(s,f['output_signal']),(A,f['output_delayed_field']),(abs2(g.forward(s)),f['output_selected_trace']),(p,f['output_projection']),(m.gradient(p,s,A,E,H),f['output_signal_gradient'])]:relative(a,b,1e-10)
    original_case=Retriever(m,d,f['input_case']);relative(np.array(original_case.mu),f['case_mu']);relative(np.array(original_case.current_max),f['case_initial_max_gradient'])

def test_original_local_global_and_pie_updates():
    f=dict(np.load(T/'test/fixtures/original_stages.npz',allow_pickle=False));d,m=load_observations(T/'data/raw_data.npz')
    for method,mode,prefix,initial in [('copra','local','local',f['input_initial']),('copra','global','global',f['local_spectrum']),('pie',None,'pie',f['input_initial'])]:
        ret=Retriever(m,d,initial,method=method);_,out=ret.step(mode=mode,capture=True)
        for k,v in out.items():
            if prefix+'_'+k in f and np.asarray(v).dtype.kind!='b':relative(np.asarray(v),f[prefix+'_'+k])

def test_original_old_input_best_save():
    f=dict(np.load(T/'test/fixtures/original_stages.npz',allow_pickle=False));d,m=load_observations(T/'data/raw_data.npz')
    for method in ['copra','pie']:
        result=Retriever(m,d,f['input_initial'],method=method).run(1);relative(result['spectrum'],f[method+'_one_iteration_saved_best']);relative(np.array(result['actual_error']),f[method+'_one_iteration_actual_error'])

def test_complete_pipeline_against_independent_original():
    from src.main import reconstruct
    e,result,model=reconstruct(T/'data/raw_data.npz')
    with np.load(T/'notebooks/reference_outputs/upstream_regen_seed1234.npz',allow_pickle=False) as f:
        relative(result['spectrum'],f['spectrum'],1e-7)
        np.testing.assert_allclose(result['trace_error'],f['trace_error'],rtol=1e-6,atol=1e-9)
    np.testing.assert_allclose(np.sum(abs(e)**2)*model.grid.dt,1.,rtol=1e-12)
