"""Consumed original NPZ fixtures; never import the upstream implementation."""
import os
from pathlib import Path
import numpy as np
import pytest
from src.preprocessing import load_data,compute_attributes,compute_mgc
from src import physics_model as graph


@pytest.fixture(scope='module')
def original():
    here=Path(__file__).resolve().parents[1]
    with np.load(here/'fixtures/source_mgc.npz',allow_pickle=False) as z:mgc={k:z[k] for k in z.files}
    with np.load(here/'fixtures/source_graph_cases.npz',allow_pickle=False) as z:g={k:z[k] for k in z.files}
    raw=os.environ.get('GCL_PUBLIC_RAW_PATH')
    if raw is None:
        raw=next((p/'data/raw_data.npz' for p in Path(__file__).resolve().parents if (p/'data/raw_data.npz').is_file()),None)
    if raw is None:raise RuntimeError('public raw path required for private standalone tests')
    tiles=load_data(raw)
    return tiles,mgc,g


def eq(a,b):
    np.testing.assert_allclose(a,b,rtol=2e-12,atol=1e-14)


def test_full_original_attributes_and_all_mgc_channels(original):
    tiles,m,g=original;attributes=compute_attributes(tiles)
    for i,a in enumerate(attributes):
        for key,value in a.items():eq(value,m[f'edge{i}_{key}'])
    scores=compute_mgc(tiles,attributes)
    assert scores.dtype==m['score'].dtype==np.float32 and scores.shape==(432,432,16)
    assert np.array_equal(np.isposinf(scores),np.isposinf(m['score']))
    finite=np.isfinite(m['score']);x=scores[finite];y=m['score'][finite]
    ulp=abs(x.view(np.int32).astype(np.int64)-y.view(np.int32).astype(np.int64))
    assert np.max(ulp)<=2
    np.testing.assert_allclose(x,y,rtol=2*np.finfo(np.float32).eps,atol=0)
    assert np.all(np.isfinite(x)) and not np.isnan(scores).any()


def test_original_initial_graph_stages(original):
    _,m,g=original;scores=m['score']
    a,b,c=graph.pairwise_nearest(scores)
    for k,v in [('rot_nn',a),('pos_nn',b),('weights',c)]:eq(v,g['initial_'+k])
    a,b,c=graph.pairwise_four(scores)
    for k,v in [('rot_4',a),('pos_4',b),('W4',c)]:eq(v,g['initial_'+k])
    nn=(c>0).astype(float);nn=(nn+nn.T)/2;nn[nn==.5]=.01*8
    b[b==0]=g['initial_pos_nn'][b==0]
    tested=graph.initial_neighbors(nn,(nn>0)*b,scores);eq(tested,g['initial_nn_test'])
    jaccard=graph.jaccard_neighbors(tested);eq(jaccard,g['initial_jaccard'])
    eq(.8*jaccard+.2*tested,g['initial_nn_before_connect'])
    # Exactly one component means this fixture path has no RNG draw.
    assert len(np.unique(graph.component_labels(g['initial_nn_before_connect'])))==1
    state=graph.build_initial_graph(scores)
    for k in ['nn','rot','rot_orig','diag_nb']:eq(state[k],g['state_'+k])
    assert not state['nn4Mat_test'].flags.writeable


def test_original_sync_operator_and_common_phase(original):
    tiles,_,g=original
    _,nums,vector,A=graph.rotation_step(g['state_nn']+g['state_diag_nb']/2,g['state_rot'],tiles)
    eq(A,g['state_A'])
    d=np.sum(abs(g['state_rot']*(g['state_nn']+g['state_diag_nb']/2)),axis=1)
    u=vector*np.sqrt(d);v=g['state_rotations']*np.sqrt(d)
    overlap=abs(np.vdot(u,v))/(np.linalg.norm(u)*np.linalg.norm(v))
    assert overlap>1-1e-10
    assert len(np.unique((nums-g['state_rots_num'].astype(int))%4))==1


def test_original_full_reorder_and_legacy_criterion(original):
    _,_,g=original;scores=g['state_scores_current'];ids=g['reorder_ids']
    reordered=graph.reorder_scores(scores,g['state_rots_num'])
    eq(reordered[np.ix_(ids,ids,np.arange(16))],g['reordered_subarray'])
    assert graph.legacy_cost(scores[:,:,:4],g['state_goal'])==float(g['state_legacy_cost'])


@pytest.mark.parametrize('case',[-1,0,1])
def test_original_mismatch_and_hole_refinement(original,case):
    _,_,g=original
    prefix='state_' if case==-1 else f'hole{case}_'
    goal=g[prefix+'goal'];scores=g['state_scores_current']
    rem,connection,median=graph.find_mismatches(goal,scores,g['initial_nn_test'])
    eq(rem,g[prefix+('goal_rem' if case==-1 else 'rem')]);eq(connection,g[prefix+'connection'])
    eq(median,g[prefix+('median_distance' if case==-1 else 'median')])
    rot,nn=graph.fill_holes(rem,g['state_rot'],g['state_rots_num'],g['state_rot_orig'],connection,g['state_nn'],scores,median)
    eq(rot,g[prefix+'rot_fill']);eq(nn,g[prefix+'nn_fill'])
    if case==-1:
        nn,rot,diag=graph.four_loops(nn,rot)
        eq(nn,g['state_nn_refined']);eq(rot,g['state_rot_refined']);eq(diag,g['state_diag_refined'])
        state={'nn4Mat_test':g['initial_nn_test'],'nn':g['state_nn'],'rot':g['state_rot'],'rot_orig':g['state_rot_orig'],'diag_nb':g['state_diag_nb']}
        fresh=graph.refine_graph(goal,scores,state,g['state_rots_num'])
        for k,key in [('nn','nn_refined'),('rot','rot_refined'),('rot_orig','rot_refined'),('diag_nb','diag_refined')]:eq(fresh[k],g['state_'+key])
    else:assert np.count_nonzero(rem==0)==[2,6][case]
