import os,importlib.util
from pathlib import Path
import numpy as np
import pytest
from src.visualization import (rotate_pose,compare_poses,render_tiles,dynamic_range_nrmse,plot_reconstructions)


def poses():
    p=np.array([(r,c) for r in range(18) for c in range(24)],np.int64)
    return p,np.arange(432,dtype=np.int64)%4


def test_exact_complete_render_and_global_rotation():
    p=np.array([(r,c) for r in range(2) for c in range(3)],np.int64);q=np.arange(6,dtype=np.int64)%4
    tiles=np.arange(6*3*3*3,dtype=np.uint16).reshape(6,3,3,3)
    image=render_tiles(tiles,p,q)
    assert image.dtype==np.uint16 and image.shape==(6,9,3)
    for k in range(4):
        a,b=rotate_pose(p,q,k)
        assert np.array_equal(render_tiles(tiles,a,b),np.rot90(image,k))
        assert compare_poses(a,b,p,q)['metrics']['joint_piece_accuracy']==1


def test_common_gauge_not_independent_metric_maxima():
    p,q=poses();a,_=rotate_pose(p,q,1)
    result=compare_poses(a,q,p,q)
    assert max(c['metrics']['position_accuracy'] for c in result['candidates'])==1
    assert max(c['metrics']['orientation_accuracy'] for c in result['candidates'])==1
    assert not result['passed']


def test_dynamic_range_formula_and_uint_overflow():
    assert dynamic_range_nrmse(np.array([65535,0],np.uint16),np.array([0,65535],np.uint16))==1
    np.testing.assert_allclose(dynamic_range_nrmse([2.,2.],[0.,2.]),2**-.5)
    with pytest.raises(ValueError):dynamic_range_nrmse([1,1],[1,1])
    with pytest.raises(ValueError):dynamic_range_nrmse([1,np.inf],[1,2])


def test_poses_and_render_malformed():
    p,q=poses();bad=p.copy();bad[-1]=bad[0]
    with pytest.raises(ValueError):compare_poses(bad,q,p,q)
    with pytest.raises(ValueError):compare_poses(p,q.astype(float),p,q)
    with pytest.raises(ValueError):compare_poses(p,q,p,q,joint_boundary=True)
    with pytest.raises(ValueError):render_tiles(np.zeros((3,2,2,3)),p,q)


@pytest.mark.parametrize('case',range(9))
def test_five_metrics_and_common_alignment_match_independent_judge(case):
    path=os.environ.get('GCL_JUDGE_PATH')
    if path is None:
        path=next((str(p/'evaluation/judge.py') for p in Path(__file__).resolve().parents if (p/'evaluation/judge.py').is_file()),None)
    if path is None:raise RuntimeError('Independent Judge path required for private comparison')
    spec=importlib.util.spec_from_file_location('independent_gcl_visual_metric_judge',path)
    judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)
    ref,rq=poses();p=ref.copy();q=rq.copy()
    if case<4:p,q=rotate_pose(p,q,case)
    elif case==4:q[:31]=(q[:31]+1)%4
    elif case==5:p[[0,431]]=p[[431,0]]
    elif case==6:p,_=rotate_pose(p,q,1)
    elif case==7:p[:,1]=23-p[:,1]
    else:
        q[:3]=(q[:3]+2)%4;p,q=rotate_pose(p,q,3)
    thresholds={name:{'comparison':'>=','boundary':.95} for name in ['joint_piece_accuracy','oriented_neighbor_accuracy']}
    chosen,candidates=judge.compare_common_gauges(p,q,ref,rq,thresholds)
    ours=compare_poses(p,q,ref,rq)
    assert ours['global_quarter_turn']==chosen['global_quarter_turn']
    assert ours['passed']==chosen['passed'] and ours['metrics']==chosen['metrics']
    assert ours['candidates']==candidates


def test_plot_uses_actual_supplied_arrays():
    import matplotlib.pyplot as plt
    p=np.array([(r,c) for r in range(2) for c in range(3)],np.int64);q=np.arange(6,dtype=np.int64)%4
    tiles=(np.arange(6*3*3*3,dtype=np.uint16)*100).reshape(6,3,3,3)
    a,b=rotate_pose(p,q,1)
    fig,report=plot_reconstructions(tiles,(a,b),(p,q),(p,q))
    assert report['source']['dynamic_range_nrmse']==report['fresh']['dynamic_range_nrmse']==0
    assert report['source']['metrics']['joint_piece_accuracy']==1
    fig.canvas.draw();plt.close(fig)
