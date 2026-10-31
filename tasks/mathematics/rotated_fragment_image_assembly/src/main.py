"""Fresh full fragment assembly; inputs are public tile observations only."""
from pathlib import Path
import argparse
import time
import numpy as np
from .preprocessing import load_data, compute_mgc
from .physics_model import build_initial_graph, rotation_step, refine_graph, reorder_scores, legacy_cost
from .solvers import assemble_positions

GLOBAL_CCW_PERM=np.array([4,1,2,3,8,5,6,7,12,9,10,11,16,13,14,15])-1


def goal_to_positions(goal,n):
    g=np.asarray(goal)
    if g.size!=n or not np.array_equal(np.sort(g.ravel()),np.arange(1,n+1)):
        raise ValueError('assembly must place every piece exactly once')
    pos=np.empty((n,2),np.int64)
    for r,c in np.ndindex(g.shape):pos[int(g[r,c])-1]=(r,c)
    return pos


def reconstruct(tiles, refinement_iterations=5, inner_iterations=5, progress=None):
    """Execute initialization and all selected GCL/LP refinement rounds.

    Returns unbatched integer poses and numerical history. No truth is read.
    Default budgets reproduce the selected author experiment.
    """
    started=time.perf_counter();tiles=np.asarray(tiles);current=tiles.copy()
    scores=compute_mgc(tiles);state=build_initial_graph(scores)
    q=np.zeros(len(tiles),np.int64);history=[];goal=None;last=None
    for stage in range(refinement_iterations+1):
        if stage:state=refine_graph(goal,scores,state,last)
        current,last,mode,operator=rotation_step(state['nn']+state['diag_nb']/2,state['rot'],current)
        q=(q-last)%4;scores=reorder_scores(scores,last)
        goal0,trace0=assemble_positions(scores[:,:,:4],rows=18,cols=24,inner_iterations=inner_iterations)
        cost0=legacy_cost(scores[:,:,:4],goal0)
        turned=scores
        for _ in range(3):turned=turned[:,:,GLOBAL_CCW_PERM]
        goal1,trace1=assemble_positions(turned[:,:,:4],rows=18,cols=24,inner_iterations=inner_iterations)
        cost1=legacy_cost(turned[:,:,:4],goal1)
        selected=int(cost0>cost1 and goal1.size>=goal0.size)
        if selected:
            goal=goal1;scores=turned;q=(q+1)%4
            current=np.rot90(current,axes=(1,2)).copy()
        else:goal=goal0
        positions=goal_to_positions(goal,len(tiles))
        # Cumulative orientation is checked against actual pixels, not last mode.
        for i in range(len(tiles)):
            if not np.array_equal(current[i],np.rot90(tiles[i],int(q[i]))):
                raise ArithmeticError('cumulative fragment rotation disagrees with pixels')
        item={'stage':stage,'positions':positions.copy(),'quarter_turns':q.copy(),
              'goal':goal.copy(),'costs':np.array([cost0,cost1]),'selected_branch':selected,
              'source_rot_nums':last.copy(),'lp_trace':(trace0,trace1),
              'elapsed_seconds':time.perf_counter()-started}
        history.append(item)
        if progress:progress(item)
    return positions,q,history


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',default='data/raw_data.npz')
    parser.add_argument('--output',default='output/reconstruction.npz')
    args=parser.parse_args()
    def report(h):print(f"stage {h['stage']}/5 costs={h['costs'].tolist()} branch={h['selected_branch']} elapsed={h['elapsed_seconds']:.3f}s",flush=True)
    p,q,_=reconstruct(load_data(args.input),progress=report)
    dest=Path(args.output);dest.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(dest,positions=p[None],quarter_turns=q[None])


if __name__=='__main__':main()
