"""Explicit selected Yu LP/component/greedy/trim-fill mathematics.

Adapted from the author-provided PuzzleDemoLP and Gallagher routines at
c957fe5a70a0e153e6ba93440dd13144f9945818. Source authors: Rui Yu et al.,
Andrew Gallagher and Vahan Huroyan. No redistribution license is inferred;
this private extraction is for the authorized local research review.

The input scores refer to patches already oriented by the outer GCL stage.
Only four directional layers, no replicated pieces, and fixed current rotations
are selected. There are no truth, seeds, upstream imports, or disk reads here.
"""
from __future__ import annotations
import warnings
import numpy as np
from scipy import sparse, ndimage
from scipy.sparse.csgraph import connected_components
from scipy.optimize import linprog, OptimizeWarning

OFFSETS = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]], dtype=np.int64)
OPPOSITE = (2, 3, 0, 1)


def normalize_scores(scores):
    """Author row/column alternative-best ratios, preserving NaN/Inf masks."""
    s = np.asarray(scores)
    if s.ndim != 3 or s.shape[0] != s.shape[1] or s.shape[2] != 4:
        raise ValueError('Expected square N by N by four directional scores')
    if s.dtype not in (np.dtype('float32'), np.dtype('float64')):
        raise ValueError('Scores must be real floating point')
    n = len(s)
    if n < 2:
        raise ValueError('At least two pieces are required')
    result = s.copy()
    tiny = np.asarray(1e-15, dtype=s.dtype)
    for side in range(4):
        a = s[:, :, side]
        ri = np.argsort(a, axis=1, kind='stable')
        ci = np.argsort(a, axis=0, kind='stable')
        rs = np.take_along_axis(a, ri[:, :2], axis=1)
        cs = np.take_along_axis(a, ci[:2, :], axis=0)
        with np.errstate(invalid='ignore', divide='ignore', over='ignore'):
            for i in range(n):
                # The multiplication is intentional: source Inf*0 gives NaN.
                first = a[i] * 0 + rs[i, 0]
                first[ri[i, 0]] = rs[i, 1]
                second = a[i] * 0 + cs[0]
                mask = ci[0] == i
                second[mask] = cs[1, mask]
                result[i, :, side] = (a[i] + tiny) / (np.fmin(first, second) + tiny)
    return result


def candidate_matches(scores, k=1):
    """Selected probabilistic mode2; .8 threshold is unused by that source mode."""
    if k != 1:
        raise ValueError('This selected method uses one candidate per side')
    norm = normalize_scores(scores)
    dtype = norm.dtype
    tiny = np.asarray(1e-15, dtype=dtype)
    rows = []
    for side in range(4):
        a = norm[:, :, side]
        ids = np.argsort(a, axis=1, kind='stable')[:, 0]
        values = a[np.arange(len(a)), ids]
        with np.errstate(invalid='ignore', over='ignore'):
            weights = np.maximum(np.exp(-np.asarray(5, dtype=dtype) * values ** 2), tiny)
        for i in range(len(a)):
            if not np.isnan(values[i]) and weights[i] != np.float32(1e-15):
                rows.append((i + 1, ids[i] + 1, weights[i], side + 1))
    return np.asarray(rows, dtype=dtype).reshape(-1, 4)


def anchor_for_matches(info, n):
    """A data-derived best-connected piece fixes only an arbitrary translation."""
    weights = np.zeros((n, 4), dtype=np.float64)
    exists = np.zeros((n, 4), dtype=bool)
    for i, j, weight, side in info:
        a, d = int(i) - 1, int(side) - 1
        if weight != 1 and not exists[a, d]:
            exists[a, d] = True
            weights[a, d] = weight
    total = np.where(exists.all(axis=1), weights.sum(axis=1), 0.)
    a = int(np.argmax(total))
    return np.array([a + 1, a + 1 + n]), np.array([-100000., -100000.])


def build_lp(n, info, fix_id, fix_value, previous_info=None):
    """Weighted L1 displacements with free row/column coordinates.

    z=[rows,columns,positive_residuals,negative_residuals]. Each match gives
    unit displacement along its stated direction and zero orthogonal offset.
    Both residual signs are nonnegative, expressed by explicit -I inequalities.
    """
    info = np.asarray(info)
    if previous_info is not None and len(previous_info):
        info = np.concatenate((info, np.asarray(previous_info)), axis=0)
    info = np.unique(info, axis=0).astype(np.float64)
    if info.ndim != 2 or info.shape[1] != 4 or not len(info):
        raise ValueError('LP needs a nonempty four-column match table')
    m = len(info)
    i, j, side = info[:, 0].astype(int) - 1, info[:, 1].astype(int) - 1, info[:, 3].astype(int)
    if np.any((i < 0) | (i >= n) | (j < 0) | (j >= n)) or np.any((side < 1) | (side > 4)):
        raise ValueError('Invalid match identities or directions')
    if not np.isfinite(info).all() or np.any(info[:, 2] <= 0):
        raise ValueError('Invalid match weights')
    horizontal = np.isin(side, [2, 4]).astype(int)
    sign = np.where(np.isin(side, [2, 3]), -1., 1.)
    rr = np.concatenate((np.arange(m), np.arange(m), np.arange(m) + m, np.arange(m) + m))
    cc = np.concatenate((i + n * horizontal, j + n * horizontal, i + n * (1-horizontal), j + n * (1-horizontal)))
    vv = np.concatenate((sign, -sign, sign, -sign))
    position = sparse.coo_matrix((vv, (rr, cc)), shape=(2*m, 2*n)).tocsr()
    eye = sparse.eye(2*m, format='csr')
    equality = sparse.hstack((position, -eye, eye), format='csr')
    fix_id = np.asarray(fix_id, dtype=int).reshape(-1) - 1
    fix_value = np.asarray(fix_value, dtype=float).reshape(-1)
    if len(fix_id) != len(fix_value) or np.any((fix_id < 0) | (fix_id >= 2*n)):
        raise ValueError('Invalid translation anchor')
    anchors = sparse.coo_matrix((np.ones(len(fix_id)), (np.arange(len(fix_id)), fix_id)), shape=(len(fix_id), 2*n+4*m)).tocsr()
    equality = sparse.vstack((equality, anchors), format='csr')
    rhs = np.concatenate((np.ones(m), np.zeros(m), fix_value))
    inequalities = sparse.hstack((sparse.csr_matrix((4*m, 2*n)), -sparse.eye(4*m)), format='csr')
    w = np.tile(info[:, 2], 2)
    objective = np.concatenate((np.zeros(2*n), w, w))
    return objective, inequalities, np.zeros(4*m), equality, rhs


def solve_lp(f, A, b, E, d):
    """General LP primitive; feasibility/objective, not a nonunique basis, matter."""
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=OptimizeWarning, message='Unrecognized options detected.*')
        result = linprog(f, A_ub=A, b_ub=b, A_eq=E, b_eq=d,
                         bounds=(None, None), method='highs-ds',
                         options={'threads': 1, 'primal_feasibility_tolerance': 1e-8,
                                  'dual_feasibility_tolerance': 1e-8})
    if not result.success or result.x is None or not np.isfinite(result.x).all():
        raise RuntimeError('Position LP failed: '+str(result.message))
    x = result.x
    diag = dict(objective=float(f@x), equality_max=float(np.max(np.abs(E@x-d))),
                inequality_max=float(max(0., np.max(A@x-b))), status=int(result.status))
    if diag['equality_max'] > 2e-6 or diag['inequality_max'] > 2e-6:
        raise RuntimeError('Position LP returned an infeasible solution')
    return x, diag


def _crop_empty(block):
    occupied = np.argwhere(block > 0)
    if not len(occupied):
        return np.zeros((0, 0), dtype=np.int64)
    lo, hi = occupied.min(0), occupied.max(0) + 1
    return np.asarray(block[lo[0]:hi[0], lo[1]:hi[1]], dtype=np.int64)


def recover_components(x, info, n):
    """Recover unit-lattice, collision-free four-connected estimated blocks."""
    info = np.asarray(info, dtype=np.float64).reshape(-1, 4)
    xy = np.asarray(x[:2*n]).reshape(2, n).T
    first, second = info[:, 0].astype(int)-1, info[:, 1].astype(int)-1
    expected = OFFSETS[info[:, 3].astype(int)-1]
    good = np.all(np.abs((xy[second]-xy[first])-expected) < 1e-4, axis=1)
    kept = info[good]
    r, c = kept[:, 0].astype(int)-1, kept[:, 1].astype(int)-1
    graph = sparse.coo_matrix((np.ones(len(r)*2), (np.r_[r,c], np.r_[c,r])), shape=(n,n)).tocsr()
    _, initial = connected_components(graph, directed=False)
    blocks = []
    for lab in np.unique(initial):
        ids = np.flatnonzero(initial == lab)
        if len(ids) == 1:
            continue
        coords = xy[ids]
        lo, hi = coords.min(0), coords.max(0)
        shape = np.floor(hi-lo+1+.5).astype(int)
        if np.any(shape < 1) or np.any(shape > n):
            raise RuntimeError('Invalid connected LP component extent')
        block = np.zeros(tuple(shape), dtype=np.int64)
        for rr in range(shape[0]):
            for cc in range(shape[1]):
                found = ids[np.all(np.abs(coords-(lo+[rr,cc])) < .1, axis=1)]
                block[rr,cc] = int(found[0])+1 if len(found)==1 else -len(found)
        mask, count = ndimage.label(block > 0, structure=np.array([[0,1,0],[1,1,1],[0,1,0]]))
        # Match source column-major bwlabel ordering, without using true positions.
        parts = sorted(range(1,count+1), key=lambda k: np.flatnonzero((mask==k).ravel(order='F'))[0])
        for part in parts:
            candidate = _crop_empty(np.where(mask==part, block, 0))
            if candidate.size > 1:
                blocks.append(candidate)
    labels = np.full(n, -1, dtype=np.int64)
    for lab, block in enumerate(blocks, 1):
        labels[block[block>0]-1] = lab
    return labels, blocks, kept


def update_scores(scores, blocks, labels):
    """Mask already established block interiors and finite-grid boundaries."""
    scores = np.array(scores, copy=True)
    n = len(scores)
    for lab in np.unique(labels[labels>=1]):
        ids = np.flatnonzero(labels==lab)
        scores[np.ix_(ids,ids,np.arange(4))] = np.inf
    for g in blocks:
        full_size = g.size == n
        for side in range(4):
            if side==0: a,b,border=g[1:],g[:-1],g[0]
            elif side==1: a,b,border=g[:,:-1],g[:,1:],g[:,-1]
            elif side==2: a,b,border=g[:-1],g[1:],g[-1]
            else: a,b,border=g[:,1:],g[:,:-1],g[:,0]
            valid=(a>0)&(b>0)
            i,j=a[valid]-1,b[valid]-1
            if len(i):
                same=labels[i]==labels[j];i,j=i[same],j[same]
                scores[i,:,side]=np.inf; scores[:,i,OPPOSITE[side]]=np.inf
                scores[:,j,side]=np.inf; scores[j,:,OPPOSITE[side]]=np.inf
            if full_size:
                ids=border[border>0]-1
                scores[ids,:,side]=np.inf; scores[:,ids,OPPOSITE[side]]=np.inf
    return scores


def join_blocks(first, second, first_id, second_id, side):
    """Explicit integer translation/overlap test for the selected fixed rotations."""
    a,b=np.asarray(first,dtype=np.int64),np.asarray(second,dtype=np.int64)
    if side not in (1,2,3,4): raise ValueError('Selected joining has four fixed-orientation directions')
    if np.intersect1d(a[a>0],b[b>0]).size:
        return b.copy(),False
    pa=np.argwhere(a==first_id);pb=np.argwhere(b==second_id)
    if len(pa)!=1 or len(pb)!=1: raise ValueError('Joining key must occur once')
    shift=pa[0]+OFFSETS[side-1]-pb[0]
    low=np.minimum([0,0],shift);high=np.maximum(a.shape,np.asarray(b.shape)+shift)
    result=np.zeros(tuple(high-low),dtype=np.int64)
    oa=-low;ob=shift-low
    result[oa[0]:oa[0]+a.shape[0],oa[1]:oa[1]+a.shape[1]]=a
    target=result[ob[0]:ob[0]+b.shape[0],ob[1]:ob[1]+b.shape[1]]
    target[b>0]=b[b>0]
    success=np.count_nonzero(result)==np.count_nonzero(a)+np.count_nonzero(b)
    return result,bool(success)


def greedy_assemble(blocks, normalized):
    """Original greedy merge, including its one final over-threshold attempt."""
    blocks=[np.asarray(b,dtype=np.int64).copy() for b in blocks]
    if not blocks: raise RuntimeError('LP produced no nontrivial component')
    cost=np.array(normalized,copy=True);n=len(cost);go=True;attempts=0
    while go:
        flat=cost.ravel(order='F')
        if np.all(np.isnan(flat)): value=np.nan;idx=0
        else: idx=int(np.nanargmin(flat));value=flat[idx]
        if np.isnan(value) or value>1.25: go=False
        i,j,side=np.unravel_index(idx,cost.shape,order='F');p1,p2=i+1,j+1
        own1=next((k for k,g in enumerate(blocks) if np.any(g==p1)),None)
        own2=next((k for k,g in enumerate(blocks) if np.any(g==p2)),None)
        a=blocks[own1] if own1 is not None else np.array([[p1]])
        b=blocks[own2] if own2 is not None else np.array([[p2]])
        if own1 is not None and own1==own2: joined=b;ok=False
        else: joined,ok=join_blocks(a,b,p1,p2,side+1)
        if ok:
            if own1 is not None:
                blocks[own1]=joined
                if own2 is not None: blocks[own2]=np.zeros((0,0),dtype=np.int64)
            elif own2 is not None: blocks[own2]=joined
            else: blocks.append(joined)
            cost[i,:,side]=np.nan;cost[:,j,side]=np.nan
            other=OPPOSITE[side];cost[j,:,other]=np.nan;cost[:,i,other]=np.nan
            if a.size>1 or b.size>1:
                ai,bi=a[a>0]-1,b[b>0]-1
                cost[np.ix_(ai,bi,np.arange(4))]=np.nan;cost[np.ix_(bi,ai,np.arange(4))]=np.nan
            if np.count_nonzero(joined)>n-1: go=False
        else:
            cost[i,j,side]=np.nan;cost[j,i,OPPOSITE[side]]=np.nan
        attempts+=1
        if attempts>4*n*n+1: raise RuntimeError('Greedy merge failed to exhaust candidates')
    return blocks[int(np.argmax([np.count_nonzero(g) for g in blocks]))].copy()


def trim_block(block, rows, cols):
    """Author independent occupancy-marginal crop; no joint optimal-crop claim."""
    g=np.asarray(block,dtype=np.int64)
    if max(g.shape)<=max(rows,cols) and min(g.shape)<=min(rows,cols): return g.copy()
    rm=(g>0).sum(1);cm=(g>0).sum(0);start=[];chopped=0
    for marginal,length in [(rm,rows),(cm,cols)]:
        if len(marginal)>length:
            kept=np.convolve(marginal,np.ones(length,dtype=int),mode='valid')
            at=int(np.argmax(kept));chopped+=int(marginal.sum()-kept[at]);start.append(at)
        else:start.append(0)
    if chopped>0:return g[start[0]:start[0]+rows,start[1]:start[1]+cols].copy()
    return g.copy()


def fill_holes(block,scores,n):
    """Selected fixed-orientation greedy filling with explicit neighbor costs.

    Original per-hole score arrays are retained, so cumulative neighbor additions
    and tie order follow the author path. No original image or GT is read.
    """
    g=np.asarray(block,dtype=np.int64).copy();choices=np.setdiff1d(np.arange(1,n+1),g)
    if not len(choices):return g
    rr,cc=np.unravel_index(np.flatnonzero((g==0).ravel(order='F')),g.shape,order='F')
    holes=[];counts=[]
    for r,c in zip(rr,cc):
        count=sum(0<=r+dr<g.shape[0] and 0<=c+dc<g.shape[1] and g[r+dr,c+dc]>0 for dr,dc in OFFSETS)
        holes.append((int(r),int(c)));counts.append(count)
    order=np.argsort(-np.asarray(counts),kind='stable');holes=[holes[k] for k in order if counts[k]>0]
    def costs_for(hole):
        r,c=hole;terms=[]
        # Source neighbor summation order: top,left,right,bottom.
        for d in [0,3,1,2]:
            dr,dc=OFFSETS[d];r2,c2=r+dr,c+dc
            if 0<=r2<g.shape[0] and 0<=c2<g.shape[1] and g[r2,c2]>0:
                terms.append(np.asarray(scores[choices-1,g[r2,c2]-1,d],dtype=float))
        return np.sum(np.stack(terms,axis=1),axis=1) if terms else np.zeros(len(choices))
    costs=[costs_for(hole) for hole in holes]
    while holes and len(choices):
        ratios=[]
        for a in costs:
            ordered=np.sort(a)
            with np.errstate(invalid='ignore',divide='ignore'):
                ratios.append(ordered[0]/ordered[1] if len(a)>1 else ordered[0])
        at=int(np.argsort(np.asarray(ratios),kind='stable')[0]);r,c=holes[at]
        candidate_order=np.argsort(costs[at],kind='stable');success=False
        for k in candidate_order:
            piece=int(choices[k])
            # Joining at the requested empty cell is the selected joinPiecesR result.
            if g[r,c]==0 and piece not in g:
                g[r,c]=piece;success=True;break
        if not success:break
        del holes[at];del costs[at]
        choices=np.delete(choices,k);costs=[np.delete(a,k) for a in costs]
        if not len(choices):break
        for d,(dr,dc) in enumerate(OFFSETS):
            r2,c2=r+dr,c+dc
            if 0<=r2<g.shape[0] and 0<=c2<g.shape[1] and g[r2,c2]==0:
                hole=(int(r2),int(c2))
                if hole in holes:
                    q=holes.index(hole);costs[q]=costs[q]+scores[choices-1,piece-1,OPPOSITE[d]]
                else:
                    holes.append(hole);costs.append(costs_for(hole))
    return g


def assemble_positions(scores4,rows=18,cols=24,inner_iterations=5):
    """Return author 1-based placement grid and a numerical execution trace."""
    scores=np.asarray(scores4)
    n=rows*cols
    if scores.shape!=(n,n,4) or scores.dtype!=np.float32:
        raise ValueError('Expected native float32 (rows*cols,rows*cols,4) scores')
    if not np.isfinite(scores[~np.isinf(scores)]).all() or np.any(np.isneginf(scores)):
        raise ValueError('Invalid score values')
    if not isinstance(inner_iterations,int) or not 0<=inner_iterations<=5:
        raise ValueError('Selected inner budget is at most five')
    info=candidate_matches(scores);fix,val=anchor_for_matches(info,n)
    f,A,b,E,d=build_lp(n,info,fix,val);x,lp=solve_lp(f,A,b,E,d)
    labels,blocks,info=recover_components(x,info,n)
    trace=dict(lp_calls=1,lp=[lp],inner_iterations_executed=0,stop_reason='budget',component_sizes=[int(np.count_nonzero(g)) for g in blocks])
    current=scores.copy()
    for iteration in range(inner_iterations):
        current=update_scores(current,blocks,labels)
        proposed=candidate_matches(current)
        if not len(proposed):trace['stop_reason']='no_potential_matches';break
        previous_labels=labels.copy();previous_info=info.copy()
        f,A,b,E,d=build_lp(n,proposed,fix,val,previous_info)
        x,lp=solve_lp(f,A,b,E,d);trace['lp'].append(lp);trace['lp_calls']+=1
        combined=np.unique(np.vstack((proposed,previous_info)),axis=0)
        labels,blocks,info=recover_components(x,combined,n)
        trace['inner_iterations_executed']=iteration+1
        if np.array_equal(previous_labels,labels):trace['stop_reason']='unchanged_labels';break
    merged=greedy_assemble(blocks,normalize_scores(update_scores(scores,blocks,labels)))
    goal=fill_holes(trim_block(merged,rows,cols),scores,n)
    trace.update(final_shape=list(goal.shape),assigned_pieces=int(np.count_nonzero(goal)),holes=int(np.count_nonzero(goal==0)))
    return goal,trace
