"""Explicit GCL graph construction, synchronization and hole refinement.

Source conventions follow Huroyan's selected published GCL implementation.
Internal goal grids contain one-based piece IDs and zero for holes.
"""
import numpy as np
from scipy.linalg import svd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

ROOTS=np.array([1,1j,-1,-1j],np.complex128)


def pairwise_nearest(scores):
    n=len(scores);best=np.argmin(scores,axis=2)
    pos=ROOTS[best%4];rot=np.zeros((n,n),complex);weights=np.zeros((n,n))
    ii,jj=np.triu_indices(n,1)
    weights[ii,jj]=scores[ii,jj,best[ii,jj]]
    rot[ii,jj]=ROOTS[best[ii,jj]//4]
    return rot+rot.conj().T,pos,weights+weights.T


def pairwise_four(scores):
    n=len(scores);weights=np.zeros((n,n));rot=np.zeros((n,n),complex);pos=np.zeros((n,n),complex)
    for i in range(n):
        for k in range(4):
            x=scores[i,:,k::4]
            ordered=np.sort(x.reshape(-1));indices=np.argmin(x,axis=0)
            vals=x[indices,np.arange(4)];r=int(np.argmin(vals));j=int(indices[r]);value=float(vals[r])
            if weights[i,j]==0:
                if ordered[0]!=ordered[1]:
                    weights[i,j]=value;pos[i,j]=ROOTS[k];rot[i,j]=ROOTS[r]
                    distinct=np.unique(scores[i,j])
                    if distinct[0]/distinct[1]>.95:
                        second=int(np.flatnonzero(scores[i,j]==distinct[1])[0])
                        rot[i,j]=(1.1*rot[i,j]+.9*ROOTS[second//4])/4
            elif value<weights[i,j] and ordered[0]!=ordered[1]:
                weights[i,j]=value;pos[i,j]=ROOTS[k];rot[i,j]=ROOTS[r]
    return (rot+rot.conj().T)/2,pos,weights


def initial_neighbors(nn,positions,scores):
    result=nn.copy()
    for i in range(len(nn)):
        neighbors=np.flatnonzero(nn[:,i]);x=positions[i,neighbors]
        for j in range(len(neighbors)-1):
            for k in range(j+1,len(neighbors)):
                if x[j]==x[k]:
                    edge=int(np.flatnonzero(ROOTS==x[j])[0]);a,b=neighbors[j],neighbors[k]
                    va=np.min(scores[i,a,edge::4]);vb=np.min(scores[i,b,edge::4])
                    if va<vb:
                        result[i,b]=result[b,i]=0;result[i,a]*=.5;result[a,i]*=.5
                    elif va>vb:
                        result[i,a]=result[a,i]=0;result[i,b]*=.5;result[b,i]*=.5
                    else:
                        result[i,a]=result[a,i]=result[i,b]=result[b,i]=0
    return result


def jaccard_neighbors(nn):
    a=nn>0;out=np.zeros(nn.shape)
    for i,j in zip(*np.nonzero(a)):
        b=a.copy();b[i,j]=b[j,i]=False
        x=np.flatnonzero(b[i]);y=np.flatnonzero(b[j])
        xx=set(x);yy=set(y)
        for k in x:xx.update(np.flatnonzero(b[k]))
        for k in y:yy.update(np.flatnonzero(b[k]))
        if len(xx&yy)>1:out[i,j]=1
    return out


def component_labels(nn):
    """Strong components, deterministic minimum-member label gauge."""
    _,label=connected_components(csr_matrix(nn!=0),directed=True,connection='strong')
    parts=sorted((np.flatnonzero(label==k) for k in np.unique(label)),key=lambda x:x[0])
    out=np.empty(len(nn),np.int64)
    for k,part in enumerate(parts):out[part]=k
    return out


def connect_graph(nn,weights,rng=None):
    out=nn.copy();w=weights.copy();np.fill_diagonal(w,np.inf)
    rng=np.random.default_rng() if rng is None else rng
    while True:
        labels=component_labels(out);sizes=np.bincount(labels);used=np.flatnonzero(labels==np.argmax(sizes))
        if len(used)==len(out):return out
        unused=np.setdiff1d(np.arange(len(out)),used);block=w[np.ix_(unused,used)]
        linear=np.flatnonzero(block.ravel(order='F')==np.min(block));chosen=int(rng.choice(linear))
        x,y=np.unravel_index(chosen,block.shape,order='F');i,j=unused[x],used[y]
        out[i,j]=out[j,i]=.001


def four_loops(nn,rot):
    nn=nn.copy();out=rot.copy();n=len(nn)
    neighbors=[np.flatnonzero(nn[:,i]) for i in range(n)]
    diag=np.zeros(nn.shape)
    for i in range(n):
        second=set()
        for k in neighbors[i]:second.update(np.flatnonzero(nn[k]))
        second.discard(i)
        for j in second:
            if len(np.intersect1d(neighbors[i],neighbors[j]))==2:diag[i,j]=1
    diag=(diag+diag.T)/2
    for i in range(n):
        for j in range(n):
            if diag[i,j]:
                x=np.intersect1d(neighbors[i],neighbors[j]);a,b=x
                u=rot[a,j]*rot[i,a];v=rot[b,j]*rot[i,b]
                if u!=0 and v!=0 and u/abs(u)==v/abs(v):out[i,j]=u
                else:
                    diag[i,j]=0
                    for x in (a,b):
                        nn[x,i]/=2;nn[i,x]/=2;nn[x,j]/=2;nn[j,x]/=2
    nn=nn/1.5
    for i in range(n):
        for j in range(n):
            if diag[i,j]:
                x=np.intersect1d(np.flatnonzero(nn[i]),np.flatnonzero(nn[j]))
                if len(x)==2:
                    for k in x:nn[i,k]=nn[k,i]=nn[j,k]=nn[k,j]=1
    return nn,out,diag


def build_initial_graph(scores16,rng=None):
    oldrot,oldpos,weights=pairwise_nearest(scores16)
    rot,pos,w=pairwise_four(scores16)
    nn=(w>0).astype(float);nn=(nn+nn.T)/2;nn[nn==.5]=.01*8
    pos[pos==0]=oldpos[pos==0]
    tested=initial_neighbors(nn,(nn>0)*pos,scores16)
    nn=.8*jaccard_neighbors(tested)+.2*tested
    nn=connect_graph(nn,weights,rng)
    rot[abs(rot)==0]=oldrot[abs(rot)==0]
    nn,rot,diag=four_loops(nn,rot)
    tested.setflags(write=False)
    return {'nn4Mat_test':tested,'nn':nn,'rot':rot,'rot_orig':rot.copy(),'diag_nb':diag}


def round_angles(rotations):
    angle=np.angle(rotations);angle[angle<0]+=2*np.pi
    # Preserve the source's strict midpoint comparisons and subtraction order.
    result=np.empty(angle.shape,np.int64)
    for i,a in enumerate(angle):
        if a<np.pi/2:result[i]=0 if a<np.pi/2-a else 1
        elif a<np.pi:result[i]=1 if a-np.pi/2<np.pi-a else 2
        elif a<3*np.pi/2:result[i]=2 if a-np.pi<3*np.pi/2-a else 3
        else:result[i]=3 if a-3*np.pi/2<2*np.pi-a else 0
    return result


def rotation_step(weighted_nn,rot,current_pieces):
    """Smallest left singular mode; physical patch action is rot90(-r)."""
    connection=rot*weighted_nn;degree=np.sum(abs(connection),axis=1)
    if np.any(degree<=0):raise ValueError('positive graph degrees required')
    lap=np.diag(degree)-connection;invroot=degree**-.5
    normalized=invroot[:,None]*lap*invroot[None,:]
    u,_,_=svd(normalized,full_matrices=False,check_finite=True)
    rotations=invroot*u[:,-1]
    nums=round_angles(rotations)
    new=np.stack([np.rot90(tile,-int(r)) for tile,r in zip(current_pieces,nums)])
    return new,nums,rotations,lap/degree[:,None]


_PERMS=np.array([
 [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16],
 [13,14,15,16,1,2,3,4,5,6,7,8,9,10,11,12],
 [9,10,11,12,13,14,15,16,1,2,3,4,5,6,7,8],
 [5,6,7,8,9,10,11,12,13,14,15,16,1,2,3,4],
 [8,5,6,7,12,9,10,11,16,13,14,15,4,1,2,3],
 [4,1,2,3,8,5,6,7,12,9,10,11,16,13,14,15],
 [16,13,14,15,4,1,2,3,8,5,6,7,12,9,10,11],
 [12,9,10,11,16,13,14,15,4,1,2,3,8,5,6,7],
 [11,12,9,10,15,16,13,14,3,4,1,2,7,8,5,6],
 [7,8,5,6,11,12,9,10,15,16,13,14,3,4,1,2],
 [3,4,1,2,7,8,5,6,11,12,9,10,15,16,13,14],
 [15,16,13,14,3,4,1,2,7,8,5,6,11,12,9,10],
 [14,15,16,13,2,3,4,1,6,7,8,5,10,11,12,9],
 [10,11,12,9,14,15,16,13,2,3,4,1,6,7,8,5],
 [6,7,8,5,10,11,12,9,14,15,16,13,2,3,4,1],
 [2,3,4,1,6,7,8,5,10,11,12,9,14,15,16,13]])-1


def reorder_scores(scores16,source_rot_nums):
    r=np.asarray(source_rot_nums,dtype=np.int64).reshape(-1)
    indices=_PERMS[4*r[:,None]+r[None,:]]
    return np.take_along_axis(scores16,indices,axis=2)


def legacy_cost(scores4,goal):
    """Historical biased boundary convention, used only for source selection."""
    g=np.asarray(goal,dtype=np.int64)-1;h,w=g.shape;cost=np.float32(0)
    for i in range(h):
        for j in range(w):
            if j>0:cost=cost+scores4[g[i,j],g[i,j-1],3]
            if i+1<h-1:cost=cost+scores4[g[i,j],g[i+1,j],2]
            if j+1<w-1:cost=cost+scores4[g[i,j],g[i,j+1],1]
            if i>0:cost=cost+scores4[g[i,j],g[i-1,j],0]
    return float(cost)


def find_mismatches(goal,scores_current,nn_test):
    g=np.asarray(goal,dtype=np.int64);n=len(scores_current);h,w=g.shape
    # columns: all, trb, lrb, tlb, tlr; original biased top condition retained.
    dist=np.full((n,5),-1.)
    connection=np.zeros((n,n))
    loc=np.full((n,2),-1,int)
    for i in range(h):
        for j in range(w):
            v=g[i,j]-1;loc[v]=i,j
            left=scores_current[v,g[i,j-1]-1,3] if j>0 else np.float32(0)
            right=scores_current[v,g[i,j+1]-1,1] if j<w-1 else np.float32(0)
            top=scores_current[v,g[i-1,j]-1,0] if i>0 else np.float32(0)
            bottom=scores_current[v,g[i+1,j]-1,2] if i<h-1 else np.float32(0)
            tl,tr,tt,tb=int(j>0),int(j<w-1),int(i>0),int(i<h-1)
            # Maintain source single summation order before division.
            dist[v]=[(left+right+top+bottom)/(tl+tr+tt+tb),(right+top+bottom)/(tr+tt+tb),
                     (left+right+bottom)/(tl+tr+tb),(left+top+bottom)/(tl+tt+tb),
                     (left+right+top)/(tl+tr+tt)]
            for di,dj,ok in [(0,-1,j>0),(-1,0,i>0),(0,1,j+1<w-1),(1,0,i+1<h-1)]:
                if ok:
                    other=g[i+di,j+dj]-1;connection[v,other]=connection[other,v]=1
    c=(connection*nn_test)>0;cx=c.copy()
    med=np.array([np.median(d[d>=0]) for d in dist.T])
    for v in range(n):
        i,j=loc[v]
        if dist[v,0]>1.5*med[0]:c[v,:]=False;c[:,v]=False
        for col,dirs in [(1,[(-1,0),(0,1),(1,0)]),(2,[(0,-1),(0,1),(1,0)]),
                         (3,[(-1,0),(0,-1),(-1,0),(0,-1)]),(4,[(-1,0),(0,-1),(0,1)])]:
            if dist[v,col]>1.5*med[col]:
                for di,dj in dirs:
                    if 0<=i+di<h and 0<=j+dj<w:cx[v,g[i+di,j+dj]-1]=False
    cx=cx|cx.T
    # bool+bool in NumPy is logical; MATLAB here performs a numeric sum.
    labels=component_labels((c.astype(float)+cx.astype(float))/2)
    keep=labels==np.argmax(np.bincount(labels));rem=g.copy();rem[~keep[g-1]]=0
    return rem,connection,float(np.median(dist[:,0]))


def corner_choice(i,j,goal,scores):
    n=len(scores);cost=np.zeros((4,n))
    for di,dj,edge in [(1,0,0),(0,-1,1),(-1,0,2),(0,1,3)]:
        if 0<=i+di<goal.shape[0] and 0<=j+dj<goal.shape[1] and goal[i+di,j+dj]:
            v=goal[i+di,j+dj]-1
            for r in range(4):cost[r]+=scores[v,:,edge+4*r]
    where=np.flatnonzero(cost.ravel(order='F')==np.min(cost))
    if len(where)!=1:raise ValueError('source corner minimizer is nonunique; unsupported vector assignment')
    r,piece=np.unravel_index(where[0],cost.shape,order='F')
    return int(piece),int(r),float(np.min(cost))


def fill_holes(goal_rem,rot,last_nums,rot_orig,connection,nn,scores,median_distance):
    rem=np.sort(goal_rem[goal_rem!=0])-1;t=ROOTS[np.asarray(last_nums,dtype=int)]
    rot=t.conj()[:,None]*rot*t[None,:];original=t.conj()[:,None]*rot_orig*t[None,:]
    con=np.zeros(nn.shape);con[np.ix_(rem,rem)]=connection[np.ix_(rem,rem)]
    nn=.6*nn+.4*con;rot_old=rot.copy()
    h,w=goal_rem.shape
    for i in range(h):
        for j in range(w):
            if goal_rem[i,j]:continue
            neighbors=[]
            for di,dj in [(1,0),(0,-1),(-1,0),(0,1)]:
                if 0<=i+di<h and 0<=j+dj<w and goal_rem[i+di,j+dj]:neighbors.append(goal_rem[i+di,j+dj]-1)
            if len(neighbors)<=1:continue
            piece,r,cost=corner_choice(i,j,goal_rem,scores)
            if r:
                rot[neighbors,piece]=ROOTS[r];rot[piece,neighbors]=ROOTS[r].conjugate()
            nb_count=np.count_nonzero(np.angle(rot[piece,neighbors])==np.angle(original[piece,neighbors]))
            if (nb_count<4 and len(neighbors)>1) or (nb_count==4 and len(neighbors)>2):
                if cost/len(neighbors)<median_distance:value=.6
                elif cost/len(neighbors)<2*median_distance:value=.3
                else:continue
                nn[neighbors,piece]=nn[piece,neighbors]=value
            else:
                rot[neighbors,piece]=rot_old[neighbors,piece];rot[piece,neighbors]=rot_old[piece,neighbors]
    mask=np.zeros(nn.shape,bool);mask[np.ix_(rem,rem)]=nn[np.ix_(rem,rem)]>0;rot[mask]=1
    return rot,nn


def refine_graph(goal,scores_current,state,last_source_rot_nums):
    rem,connection,median=find_mismatches(goal,scores_current,state['nn4Mat_test'])
    rot,nn=fill_holes(rem,state['rot'],last_source_rot_nums,state['rot_orig'],connection,state['nn'],scores_current,median)
    nn,rot,diag=four_loops(nn,rot)
    return {'nn4Mat_test':state['nn4Mat_test'],'nn':nn,'rot':rot,'rot_orig':rot.copy(),'diag_nb':diag}
