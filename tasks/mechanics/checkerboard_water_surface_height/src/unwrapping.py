"""Explicit reliability-sorted Herraez phase unwrapping with weighted union-find.

Implements the published 2002 noncontinuous-path algorithm; Windows source
boundary reliability generator matches the captured scikit-image C reference.
No domain unwrapping library is imported.
"""
import numpy as np,time

def unwrap_reliability(phase):
    phase=np.asarray(phase,dtype='float64');m,n=phase.shape;total=m*n
    # Windows Universal C Runtime rand(), seeded with zero by the captured source.
    state=0;random=np.empty(total,dtype=float)
    for k in range(total):
        state=(214013*state+2531011)&0xffffffff;random[k]=(state>>16)&32767
    reliability=random.reshape(m,n)
    wrap=lambda d:np.where(d>np.pi,d-2*np.pi,np.where(d< -np.pi,d+2*np.pi,d))
    c=phase[1:-1,1:-1]
    h=wrap(phase[1:-1,:-2]-c)-wrap(c-phase[1:-1,2:])
    v=wrap(phase[:-2,1:-1]-c)-wrap(c-phase[2:,1:-1])
    d1=wrap(phase[:-2,:-2]-c)-wrap(c-phase[2:,2:])
    d2=wrap(phase[:-2,2:]-c)-wrap(c-phase[2:,:-2])
    reliability[1:-1,1:-1]=h*h+v*v+d1*d1+d2*d2
    grid=np.arange(total).reshape(m,n)
    first=np.concatenate([grid[:,:-1].ravel(),grid[:-1,:].ravel()]);second=np.concatenate([grid[:,1:].ravel(),grid[1:,:].ravel()])
    score=reliability.ravel()[first]+reliability.ravel()[second]
    diff=phase.ravel()[first]-phase.ravel()[second]
    increment=np.where(diff>np.pi,-1,np.where(diff< -np.pi,1,0))
    parent=np.arange(total);size=np.ones(total,dtype=int);offset=np.zeros(total,dtype=int)
    def find(x):
        root=x;value=0
        while parent[root]!=root:value+=offset[root];root=parent[root]
        while parent[x]!=x:
            nxt=parent[x];delta=offset[x];parent[x]=root;offset[x]=value;value-=delta;x=nxt
        return root
    for edge in np.argsort(score,kind='stable'):
        a=int(first[edge]);b=int(second[edge]);ra=find(a);rb=find(b)
        if ra==rb:continue
        difference=int(offset[a])-int(offset[b])-int(increment[edge])
        if size[ra]>=size[rb]:parent[rb]=ra;offset[rb]=difference;size[ra]+=size[rb]
        else:parent[ra]=rb;offset[ra]=-difference;size[rb]+=size[ra]
    for k in range(total):find(k)
    return phase+2*np.pi*offset.reshape(m,n)

