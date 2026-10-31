"""PALA multi-object temporal assignment, smoothing, and vessel rendering.

Copyright (c) 2016, Jean-Yves Tinevez
Copyright (c) 2009, Yi Cao
Copyright (c) 2011, Jean-Yves Tinevez
All rights reserved.

Adapted from PALA (CC BY-NC-SA 4.0), simpletracker (Jean-Yves Tinevez),
and Yi Cao's Munkres v2.3. Full terms and attributions: test/validation.md.
"""
import numpy as np
from time import perf_counter
from .physics_model import localize
from .preprocessing import preprocess_block

METHODS=('radial','wa','integer')
OUTPUT_SHAPE=(831,1281)

def munkres(cost):
    """Partial squared-distance assignment; preserve upstream column-first ties."""
    original=np.asarray(cost,dtype=float)
    assignment=np.full(original.shape[0],-1,dtype=np.int64)
    valid=np.isfinite(original)
    rows=np.flatnonzero(valid.any(axis=1)); cols=np.flatnonzero(valid.any(axis=0))
    nr,nc=len(rows),len(cols);n=max(nr,nc)
    if n==0:return assignment
    finite=original[valid]; total=finite.sum()
    big=10**(np.ceil(np.log10(total))+1) if total>0 else 0.0
    matrix=np.where(valid,original,big)
    d=np.full((n,n),10*finite.max());d[:nr,:nc]=matrix[np.ix_(rows,cols)]
    minr=d.min(axis=1);minc=(d-minr[:,None]).min(axis=0)
    zp=d==(minr[:,None]+minc)
    star=np.full(n,-1,dtype=np.int64)
    while zp.any():
        flat=np.flatnonzero(zp.ravel(order='F'))[0];r,c=flat%n,flat//n
        star[r]=c;zp[r]=False;zp[:,c]=False
    while np.any(star<0):
        coveredc=np.zeros(n,bool);coveredc[star[star>=0]]=True
        coveredr=np.zeros(n,bool);prime=np.full(n,-1,dtype=np.int64)
        cr=np.flatnonzero(~coveredr);cc=np.flatnonzero(~coveredc)
        zz=d[np.ix_(cr,cc)]==(minr[cr,None]+minc[cc])
        rr,ccc=np.unravel_index(np.flatnonzero(zz.ravel(order='F')),zz.shape,order='F')
        rr=cr[rr];ccc=cc[ccc]
        while True:
            found=False
            while len(ccc):
                ur,uc=int(rr[0]),int(ccc[0]);prime[ur]=uc;st=star[ur]
                if st<0:found=True;break
                coveredr[ur]=True;coveredc[st]=False
                keep=rr!=ur;rr=rr[keep];ccc=ccc[keep]
                cr=np.flatnonzero(~coveredr)
                extra=cr[d[cr,st]==minr[cr]+minc[st]]
                rr=np.concatenate([rr,extra]);ccc=np.concatenate([ccc,np.full(len(extra),st)])
            if found:break
            cr=np.flatnonzero(~coveredr);cc=np.flatnonzero(~coveredc)
            residual=d[np.ix_(cr,cc)]-(minr[cr,None]+minc[cc])
            value=residual.min()
            zz=residual==value
            ri,ci=np.unravel_index(np.flatnonzero(zz.ravel(order='F')),zz.shape,order='F')
            rr=cr[ri];ccc=cc[ci]
            minc[~coveredc]+=value;minr[coveredr]-=value
        row=np.flatnonzero(star==uc)
        star[ur]=uc
        while len(row):
            r=int(row[0]);star[r]=-1;uc=int(prime[r]);row=np.flatnonzero(star==uc);star[r]=uc
    for r,c in enumerate(star[:nr]):
        if c<nc and valid[rows[r],cols[c]]:assignment[rows[r]]=cols[c]
    return assignment

def link_frames(source,target,max_distance=1.0):
    """Return target index or -1 per source; inclusive squared-distance gate."""
    delta=np.asarray(source,dtype=float)[:,None,:]-np.asarray(target,dtype=float)[None,:,:]
    cost=(delta*delta).sum(axis=2)
    cost[cost>max_distance**2]=np.inf
    return munkres(cost)

def smooth_track(x,span=20):
    """MATLAB default centered moving average with shrinking odd endpoints."""
    x=np.asarray(x,dtype=float);n=len(x)
    span=min(span,n);span-=int(span%2==0);half=(span-1)//2
    result=np.empty(n)
    for i in range(n):
        h=min(half,i,n-1-i);result[i]=np.mean(x[i-h:i+h+1])
    return result

def track_localizations(localizations,n_frames=200,min_length=8,max_distance=1.0):
    """Return ordered float64 tracks [z,x,frame], independently per block."""
    a=np.asarray(localizations,dtype=float).copy()
    if not len(a):return []
    a[:,3]-=a[:,3].min()-1
    per=[a[a[:,3]==k] for k in range(1,n_frames+1)]
    counts=np.array([len(x) for x in per]);starts=np.r_[0,counts.cumsum()]
    points=np.concatenate(per)
    nxt=np.full(len(points),-1,dtype=np.int64);has_source=np.zeros(len(points),bool)
    for k in range(n_frames-1):
        target=link_frames(per[k][:,1:3],per[k+1][:,1:3],max_distance)
        good=np.flatnonzero(target>=0)
        nxt[starts[k]+good]=starts[k+1]+target[good];has_source[starts[k+1]+target[good]]=True
    tracks=[]
    for start in np.flatnonzero(~has_source):
        ids=[];current=int(start)
        while current>=0:ids.append(current);current=int(nxt[current])
        if len(ids)>min_length:tracks.append(points[ids,1:4].copy())
    return tracks

def interpolate_tracks(tracks,frame_rate=500.0,max_distance=1.0,resolution=10):
    """Original PALA path speed uses n*dt and step .8/(max_distance*res)."""
    step=.8/max_distance/resolution
    result=[]
    for raw in tracks:
        n=len(raw);q=1+np.arange(int(np.floor((n-1)/step))+1)*step
        # MATLAB colon excludes an endpoint beyond its roundoff tolerance.
        q=q[q<=n+np.finfo(float).eps*n]
        points=np.column_stack([np.interp(q,np.arange(1,n+1),smooth_track(raw[:,j])) for j in (0,1)])
        speed=np.linalg.norm(np.diff(points,axis=0),axis=1).sum()/n*frame_rate
        result.append(np.column_stack([points,np.full(len(q),speed)]).astype(np.float32))
    return result

def render_tracks(tracks,shape=OUTPUT_SHAPE,resolution=10,with_speed=True):
    """Count each track once per visited pixel; mean constant path speed."""
    density=np.zeros(shape,dtype=np.float64);speed_sum=np.zeros(shape,dtype=np.float64)
    for track in tracks:
        mapped=(track[:,:2]+np.float32(1))*np.float32(resolution)
        pixel=np.floor(mapped+.5).astype(np.int64)-1
        valid=(pixel>=0).all(axis=1)&(pixel[:,0]<shape[0])&(pixel[:,1]<shape[1])
        index=np.unique(pixel[valid,0]+shape[0]*pixel[valid,1])
        z=index%shape[0];x=index//shape[0]
        density[z,x]+=1
        if with_speed:speed_sum[z,x]+=float(track[0,2])
    speed=np.divide(speed_sum,density,out=np.zeros_like(speed_sum),where=density>0)
    return density,speed

def reconstruct_block(iq,methods=METHODS):
    """Reconstruct all requested methods on one complete IQ block."""
    filtered=preprocess_block(iq);result={}
    for method in methods:
        started=perf_counter()
        positions=localize(filtered,method)
        tracks=track_localizations(positions)
        dense=interpolate_tracks(tracks)
        density,speed=render_tracks(dense)
        nointerp,_=render_tracks([x.astype(np.float32) for x in tracks],with_speed=False)
        histogram=np.bincount([len(x) for x in tracks],minlength=201)
        result[method]={'density':density,'speed':speed,'nointerp':nointerp,'counts':np.array([len(positions),len(tracks)]),'track_length_histogram':histogram,'method_runtime_seconds':perf_counter()-started}
    return result

def reconstruct(iq,methods=METHODS,progress=None):
    """Accumulate full blocks without linking across known acquisition gaps."""
    result={m:{'density':np.zeros(OUTPUT_SHAPE),'speed_sum':np.zeros(OUTPUT_SHAPE),'nointerp':np.zeros(OUTPUT_SHAPE),'counts':[],'saturation':[],'track_length_histogram':np.zeros(201,dtype=np.int64),'method_runtime_seconds':[]} for m in methods}
    for ib,block in enumerate(iq):
        value=reconstruct_block(block,methods)
        for m in methods:
            for k in ('density','nointerp'):result[m][k]+=value[m][k]
            result[m]['speed_sum']+=value[m]['density']*value[m]['speed'];result[m]['counts'].append(value[m]['counts'])
            result[m]['saturation'].append(np.count_nonzero(result[m]['density']))
            result[m]['track_length_histogram']+=value[m]['track_length_histogram']
            result[m]['method_runtime_seconds'].append(value[m]['method_runtime_seconds'])
        if progress is not None:progress(ib+1,len(iq))
    for m in methods:
        a=result[m];a['speed']=np.divide(a.pop('speed_sum'),a['density'],out=np.zeros(OUTPUT_SHAPE),where=a['density']>0);a['counts']=np.array(a['counts'])
        a['saturation']=np.asarray(a['saturation'],dtype=np.int64)
        a['method_runtime_seconds']=np.asarray(a['method_runtime_seconds'])
    return result
