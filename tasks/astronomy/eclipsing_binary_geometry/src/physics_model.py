"""Selected spherical ellc model, extracted from GPL-3.0-or-later ellc.

Original copyright P. F. L. Maxted; numerical conventions follow verified V2.
Selected spherical branch: no upstream imports or cached target parameters.
"""
import numpy as np
from numpy.polynomial.legendre import leggauss

P=14.2769001
T0=2456023.26988
Q=.1968
GL={n:leggauss(n) for n in range(2,9)}

def brent(f, ax, bx, cx):
    a,b=min(ax,cx),max(ax,cx);v=w=x=bx;e=d=0.;fx=f(x);fv=fw=fx
    for _ in range(100):
        xm=.5*(a+b);t1=1e-8*abs(x)+1e-10;t2=2*t1
        if abs(x-xm)<=t2-.5*(b-a):return x
        if abs(e)>t1:
            r=(x-w)*(fx-fv);q=(x-v)*(fx-fw);p=(x-v)*q-(x-w)*r;q=2*(q-r)
            if q>0:p=-p
            q=abs(q);et=e;e=d
            if abs(p)>=abs(.5*q*et) or p<=q*(a-x) or p>=q*(b-x):
                e=a-x if p>=q*(b-x) else b-x;d=.3819660*e
            else:
                d=p/q;u=x+d
                if u-a<t2 or b-u<t2:d=np.copysign(t1,xm-x)
        else:e=a-x if x>=xm else b-x;d=.3819660*e
        u=x+d if abs(d)>=t1 else x+np.copysign(t1,d);fu=f(u)
        if fu<=fx:
            if u>=x:a=x
            else:b=x
            v,w,x=w,x,u;fv,fw,fx=fw,fx,fu
        else:
            if u<x:a=u
            else:b=u
            if fu<=fw or w==x:v,w=w,u;fv,fw=fw,fu
            elif fu<=fv or v==x or v==w:v=u;fv=fu
    return x

def periastron(t0,e,omega,inc):
    theta=np.pi/2-omega
    if inc!=np.pi/2:
        def delta(t):return (1-e*e)*np.sqrt(1-np.sin(inc)**2*np.sin(t+omega)**2)/(1+e*np.cos(t))
        theta=brent(delta,theta-np.pi/4,theta,theta+np.pi/4)
    E=2*np.arctan(np.sqrt((1-e)/(1+e))*np.tan(theta/2))
    return t0-(E-e*np.sin(E))*P/(2*np.pi)

def eccentric_anomaly(m,e):
    m,e=np.broadcast_arrays(np.asarray(m,dtype=float),np.asarray(e,dtype=float))
    if not np.isfinite(m).all() or not np.isfinite(e).all() or np.any((e<0)|(e>=1)):
        raise ValueError('Kepler inputs require finite anomalies and 0<=e<1')
    E=np.fmod(m,2*np.pi)+e*np.sin(m)+e*e*np.sin(2*m)/2
    with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
        for _ in range(100):
            step=(m-(E-e*np.sin(E)))/(1-e*np.cos(E));E=E+step
            if np.max(np.abs(step),initial=0)<=1e-9:break
        E=np.mod(E,2*np.pi);wrapped=np.mod(m,2*np.pi)
        bad=~np.isfinite(E)|(np.abs(E-e*np.sin(E)-wrapped)>1e-10)
    # Versioned robustness extension: unguarded source Newton can diverge at
    # extreme e. Kepler's equation is monotone for e<1, giving a unique bracket.
    # No bound-data/original-fixture Newton evaluation uses this fallback.
    if np.any(bad):
        target=wrapped[bad];ecc=e[bad];lo=np.zeros_like(target);hi=np.full_like(target,2*np.pi)
        for _ in range(72):
            mid=(lo+hi)/2;left=mid-ecc*np.sin(mid)<target
            lo=np.where(left,mid,lo);hi=np.where(left,hi,mid)
        E=np.array(E,copy=True);E[bad]=(lo+hi)/2
    return E

def orbit(t,p,lighttime=False,geometry=False):
    r1,r2,ideg,fs,fc,sj,k=p;e=fs*fs+fc*fc;om=np.arctan2(fs,fc);inc=np.deg2rad(ideg)
    a1=.019771142*k*P*np.sqrt(1-e*e)/np.sin(inc);a=(1+1/Q)*a1
    alite=a*6.957e8/2.99792458e8/86400 if lighttime else 0.
    dlt=alite*np.sin(inc)*(Q-1)/(Q+1)*(1-e*e)/(1+e*np.sin(om))
    tp=periastron(T0-dlt,e,om,inc)
    def state(times,positive=False):
        phase=np.fmod((times-tp)/P,1)
        if positive:phase=np.fmod(1+phase,1)
        E=eccentric_anomaly(2*np.pi*phase,e);nu=2*np.arctan(np.sqrt((1+e)/(1-e))*np.tan(E/2));return 1-e*np.cos(E),nu
    rr,nu=state(t,True);w=-rr*np.sin(inc)*np.sin(nu+om)
    ra,na=state(t+alite*w*Q/(1+Q));rb,nb=state(t-alite*w/(1+Q))
    u1=ra*np.cos(na+om)*Q/(1+Q);v1=ra*np.cos(inc)*np.sin(na+om)*Q/(1+Q)
    u2=-rb*np.cos(nb+om)/(1+Q);v2=-rb*np.cos(inc)*np.sin(nb+om)/(1+Q)
    sep=np.hypot(u2-u1,v2-v1)
    rv=a*Q/(1+Q)/(.019765685498705825*P*np.sqrt(1-e*e))*np.sin(inc)*(np.cos(na+om)+e*np.cos(om))
    return (sep,w<0,rv,u2-u1,v2-v1) if geometry else (sep,w<0,rv)

def brightness(x,y,R,ld):
    mu=np.sqrt(np.maximum(0,1-(x*x+y*y)/(R*R)));u,v=ld
    return 1-u*(1-mu)-v*(1-mu)**2

def disk_grid():
    ys,ws=GL[8];xx=[];yy=[];ww=[]
    for y,wy in zip(ys,ws):
        rank=int(np.argmin(abs(abs(y)-np.sort(abs(ys))[::2])))
        n=[8,7,5,2][rank];xs,wx=GL[n];half=np.sqrt(1-y*y)
        xx.extend(xs*half);yy.extend(np.full(n,y));ww.extend(wx*half*wy)
    return np.array(xx),np.array(yy),np.array(ww)
DX,DY,DW=disk_grid()

def disk_mean(R,ld):return np.sum(DW*brightness(DX*R,DY*R,R,ld))/DW.sum()

def region_integrals(a,b,lower,upper,R,ld):
    """Per-region variable-order Gaussian brightness and area integrals."""
    a=np.asarray(a);b=np.asarray(b);xs=(b-a)/2;x0=(a+b)/2
    ng=np.clip(np.floor(4*abs(b-a)/R+.5).astype(int),4,8)
    intensity=np.zeros(len(a));area=np.zeros(len(a))
    for n in np.unique(ng):
        rows=np.flatnonzero(ng==n);nodes,weights=GL[int(n)];x=x0[rows,None]+xs[rows,None]*nodes
        lo=lower(x,rows);hi=upper(x,rows);ym=(hi+lo)/2;ys=(hi-lo)/2
        ny=np.clip(np.floor(.5*n*abs((hi-lo)/xs[rows,None])+.5).astype(int),4,8)
        for m in np.unique(ny):
            ir,ic=np.nonzero(ny==m);yn,yw=GL[int(m)];y=ym[ir,ic,None]+ys[ir,ic,None]*yn
            val=np.sum(brightness(x[ir,ic,None],y,R,ld)*yw,axis=1)
            wt=weights[ic]*xs[rows[ir]]*ys[ir,ic]
            np.add.at(intensity,rows[ir],wt*val);np.add.at(area,rows[ir],wt*2)
    return intensity,area

def blocked(R,r,d,ld,dx=None,dy=None):
    d=np.asarray(d);out=np.zeros_like(d);mean=disk_mean(R,ld)
    cover=(r>=R)&(d<=r-R);out[cover]=1
    inside=(R>r)&(d<=R-r)
    if np.any(inside):
        ix=np.flatnonzero(inside);cx=d[ix] if dx is None else dx[ix];cy=np.zeros(len(ix)) if dy is None else dy[ix]
        vals=brightness(cx[:,None]+r*DX,cy[:,None]+r*DY,R,ld)
        out[ix]=(r/R)**2*np.sum(vals*DW,axis=1)/DW.sum()/mean
    ix=np.flatnonzero((d<r+R)&~cover&~inside)
    if not len(ix):return out
    dd=d[ix];c=(R*R-r*r+dd*dd)/(2*dd)
    area=R*R*np.arccos(np.clip(c/R,-1,1))+r*r*np.arccos(np.clip((dd-c)/r,-1,1))-.5*np.sqrt(np.maximum(0,(-dd+R+r)*(dd+R-r)*(dd-R+r)*(dd+R+r)))
    small=area<.5*np.pi*R*R
    for visible in (False,True):
        ids=np.flatnonzero(~small if visible else small)
        if not len(ids):continue
        di=dd[ids];ci=c[ids];I=np.zeros(len(ids));A=np.zeros(len(ids))
        bg=lambda x,rows:np.sqrt(np.maximum(0,R*R-x*x))
        fg=lambda x,rows:np.sqrt(np.maximum(0,r*r-(x-di[rows,None])**2))
        regions=[(ci,np.full(len(ids),R),lambda x,rows:-bg(x,rows),bg),(di-r,ci,lambda x,rows:-fg(x,rows),fg)]
        if visible:regions=[(di-r,ci,lambda x,rows:-bg(x,rows),lambda x,rows:-fg(x,rows)),(di-r,ci,fg,bg),(np.full(len(ids),-R),di-r,lambda x,rows:-bg(x,rows),bg)]
        for a,b,lo,hi in regions:
            bi,ba=region_integrals(a,b,lo,hi,R,ld);I+=bi;A+=ba
        fraction=(np.pi*R*R-area[ids] if visible else area[ids])*(I/A)/(np.pi*R*R*mean)
        out[ix[ids]]=1-fraction if visible else fraction
    return out

def photometry(t,p,group):
    sep,front,_,dx,dy=orbit(t,p,group==4,True);r1,r2,_,_,_,sj,_=p
    if group<4:sj=0.;ld1=(.3532,.2606) if group<2 else (.2552,.2598);ld2=(0.,0.)
    else:ld1=(0.,0.);ld2=(.1076,.2857)
    f=np.ones(len(t));i=np.flatnonzero(front);j=np.flatnonzero(~front);ratio=sj*(r2/r1)**2
    f[i]-=blocked(r1,r2,sep[i],ld1,dx[i],dy[i])/(1+ratio)
    if sj:f[j]-=blocked(r2,r1,sep[j],ld2,-dx[j],-dy[j])*ratio/(1+ratio)
    return f
