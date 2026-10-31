import numpy as np
from scipy.optimize import nnls
from .physics_model import kernel,curvature_matrix,predict

def solve(time_s,g2,relaxation_s,beta,field,valid,alpha=.1):
    alphas=np.broadcast_to(np.asarray(alpha),len(beta));k=kernel(time_s,relaxation_s);n=len(relaxation_s);m=curvature_matrix(n);p=[];norms=[];penalties=[]
    for j,end in enumerate(valid):
        anchor=np.zeros((3,n));anchor[0]=1;anchor[1,0]=1;anchor[2,-1]=1
        c=np.vstack([k[:end],np.sqrt(1000)*anchor,np.sqrt(alphas[j])*m]);d=np.r_[field[:end,j],np.sqrt(1000),0.,0.,np.zeros(n)]
        x,residual=nnls(c,d);p.append(x);norms.append(residual);penalties.append(np.linalg.norm(m@x))
    p=np.asarray(p)
    return p,predict(p,time_s,relaxation_s,beta),np.asarray(norms),np.asarray(penalties)

def lcurve_corner(residual,penalty):
    x=np.log(residual);y=np.log(penalty);x=(x-x.min())/np.ptp(x)*100;y=(y-y.min())/np.ptp(y)*100;choices=[];angles=[];c=np.array([x[-1],y[-1]])
    for i in range(len(x)-4):
        b=np.array([x[i],y[i]])
        for j in range(i+2,len(x)-2):
            a=np.array([x[j],y[j]]);d1=np.linalg.norm(b-a);d2=np.linalg.norm(a-c);d3=np.linalg.norm(b-c)
            angle=np.degrees(np.arccos(np.clip((d1*d1+d2*d2-d3*d3)/(2*d1*d2),-1,1)))
            area=(b[0]-a[0])*(a[1]-c[1])-(a[0]-c[0])*(b[1]-a[1])
            if angle<160 and area>0:choices.append(j);angles.append(angle)
    if not choices:raise ValueError('No L-curve corner')
    return choices[int(np.argmin(angles))]

def compare_lcurve(t,g,s,beta,field,valid):
    alphas=(5**np.arange(-6,2,.1,dtype=float))**2;res=[];pen=[]
    for a in alphas:
        _,_,r,p=solve(t,g,s,beta,field,valid,a);res.append(r);pen.append(p)
    res=np.asarray(res);pen=np.asarray(pen);idx=[lcurve_corner(res[:,j],pen[:,j]) for j in range(len(beta))];selected=alphas[idx];p,pred,_,_=solve(t,g,s,beta,field,valid,selected)
    return alphas,res,pen,selected,p,pred
