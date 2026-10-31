"""Bayesian smoothing selection and weighted local FORC inversion.
MIT-derived numerical stages of FORCsensei, executed sequentially in NumPy.
The signed switching response is not a nonnegative probability distribution.
"""
import numpy as np
from scipy.special import gammaln,logsumexp
from scipy.stats import t
from .physics_model import design_matrix

def inv_jit(a):
    return np.linalg.inv(a)

def dot_jit(a,b):
    return np.dot(a,b)

def OLS2BF(X,y):

    N = y.size    
    XT = X.T
    XTX = dot_jit(XT,X)
    yc = y - np.mean(y)
    ydev = dot_jit(yc.T,yc)

    # Test 2nd order model woth zero derivative
    status = True
    BF1 = 0.0
    try:
        iv = inv_jit(XTX[:5,:5])
    except np.linalg.LinAlgError as err:
        status = False
    
    if status:
        resid = dot_jit(X[:,:5],(dot_jit(dot_jit(iv,XT[:5,:]),y))) - y 
        r2_1 = 1 - dot_jit(resid.T,resid) / ydev
        BF1, B = Bayes_factor(N,r2_1)
        
    # Test 2nd order model    
    status = True
    BF2 = 0.0
    try:
        iv = inv_jit(XTX[:6,:6])
    except np.linalg.LinAlgError as err:
        status = False
    
    if status:
        resid = dot_jit(X[:,:6],(dot_jit(dot_jit(iv,XT[:6,:]),y))) - y
        r2_2 = 1 - dot_jit(resid.T,resid) / ydev
        BF2, _ = Bayes_factor(N,r2_2,2,B)    
    
    # Test 3rd order model
    status = True
    BF3 = 0.0      
    try:
        iv = inv_jit(XTX)
    except np.linalg.LinAlgError as err:
        status = False
    
    if status:
        resid = dot_jit(X,(dot_jit(dot_jit(iv,XT),y))) - y 
        r2_3 = 1 - dot_jit(resid.T,resid) / ydev   
        BF3, _ = Bayes_factor(N,r2_3,3,B)
    
    return np.array((0,BF1,BF2,BF3))

def Bayes_factor(N,R2,order=1,B=0):
    
    a = -3/4
    if (order==1):
        # Paper Eq. (A4): j=4 and a=-3/4 imply log Gamma(2.25)/Gamma(.25).
        # Upstream hardcodes log(.25), missing a factor 1.25. This explicit
        # paper-faithful correction is validated against an independent
        # corrected-source oracle; unchanged-upstream parity is not claimed.
        A = -1.1631508098056809
        B = gammaln((N-1)/2)
        BF = A+gammaln((N-4-1)/2)-B+(-(N-4-1)/2+a+1)*np.log(1-R2)
    elif (order==2):
        A = -0.8128078577831402
        BF = A+gammaln((N-5-1)/2)-B+(-(N-5-1)/2+a+1)*np.log(1-R2)
    else:
        A = 1.5205488938776595
        BF = A+gammaln((N-9-1)/2)-B+(-(N-9-1)/2+a+1)*np.log(1-R2)

    
    return BF, B

def OLS_pts(sc0,sc1,lamb_sc,sb0,sb1,lamb_sb,Hc,Hb,dH,Hc0,Hb0):
       
    Sc=vari_s(sc0,sc1,lamb_sc,Hc0,dH)
    Sb=vari_s(sb0,sb1,lamb_sb,Hb0,dH)
    idx = np.logical_and(np.abs(Hc-Hc0)/dH<Sc,np.abs(Hb-Hb0)/dH<Sb)

    return idx    

def vari_s(s0,s1,lamb,H,dH):
    
    #calculate local smoothing factor
    RH = np.maximum(s0,np.abs(H)/dH)
    LH = (1-lamb)*s1+lamb*np.abs(H)/dH
    
    return np.minimum(LH,RH)

def vari_T(u,s):
    
    T=np.zeros(u.shape) #initialize array

    absu=np.abs(u)
    absu_s=absu-s

    idx=(absu<=s)
    T[idx]= 2.*absu_s[idx]**2 #3rd condition

    idx=(absu<=s-0.5)
    T[idx]=1.-2.*(absu_s[idx]+1.)**2 #2nd condition

    idx=(absu<=s-1.)
    T[idx]=1.0 #first condition
  
    return T

def vari_W(Hc,Hc0,Hb,Hb0,dH,Sc,Sb):
    # calculate grid of weights
    #Hc = Hc grid
    #Hb = Hb grid
    #Hc0,Hb0 = center of weighting function
    #dH = field spacing
    #Sc = Hc-axis smoothing factor
    #Sb = Hb-axis smoothing factor
    
    x=Hc-Hc0
    y=Hb-Hb0
        
    return vari_T(x/dH,Sc)*vari_T(y/dH,Sb)

def vari_weights(sc0,sc1,lamb_sc,sb0,sb1,lamb_sb,Hc,Hb,dH,Hc0,Hb0):
       
    Sc=vari_s(sc0,sc1,lamb_sc,Hc0,dH)
    Sb=vari_s(sb0,sb1,lamb_sb,Hb0,dH)
        
    idx=((np.abs(Hc-Hc0)/dH<Sc) & (np.abs(Hb-Hb0)/dH<Sb))
    weights=vari_W(Hc[idx],Hc0,Hb[idx],Hb0,dH,Sc,Sb)
    
    return weights, idx

def reconstruct(state,smoothing_factors=(2.,3.,4.,5.,7.,10.),scoring_count=2000):
    """Infer density and standard error on the native author reporting domain.
    state is preprocess() output. The source RandomState(999) samples measured
    positions for smoothing selection; all measurements enter final inference.
    """
    h=state['H'];r=state['Hr'];m=state['DM'];dh=state['dH']
    hc=(h-r)/2;hb=(h+r)/2;design=design_matrix(h,r)
    window=(hc>=0)&(hc<=0.12)&(hb<=0.05)&(hb>=(-0.05-0.12+hc))
    points=np.flatnonzero(window)
    score=np.flatnonzero((np.abs(h[points])<state['Hopen'])&(np.abs(r[points])<state['Hopen']))
    indices=np.sort(np.random.RandomState(999).choice(score.size,scoring_count,replace=False))
    chosen=points[score[indices]]
    normalized=m/np.max(m)
    counts=[]
    def bayes_at(factor,positions):
        values=[]
        for j in positions:
            use=OLS_pts(factor,factor,0.,factor,factor,0.,hc,hb,dh,hc[j],hb[j])
            values.append(OLS2BF(design[use],normalized[use]))
        result=np.array(values);result[np.isinf(result)]=1e200
        return result
    for factor in smoothing_factors:
        bf=bayes_at(factor,chosen)
        labels=np.argmax(bf-[np.log(10),np.log(10),0,np.log(10)],axis=1)
        labels[np.max(bf,axis=1)<1]=0
        counts.append(np.bincount(labels,minlength=4))
    counts=np.asarray(counts)
    factor=float(smoothing_factors[np.argmax(counts[:,2])])
    bf=bayes_at(factor,points);order=np.argmax(bf,axis=1)
    rho=np.zeros(h.size);se=np.zeros(h.size);pvalue=np.zeros(h.size)
    for position,j in enumerate(points):
        if order[position]>1:
            weights,use=vari_weights(factor,factor,0.,factor,factor,0.,hc,hb,dh,hc[j],hb[j])
            a=design[use,:6]*np.sqrt(weights[:,None]);b=m[use]*np.sqrt(weights)
            fit=np.linalg.lstsq(a,b,rcond=0)
            if fit[1].size!=1:
                raise ValueError(f'Rank-deficient selected local quadratic at sample {j}: rank={fit[2]}, neighbors={b.size}, BF={bf[position]}')
            rho[j]=-0.5*fit[0][5]
            variance=fit[1]/(b.size-6)
            covariance=variance*np.linalg.inv(a.T@a)
            derivative=np.zeros((6,1));derivative[5]=-0.5
            se[j]=np.sqrt((derivative.T@covariance@derivative).item())
            pvalue[j]=2*(1-t.cdf(abs(rho[j]/se[j]),len(b)))
    return dict(reconstruction=rho[points],standard_error=se[points],bayes_factors=bf,
        model_probabilities=np.exp(bf-logsumexp(bf,axis=1)[:,None]),
        smoothing_scores=counts,selected_smoothing=factor,
        scoring_indices=chosen,domain_indices=points,pvalue=pvalue[points])
