"""Measured high-field susceptibility and lower-branch corrections.
MIT adaptation of FORCsensei. Fields are mu0 H in tesla; moments are A m^2.
"""
import numpy as np
from types import SimpleNamespace

def find_Hopen(X,dMlim=0.05):
    
    H = X["H"]    
    Hr = X["Hr"]
    M = X["M"]
    Fk = X["Fk"]
    Fj = X["Fj"]
    dH = X["dH"]
    
    Hmin = np.min(H)
    Hmax = np.max(H)

    Nbar = 10
    nH = int((Hmax - Hmin)/dH)
    Hi = np.linspace(Hmin,Hmax,nH*50+1)
    Mi = np.empty(Hi.size)
    
    #perform basic loess
    for i in range(Hi.size):
        idx = (H>=Hi[i]-2.5*dH) & (H<=Hi[i]+2.5*dH)
        Mbar = M[idx]
        Hbar = H[idx]
        Fbar = Fk[idx]
        F0 = np.sort(np.unique(Fbar))
        if F0.size>Nbar:
            F0=F0[-Nbar]
        else:
            F0=np.min(F0)
        idx = Fbar>=F0
        
        p = np.polyfit(Hbar[idx],Mbar[idx],2)
        Mi[i] = np.polyval(p,Hi[i])
    
    Hlower = Hi
    Mlower = Mi

    idx0 = X['Fj']==1.0
    idx1 = X['Fk']==np.max(X['Fk'])
    
    lower = np.interp(X['H'][idx0],Hlower,Mlower,left=np.nan,right=np.nan)

    dM = X['M'][idx0]-lower
    dM0 = np.interp(0,np.flip(X['H'][idx0]),np.flip(dM),left=np.nan,right=np.nan)

    idx = (dM<=dMlim*dM0) & (X['H'][idx0]<0)
    Hopen = -np.max(X['H'][idx0][idx])


    X['Hopen'] = Hopen
    
    return X

def lowerbranch_subtract(X):
    """Function to subtract lower hysteresis branch from FORC magnetizations
    
    Inputs:
    H: Measurement applied field [float, SI units]
    Hr: Reversal field [float, SI units]
    M: Measured magnetization [float, SI units]
    Fk: Index of measured FORC (int)
    Fj: Index of given measurement within a given FORC (int)
    
    Outputs:
    M: lower branch subtracted magnetization [float, SI units]
   
    
    """
    
    #unpack
    H = X["H"]    
    Hr = X["Hr"]
    M = X["M"]
    Fk = X["Fk"]
    Fj = X["Fj"]
    dH = X["dH"]
    
    Hmin = np.min(H)
    Hmax = np.max(H)


    Nbar = 10
    nH = int((Hmax - Hmin)/dH)
    Hi = np.linspace(Hmin,Hmax,nH*50+1)
    Mi = np.empty(Hi.size)
    
    #perform basic loess
    for i in range(Hi.size):
        idx = (H>=Hi[i]-2.5*dH) & (H<=Hi[i]+2.5*dH)
        Mbar = M[idx]
        Hbar = H[idx]
        Fbar = Fk[idx]
        F0 = np.sort(np.unique(Fbar))
        if F0.size>Nbar:
            F0=F0[-Nbar]
        else:
            F0=np.min(F0)
        idx = Fbar>=F0
        
        p = np.polyfit(Hbar[idx],Mbar[idx],2)
        Mi[i] = np.polyval(p,Hi[i])
    
    Hlower = Hi
    Mlower = Mi
    Mcorr=M-np.interp(H,Hlower,Mlower,left=np.nan,right=np.nan) #subtracted lower branch from FORCs via interpolation

    Fk=Fk[~np.isnan(Mcorr)] #remove any nan
    Fj=Fj[~np.isnan(Mcorr)] #remove any nan
    H=H[~np.isnan(Mcorr)] #remove any nan
    Hr=Hr[~np.isnan(Mcorr)] #remove any nan
    M=M[~np.isnan(Mcorr)] #remove any nan
    Mcorr = Mcorr[~np.isnan(Mcorr)] #remove any nan
    
    #repack
    X["H"] = H    
    X["Hr"] = Hr
    X["M"] = M
    X["Fk"] = Fk
    X["Fj"] = Fj
    X["DM"] = Mcorr
    #X["Hlower"] = Hi
    #X["Mlower"] = Mi
    
    return X

def slope_correction(X):
  
    #unpack
    H = X["H"]
    M = X["M"]
  
    # high field slope correction
    Hidx = H > (X["slope"].value/100) * np.max(H)
    p = np.polyfit(H[Hidx],M[Hidx],1)
    M = M - H*p[0]
  
    #repack
    X["M"]=M
  
    return X

def preprocess(raw, slope_percent=70.0):
    """Correct all measured branches and determine the opening field.
    raw contains five equal length float64 acquisition arrays. The actual mean
    step of the final branch fixes the local regression scale in tesla.
    """
    names=dict(H='field',Hr='reversal_field',M='moment',Fk='curve_index',Fj='point_index')
    state={k:np.asarray(raw[v],dtype=np.float64).copy() for k,v in names.items()}
    state['dH']=np.mean(np.diff(state['H'][state['Fk']==np.max(state['Fk'])]))
    state['slope']=SimpleNamespace(value=slope_percent)
    state=slope_correction(state)
    state=find_Hopen(state)
    return lowerbranch_subtract(state)
