"""Explicit wet/dry, reference-power, attenuation and rain-rate mathematics.

Mathematical adaptation of RAINLINK1.31, Aart Overeem2024,
GPL-3.0-or-later; no upstream package is imported.
"""
import warnings
import math
import numpy as np
from scipy.spatial.distance import cdist

def retrieve_links(data,projected_links,rain_coefficients,*,wet_dry=True,outlier=True,step8=True,wet_threshold=-1.4,specific_threshold=-.7,antenna_db=2.3):
 """Return original-order Dry,F,Pref,corrected powers and rain intensity.

 Fixed native15min/48h acquisition. Powers dBm, corrections dB, distanceskm,
 Rmean mm/h; invalid intermediate estimates are NaN, never zero-filled rain.
 """
 d=np.asarray(data,dtype=float);times,ti=np.unique(d[:,1],return_inverse=True)
 _,first=np.unique(d[:,9],return_index=True);ids=d[np.sort(first),9]
 index={identifier:i for i,identifier in enumerate(ids)};li=np.array([index[x] for x in d[:,9]])
 nt,nl=len(times),len(ids)
 pmin=np.full((nt,nl),np.nan);pmin[ti,li]=d[:,2]
 lengths=d[np.sort(first),4];delta=np.full_like(pmin,np.nan)
 for t in range(1,nt):
  z=pmin[max(0,t-95):t+1];good=np.isfinite(z).sum(axis=0)>=24
  if good.any():delta[t,good]=pmin[t,good]-np.nanmax(z[:,good],axis=0)
 specific=delta/lengths[None]
 geometry={row[0]:row[1:] for row in np.asarray(projected_links)}
 points=np.array([geometry[x] for x in ids]);start,end=points[:,:2],points[:,2:]
 neighbors=(cdist(start,start)<15)&(cdist(end,start)<15)&(cdist(start,end)<15)&(cdist(end,end)<15)
 dry=np.full_like(pmin,np.nan);fout=np.full_like(pmin,np.nan)
 # Upstream F is reused across link loops; unsupported entries retain stale
 # values. They coincide with invalid Dry and cannot supply rain observations.
 f=np.full(nt,np.nan);f[0]=0
 for i in range(nl):
  near=np.flatnonzero(neighbors[i]);z=delta[:,near];zs=specific[:,near]
  enough=np.isfinite(z).sum(axis=1)>=3
  with warnings.catch_warnings():
   warnings.simplefilter('ignore',RuntimeWarning);med=np.nanmedian(z,axis=1);meds=np.nanmedian(zs,axis=1)
  med[~enough]=np.nan;meds[~enough]=np.nan
  difference=np.nan_to_num(specific[:,i]-meds,nan=0).tolist()
  # Original R sum accumulates in extended precision. Faithful compensated
  # summation preserves exact threshold decisions without epsilon padding.
  for t in np.flatnonzero(enough):f[t]=math.fsum(difference[max(0,t-95):t+1])*.25
  fout[:,i]=f
  temp=np.where((med>=wet_threshold)|(meds>=specific_threshold),1.,0.);temp[~np.isfinite(med)|~np.isfinite(meds)]=np.nan
  if step8:
   initial=temp.copy();wet=np.flatnonzero((temp==0)&(delta[:,i]<-2))
   for offset in [-2,-1,1]:
    j=wet+offset;j=j[(j>=0)&(j<nt)];temp[j]=0
   temp[~np.isfinite(initial)]=np.nan
  dry[:,i]=temp
 dryvec=dry[ti,li];fvec=fout[ti,li]
 if not wet_dry:dryvec=np.ones(len(d))
 mean=np.where(dryvec==1,(d[:,2]+d[:,3])/2,np.nan);meanmat=np.full_like(pmin,np.nan);meanmat[ti,li]=mean
 prefmat=np.full_like(pmin,np.nan)
 for t in range(1,nt):
  z=meanmat[max(0,t-95):t+1];good=np.isfinite(z).sum(axis=0)>=10
  if good.any():prefmat[t,good]=np.nanmedian(z[:,good],axis=0)
 pref=prefmat[ti,li]
 filtered=d[:,2].copy()
 if outlier:filtered[fvec<= -32.5]=np.nan
 correction_dry=dryvec if wet_dry else np.zeros(len(d))
 pcmin=pref.copy();chosen=(filtered<pref)&(correction_dry==0);pcmin[chosen]=filtered[chosen]
 pcmin[~np.isfinite(pref)|~np.isfinite(filtered)|~np.isfinite(correction_dry)]=np.nan
 pcmax=pref.copy();chosen=(pcmin<pref)&(d[:,3]<pref);pcmax[chosen]=d[chosen,3];pcmax[~np.isfinite(pcmin)|~np.isfinite(d[:,3])]=np.nan
 rain=rainfall_from_powers(d,pref,pcmin,pcmax,rain_coefficients,antenna_db=antenna_db)
 return dict(Dry=dryvec,F=fvec,Pref=pref,PminCor=pcmin,PmaxCor=pcmax,Rmean=rain)

def rainfall_from_powers(data,pref,pmin_cor,pmax_cor,rain_coefficients,*,antenna_db=2.3):
 """Invert corrected attenuation extrema to mean rain(mm/h), without refitting powers.

 Data columns and coefficient units follow retrieve_links. antenna_db is the
 independent effective antenna correction(dB), not a new observation.
 """
 d=np.asarray(data);pcmin=np.asarray(pmin_cor);pcmax=np.asarray(pmax_cor)
 coefficients=np.asarray(rain_coefficients);abc=np.empty((len(d),2))
 for polarization,table in [(False,coefficients[1]),(True,coefficients[0])]:
  which=(d[:,10]==2) if polarization else (d[:,10]!=2)
  for j in range(2):abc[which,j]=np.interp(np.log(d[which,0]),np.log(table[:,0]),table[:,j+1])
 rmin=abc[:,0]*(np.maximum(pref-pcmax-antenna_db,0)/d[:,4])**abc[:,1]
 rmax=abc[:,0]*(np.maximum(pref-pcmin-antenna_db,0)/d[:,4])**abc[:,1]
 rain=.67*rmin+.33*rmax
 return rain
