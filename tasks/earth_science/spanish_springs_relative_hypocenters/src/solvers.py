"""GrowClust hierarchical clustering/relocation; originalGPL3notice in physics_model."""
import numpy as np
from .physics_model import F,relocate_pair
def heap_order(values):
 n=len(values);ind=np.arange(n,dtype=np.int64)
 if n<2:return ind
 left=n//2;right=n-1
 while True:
  if left>0:left-=1;item=ind[left];q=values[item]
  else:
   item=ind[right];q=values[item];ind[right]=ind[0];right-=1
   if right==0:ind[0]=item;return ind
  i=left;j=2*left+1
  while j<=right:
   if j<right and values[ind[j]]<values[ind[j+1]]:j+=1
   if q<values[ind[j]]:ind[i]=ind[j];i=j;j=2*j+1
   else:break
  ind[i]=item

def quality_order(d,rmin=.6,max_distance=80.):
 score=np.empty(len(d['start']),dtype=np.float32)
 for k,(start,end) in enumerate(zip(d['start']-1,d['end'])):
  mask=(d['cross_correlation'][start:end]>=F(rmin))&(d['station_distance'][start:end]<=F(max_distance));v=d['cross_correlation'][start:end][mask];n=len(v)
  score[k]=F(F(np.cumsum(v,dtype=np.float32)[-1]/F(n))*F(n)) if n else F(0)
 return score,heap_order(score)

def reconstruct_catalog(d,verbose=False,stop_after=None,rmin=.6,nit=15):
 lat=d['latitude'].copy();lon=d['longitude'].copy();dep=d['depth'].copy();tim=np.zeros_like(dep);n=len(lat);cl=np.arange(n);members={i:np.array([i]) for i in range(n)};tlat=lat.copy();tlon=lon.copy();tdep=dep.copy();score,order=quality_order(d,rmin=rmin);pairs1=d['pair_first']-1;pairs2=d['pair_second']-1;starts=d['start']-1;ends=d['end'];accepted=[];attempts=0;start_time=__import__('time').perf_counter();deg=111.1949266;rad=180/3.1415927
 ordered=order[::-1]
 for cursor,pair in enumerate(ordered):
  if score[pair]<0:continue
  i,j=pairs1[pair],pairs2[pair]
  if i<0 or j<0 or i==j:continue
  a,b=int(cl[i]),int(cl[j])
  if a==b:continue
  cos=np.cos(((lat[i]+lat[j])/2)/rad);dx=(tlon[a]-tlon[b])*deg*cos;dy=(tlat[a]-tlat[b])*deg;dz=float(tdep[a]-tdep[b]);distance=F(np.sqrt(dx*dx+dy*dy+dz*dz))
  if distance>8:continue
  remainder=ordered[cursor:];c1=cl[pairs1[remainder]];c2=cl[pairs2[remainder]];mask=((c1==a)&(c2==b))|((c1==b)&(c2==a));links=remainder[np.flatnonzero(mask)[:10]]
  obs=np.concatenate([np.arange(starts[k],ends[k]) for k in links]);counts=ends[links]-starts[links];first=np.repeat(pairs1[links],counts);second=np.repeat(pairs2[links],counts);reverse=cl[first]!=a;e1=np.where(reverse,second,first);e2=np.where(reverse,first,second)
  g={k:d[k][obs] for k in ['phase','station_latitude','station_longitude']};g['delta_time']=np.where(reverse,-d['delta_time'][obs],d['delta_time'][obs]);g.update(latitude1=lat[e1]-tlat[a],longitude1=lon[e1]-tlon[a],depth1=dep[e1]-tdep[a],time1=tim[e1],latitude2=lat[e2]-tlat[b],longitude2=lon[e2]-tlon[b],depth2=dep[e2]-tdep[b],time2=tim[e2])
  center=((tlat[a]+tlat[b])/2,(tlon[a]+tlon[b])/2,F(F(tdep[a]+tdep[b])/F(2)));result,dt,res,sep,rms,rmed,resol=relocate_pair(center,g,d['travel_times'],d['distance_grid'],d['depth_grid'],nit=nit);attempts+=1
  if min(result[2],result[5])<min(0,float(d['depth'].min())) or rms>F(.2) or rmed>F(.1) or sep>F(6):continue
  if abs(dt)>50:raise ValueError('origin correction exceeds source safety bound')
  na=len(members[a]);nb=len(members[b]);nt=na+nb;old=np.array([(tlat[a]*na+tlat[b]*nb)/nt,(tlon[a]*na+tlon[b]*nb)/nt,F(F(F(tdep[a]*F(na))+F(tdep[b]*F(nb)))/F(nt))]);new=np.array([(result[0]*na+result[3]*nb)/nt,(result[1]*na+result[4]*nb)/nt,F(F(F(result[2])*F(na)+F(result[5])*F(nb))/F(nt))]);dc=new-old;shift1=result[:3]-np.array([tlat[a],tlon[a],tdep[a]])-dc;shift2=result[3:]-np.array([tlat[b],tlon[b],tdep[b]])-dc;cos=np.cos(center[0]/rad)
  if any(abs(v[2])>2 or np.sqrt((v[1]*cos)**2+v[0]**2)*deg>2 for v in [shift1,shift2]):continue
  for ids,offset,tsign in [(members[a],result[:3]-[tlat[a],tlon[a],tdep[a]],-1),(members[b],result[3:]-[tlat[b],tlon[b],tdep[b]],1)]:
   lat[ids]+=offset[0];lon[ids]+=offset[1];dep[ids]=dep[ids]+F(offset[2]);tim[ids]=tim[ids]+F(tsign*dt/F(2))
  ids=np.sort(np.concatenate([members[a],members[b]]));cl[ids]=a;members[a]=ids;del members[b];tlat[a]=np.cumsum(d['latitude'][ids])[-1]/nt;tlon[a]=np.cumsum(d['longitude'][ids])[-1]/nt;tdep[a]=F(np.cumsum(d['depth'][ids],dtype=np.float32)[-1]/F(nt));la=np.cumsum(lat[ids])[-1]/nt;lo=np.cumsum(lon[ids])[-1]/nt;de=F(np.cumsum(dep[ids],dtype=np.float32)[-1]/F(nt));ti=F(np.cumsum(tim[ids],dtype=np.float32)[-1]/F(nt));lat[ids]+=tlat[a]-la;lon[ids]+=tlon[a]-lo;dep[ids]+=F(tdep[a]-de);tim[ids]-=ti
  accepted.append((pair,a,b,len(ids),rms,rmed))
  if verbose and len(accepted)%100==0:print('merges',len(accepted),'attempts',attempts,'cursor',cursor,'seconds',__import__('time').perf_counter()-start_time,flush=True)
  if stop_after and len(accepted)>=stop_after:break
 return dict(latitude=lat,longitude=lon,depth=dep,origin_correction=tim,cluster=cl,accepted=np.asarray(accepted),attempts=np.array(attempts))


def reconstruct(data,verbose=False,rmin=.6,nit=15):
 from .preprocessing import algorithm_inputs
 d=algorithm_inputs(data);out=reconstruct_catalog(d,verbose=verbose,rmin=rmin,nit=nit)
 result={'reconstruction':np.column_stack([out[k] for k in ['latitude','longitude','depth','origin_correction']])[None].astype(np.float64)}
 return result,out
