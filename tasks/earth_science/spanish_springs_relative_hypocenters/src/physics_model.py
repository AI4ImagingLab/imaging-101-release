"""GrowClust explicit travel-time interpolation and relative L1 grid search.
Adapted from GrowClust, Copyright2022DanielTrugman, GPL-3.0-or-later.
"""
import numpy as np
F=np.float32
DEG=F(111.19493);RAD=F(F(180)/F(3.1415927))

def travel_time(table,xgrid,zgrid,phase,distance,depth):
 distance=np.asarray(distance,dtype=np.float32);depth=np.asarray(depth,dtype=np.float32);phase=np.broadcast_to(np.asarray(phase,dtype=np.int64),distance.shape)-1;z=np.maximum(depth,F(0));dx=xgrid[1]-xgrid[0];dz=zgrid[1]-zgrid[0]
 if np.any(z>zgrid[-1]) or np.any(distance>xgrid[-1]) or np.any(distance<0):raise ValueError('outside known travel-time domain')
 ix=np.minimum(((distance-xgrid[0])/dx).astype(np.int64),len(xgrid)-2);iz=np.minimum(((z-zgrid[0])/dz).astype(np.int64),len(zgrid)-2)
 fx=(distance-xgrid[ix])/dx;fz=(z-zgrid[iz])/dz;a=table[ix,iz,phase];b=table[ix+1,iz,phase];c=table[ix,iz+1,phase];d=table[ix+1,iz+1,phase]
 out=(a+fx*(b-a))+fz*((c+fx*(d-c))-(a+fx*(b-a)))
 bad=(a==0)|(b==0)|(c==0)|(d==0)
 if np.any(bad):
  for at in zip(*np.where(bad)):
   xx=distance[at];zz=int(iz[at]);pp=int(phase[at]);ys=[]
   for k in [zz,zz+1]:
    allowed=np.flatnonzero((table[:-1,k,pp]!=0)&(table[1:,k,pp]!=0));j=allowed[np.argmin(np.abs((xgrid[allowed]+xgrid[allowed+1])/F(2)-xx))];t1=table[j,k,pp];t2=table[j+1,k,pp];ys.append(t1+(xx-xgrid[j])/dx*(t2-t1))
   out[at]=ys[0]+fz[at]*(ys[1]-ys[0])
 vel=dz/(table[0,1,phase]-table[0,0,phase]);out=out+np.where(depth<0,np.abs(depth)/vel,F(0));return out.astype(np.float32)

def relocate_pair(center,g,table,xgrid,zgrid,nit=15,box=3.):
 lat,lon,dep=center;dep=F(dep);coslat=F(np.cos(lat/float(RAD)));dlat=float(F(F(.5)*F(box)/DEG));dlon=dlat/float(coslat);zbox=F(box)
 if dep<0:zbox=min(F(1),zbox)
 elif dep<F(box)/F(2):zbox=max(F(1),F(2)*dep)
 dd=F(F(.5)*zbox);offlat=0.;offlon=0.;offdep=F(0);trials=np.array([(a,b,c) for a in [-1,0,1] for b in [-1,0,1] for c in [-1,0,1]],dtype=np.int32)
 def residual(la1,lo1,de1,la2,lo2,de2):
  tt=[]
  for la,lo,de,num in [(la1,lo1,de1,1),(la2,lo2,de2,2)]:
   dy=(la[:,None]+g['latitude'+str(num)][None,:]-g['station_latitude']).astype(np.float32);dx=((lo[:,None]+g['longitude'+str(num)][None,:]-g['station_longitude'])*float(coslat)).astype(np.float32);distance=np.sqrt(dx*dx+dy*dy)*DEG;z=(de[:,None]+g['depth'+str(num)][None,:]).astype(np.float32)
   tt.append(travel_time(table,xgrid,zgrid,g['phase'],distance,z))
  pred=(tt[1]+g['time2'])-(tt[0]+g['time1']);return g['delta_time']-pred
 for iteration in range(nit):
  la1=lat+offlat+dlat*trials[:,0];la2=lat-offlat-dlat*trials[:,0];lo1=lon+offlon+dlon*trials[:,1];lo2=lon-offlon-dlon*trials[:,1];de1=F(dep+offdep)+F(dd*trials[:,2]);de2=F(dep-offdep)-F(dd*trials[:,2]);r=residual(la1,lo1,de1,la2,lo2,de2);med=np.median(r,axis=1);fit=np.cumsum(np.abs(r-med[:,None]),axis=1,dtype=np.float32)[:,-1];best=int(np.argmin(fit));a=la1[best];b=lo1[best];c=de1[best];d=la2[best];e=lo2[best];f=de2[best];time=med[best];offlat=a-lat;offlon=b-lon;offdep=F(c-dep);dlat*=float(F(.67));dlon*=float(F(.67));dd=F(dd*F(.67))
 r=residual(np.array([a]),np.array([b]),np.array([c]),np.array([d]),np.array([e]),np.array([f]))[0]-time;dy=F(d-a);dx=F((e-b)*float(coslat));dist=F(np.sqrt(dx*dx+dy*dy)*DEG);sep=F(np.sqrt(dist*dist+F(f-c)**2));rms=F(np.sqrt(np.cumsum(r*r,dtype=np.float32)[-1]/F(len(r))));return np.array([a,b,c,d,e,f],dtype=np.float64),time,r,sep,rms,F(np.median(np.abs(r))),dlat/float(F(.67))*float(DEG)


def physical_residuals(data,q):
 """Double-precision declared public travel-time interpolant, all observations."""
 pairs=data['event_pairs'];bounds=data['observation_bounds'];obs_pair=np.repeat(np.arange(len(pairs)),bounds[:,1]-bounds[:,0]);ia=pairs[obs_pair,0];ib=pairs[obs_pair,1];s=data['station_positions'][data['station_index']];phase=data['phase']-1;tab=data['travel_time_table'].astype(np.float64);x=data['distance_grid'].astype(np.float64);z=data['depth_grid'].astype(np.float64);cos=np.cos(.5*(q[ia,0]+q[ib,0])/(180/3.1415927));pred=[]
 for ev in (ia,ib):
  dy=q[ev,0]-s[:,0];dx=(q[ev,1]-s[:,1])*cos;r=np.sqrt(dx*dx+dy*dy)*111.1949266;depth=q[ev,2]
  if np.any(r>x[-1]) or np.any(depth<z[0]) or np.any(depth>z[-1]):raise ValueError('prediction outside public travel-time domain')
  ix=np.minimum(((r-x[0])/(x[1]-x[0])).astype(np.int64),len(x)-2);iz=np.minimum(((depth-z[0])/(z[1]-z[0])).astype(np.int64),len(z)-2);fx=(r-x[ix])/(x[1]-x[0]);fz=(depth-z[iz])/(z[1]-z[0]);a=tab[ix,iz,phase];b=tab[ix+1,iz,phase];c=tab[ix,iz+1,phase];d=tab[ix+1,iz+1,phase];top=a+fx*(b-a);bottom=c+fx*(d-c);value=top+fz*(bottom-top);bad=(a==0)|(b==0)|(c==0)|(d==0)
  for at in np.flatnonzero(bad):
   vv=[]
   for k in (iz[at],iz[at]+1):
    allowed=np.flatnonzero((tab[:-1,k,phase[at]]!=0)&(tab[1:,k,phase[at]]!=0));j=allowed[np.argmin(np.abs((x[allowed]+x[allowed+1])/2-r[at]))];vv.append(tab[j,k,phase[at]]+(r[at]-x[j])/(x[1]-x[0])*(tab[j+1,k,phase[at]]-tab[j,k,phase[at]]))
   value[at]=vv[0]+fz[at]*(vv[1]-vv[0])
  pred.append(value+q[ev,3])
 initial=data['initial_catalog'];center=.5*(initial[pairs[:,0],:2]+initial[pairs[:,1],:2]);center=center[obs_pair];co=np.cos(.5*(s[:,0]+center[:,0])/(180/3.1415927));dist=np.sqrt(((s[:,1]-center[:,1])*co)**2+(s[:,0]-center[:,0])**2)*111.1949266;mask=(data['cross_correlation']>=.6)&(dist<=80)
 return data['differential_times']-(pred[1]-pred[0]),mask

def observation_components(data):
 from scipy.sparse import coo_matrix
 from scipy.sparse.csgraph import connected_components
 n=len(data['initial_catalog']);pairs=data['event_pairs'];g=coo_matrix((np.ones(len(pairs)),(pairs[:,0],pairs[:,1])),shape=(n,n));return connected_components(g,directed=False)[1]

def physical_metrics(data,q):
 residual,mask=physical_residuals(data,q);labels=observation_components(data);count=np.bincount(labels);initial=data['initial_catalog'];delta=q-initial;means=np.column_stack([np.bincount(labels,weights=delta[:,k])/count for k in range(4)]);ref_lat=np.bincount(labels,weights=initial[:,0])/count;offset=np.sqrt((means[:,0]*111.1949266)**2+(means[:,1]*np.cos(ref_lat/(180/3.1415927))*111.1949266)**2+means[:,2]**2)
 return dict(p_mean_absolute_residual_s=float(np.mean(np.abs(residual[mask&(data['phase']==1)]))),s_mean_absolute_residual_s=float(np.mean(np.abs(residual[mask&(data['phase']==2)]))),maximum_component_centroid_offset_km=float(offset.max()),maximum_component_mean_origin_correction_s=float(np.abs(means[:,3]).max()))
