"""Explicit midpoint aggregation and ordinary-kriging saddle-system solves.

Mathematical adaptation of RAINLINK1.31, Aart Overeem2024,
GPL-3.0-or-later; SciPy supplies only neighbor searches and linear algebra.
"""
import datetime
import numpy as np
from scipy.spatial import cKDTree
from .preprocessing import prepare
from .physics_model import retrieve_links

def climatological_variogram(timestamp,time_scale_hours=.25):
 """Return partial sill,nugget(mm²/h²),range(km), from independent gauge climatology."""
 date=datetime.datetime.strptime(str(int(timestamp))[:8],'%Y%m%d');day=date.timetuple().tm_yday;duration=float(time_scale_hours)
 radius=(15.51*duration**.09+2.06*duration**(-.12)*np.cos(2*np.pi/365*(day-7.37*duration**.22)))**4/1000
 sill=(.84*duration**(-.25)+.20*duration**(-.37)*np.cos(2*np.pi/365*(day-162*duration**(-.03))))**4
 return sill,.1*sill,radius

def spherical_variogram(distance,sill,nugget,radius):
 """Semivariogram zero exactly at identical points; finite nugget elsewhere."""
 d=np.asarray(distance);q=d/radius
 return np.where(d==0,0,nugget+sill*np.where(q<1,1.5*q-.5*q**3,1))

def midpoint_observations(data,rainfall,projected_links,timestamp):
 """Return unique projected midpoint(km) and mean simultaneous rain(mm/h)."""
 use=(data[:,1]==timestamp)&np.isfinite(rainfall);d=data[use];r=rainfall[use]
 if not len(d):raise ValueError('No valid rainfall observations for requested interval')
 geometry={row[0]:row[1:] for row in projected_links};points=np.array([geometry[x] for x in d[:,9]]);mid=(points[:,:2]+points[:,2:])/2
 unique,first,inverse=np.unique(mid,axis=0,return_index=True,return_inverse=True)
 means=np.bincount(inverse,weights=r)/np.bincount(inverse);order=np.argsort(first)
 return unique[order],means[order]

def interpolate(points,rainfall,grid,timestamp,*,method='OK',neighbors=50,chunk_size=256):
 """Return all grid rain intensities(mm/h), with original negative predictions clipped.

 OK uses explicit unit-sum constrained spherical semivariogram equations.
 IDW uses all midpoint observations with inverse-square Euclidean weights.
 """
 points=np.asarray(points);values=np.asarray(rainfall);grid=np.asarray(grid);out=np.empty(len(grid))
 if method=='IDW':
  for first in range(0,len(grid),chunk_size):
   dist=np.linalg.norm(grid[first:first+chunk_size,None,:]-points[None],axis=-1);equal=dist==0
   weights=np.divide(1,dist**2,out=np.zeros_like(dist),where=dist>0)
   exact=equal.any(axis=1);weights[exact]=equal[exact];out[first:first+len(dist)]=(weights@values)/weights.sum(axis=1)
  return out
 if method!='OK':raise ValueError('Unknown interpolation method')
 n=min(neighbors,len(points));_,near=cKDTree(points).query(grid,k=n)
 if n==1:near=near[:,None]
 sill,nugget,radius=climatological_variogram(timestamp)
 for first in range(0,len(grid),chunk_size):
  j=near[first:first+chunk_size];p=points[j];dist=np.linalg.norm(p[:,:,None,:]-p[:,None,:,:],axis=-1)
  matrix=np.ones((len(j),n+1,n+1));matrix[:,:n,:n]=spherical_variogram(dist,sill,nugget,radius);matrix[:,n,n]=0
  rhs=np.ones((len(j),n+1));rhs[:,:n]=spherical_variogram(np.linalg.norm(p-grid[first:first+len(j),None,:],axis=-1),sill,nugget,radius)
  weights=np.linalg.solve(matrix,rhs[...,None])[...,:n,0];out[first:first+len(j)]=np.maximum(0,np.sum(weights*values[j],axis=1))
 return out

def reconstruct(raw,*,method='OK',wet_dry=True,outlier=True,retain_fields=False,**retrieval_options):
 """Actively reconstruct the whole native96-interval daily field, mm accumulated.

 Return reconstruction(1,38063), full valid records and named retrieval stages;
 optional fields(96,38063) are fresh computations, never archive substitutions.
 """
 data=prepare(raw);geometry=raw['projected_links'][0];grid=raw['grid_xy'][0]
 stages=retrieve_links(data,geometry,raw['rain_coefficients'][0],wet_dry=wet_dry,outlier=outlier,**retrieval_options)
 stamps=np.unique(data[:,1]);stamps=stamps[(stamps>201109100800)&(stamps<=201109110800)];assert len(stamps)==96
 total=np.zeros(len(grid));fields=[]
 for stamp in stamps:
  points,values=midpoint_observations(data,stages['Rmean'],geometry,stamp);field=interpolate(points,values,grid,stamp,method=method);total+=field*.25
  if retain_fields:fields.append(field)
 result=dict(reconstruction=total[None],data=data,stages=stages,timestamps=stamps)
 if retain_fields:result['fields']=np.array(fields)
 return result
