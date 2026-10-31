"""Independent conditional hypocenter-likelihood evaluator; no solver imports."""
import argparse,json,sys
from pathlib import Path
import numpy as np

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

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--prediction',required=True);ap.add_argument('--report',required=True);a=ap.parse_args();r=dict(status='error',passed=False,metrics={},thresholds={},errors=[]);exitcode=2
 try:
  here=Path(__file__).resolve().parent;cfg=json.loads((here/'metrics.json').read_text(encoding='utf8'));ts=cfg.get('thresholds');expected={'p_mean_absolute_residual_s','s_mean_absolute_residual_s','maximum_component_centroid_offset_km','maximum_component_mean_origin_correction_s'}
  if not isinstance(ts,dict) or set(ts)!=expected:raise ValueError('invalid evaluator metric set')
  for v in ts.values():
   if not isinstance(v,dict) or set(v)!={'comparison','boundary'} or v['comparison']!='<=':raise ValueError('invalid evaluator comparison')
   b=v['boundary']
   if isinstance(b,bool) or not isinstance(b,(int,float)) or not np.isfinite(b) or b<=0:raise ValueError('invalid evaluator boundary')
  r['thresholds']=ts
  with np.load(a.prediction,allow_pickle=False) as f:
   if 'reconstruction' not in f:raise ValueError('missing reconstruction')
   q=f['reconstruction']
   if q.shape!=(1,1616,4) or q.dtype!=np.float64 or not np.isfinite(q).all():raise ValueError('require finite realfloat64 (1,1616,4)')
   q=q[0]
  with np.load(here.parent/'data/raw_data.npz',allow_pickle=False) as f:data={k:f[k][0] for k in f.files}
  if np.any(abs(q[:,0])>90) or np.any(abs(q[:,1])>180) or np.any(q[:,2]<0) or np.any(q[:,2]>26) or np.any(abs(q[:,3])>50):r.update(status='fail',errors=['outside public physical coordinate/depth/time domain']);exitcode=1
  else:
   try:measured=physical_metrics(data,q)
   except ValueError as e:r.update(status='fail',errors=[str(e)]);exitcode=1
   else:
    if not all(np.isfinite(v) for v in measured.values()):raise ValueError('nonfinite evaluator metric')
    passed=all(measured[k]<=v['boundary'] for k,v in ts.items());r.update(status='pass' if passed else 'fail',passed=passed,metrics=measured);exitcode=0 if passed else 1
 except Exception as e:r.update(status='error',passed=False,metrics={},errors=[str(e)]);exitcode=2
 try:
  p=Path(a.report);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(r,indent=2,allow_nan=False),encoding='utf8')
 except Exception as e:print('report write failed: '+str(e).encode('ascii','backslashreplace').decode(),file=sys.stderr);return 2
 return exitcode
if __name__=='__main__':raise SystemExit(main())
