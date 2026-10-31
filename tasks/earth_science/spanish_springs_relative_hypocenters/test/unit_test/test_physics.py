from pathlib import Path
import sys,numpy as np
T=Path(__file__).resolve().parents[2];sys.path.insert(0,str(T))
from src.preprocessing import load
from src.physics_model import travel_time,physical_metrics,observation_components

def test_independent_layered_vertical_travel_time_and_s_speed():
 from scipy.integrate import quad
 d=load(T/'data/raw_data.npz');v=d['velocity_model'];z=d['depth_grid'];F=np.float32;R=F(6371);ideal=[[],[]];rounded=[[],[]]
 for dep in z:
  for phase in range(2):
   exact=0.;integral=0.
   for j in range(len(v)-1):
    top,bottom=v[j,0],v[j+1,0]
    if bottom<=top or top>=dep:continue
    bottom=min(bottom,float(dep));vp=v[j,1];factor=1. if phase==0 else 1.732;exact+=(bottom-top)/vp*factor
    # Independently integrate exponential slowness through the actual finite-precision flat-Earth calibration. Source VZFILLIN serializes S speed to four decimal km/s.
    for a in np.arange(top,bottom,1.):
     b=min(a+1,bottom);za=F(-R*np.log(F((R-F(a))/R)));zb=F(-R*np.log(F((R-F(b))/R)));h=float(F(zb-za));speed=F(vp) if phase==0 else F(round(vp/1.732,4));ua=float(F(F(1)/F(speed*F(R/F(R-F(a))))));ub=float(F(F(1)/F(speed*F(R/F(R-F(b))))));integral+=quad(lambda t:ua*np.exp(np.log(ub/ua)*t/h),0,h,epsabs=1e-13)[0]
   ideal[phase].append(exact);rounded[phase].append(integral)
 for phase in range(2):
  np.testing.assert_allclose(d['travel_time_table'][0,:,phase],rounded[phase],rtol=0,atol=2e-6)
  # Analytic Earth-radius float32 roundoff scale, not tuned to the observed discrepancy.
  budget=float(R)*np.finfo(np.float32).eps/(v[:,1].min()/(1. if phase==0 else 1.732))
  assert np.max(np.abs(d['travel_time_table'][0,:,phase]-ideal[phase]))<budget

def test_analytic_constant_speed_vertical_interpolation_and_negative_depth():
 x=np.array([0,2,4,6],np.float32);z=np.array([0,1,2,3],np.float32);table=np.empty((4,4,2),np.float32)
 for p,v in enumerate([5.,3.]):table[:,:,p]=np.sqrt(x[:,None]**2+z[None,:]**2)/v
 got=travel_time(table,x,z,np.array([1,2]),np.array([0.,0.],np.float32),np.array([1.5,1.5],np.float32));np.testing.assert_allclose(got,[1.5/5,1.5/3],rtol=1e-7)
 # At positive range a negative-depth correction adds |depth| / surface vertical speed.
 p=travel_time(table,x,z,np.array([1]),np.array([4.],np.float32),np.array([-1.],np.float32));np.testing.assert_allclose(p,[1.],rtol=1e-7)

def test_physical_metric_analytic_time_difference_and_component_anchor():
 initial=np.array([[39.,-119.,2.,0.],[39.,-119.,2.,0.]]);x=np.array([0,2,4,6],np.float32);z=np.array([0,1,2,3],np.float32);tab=np.empty((4,4,2),np.float32)
 for p,v in enumerate([5.,3.]):tab[:,:,p]=np.sqrt(x[:,None]**2+z[None,:]**2)/v
 d=dict(initial_catalog=initial,event_pairs=np.array([[0,1]]),observation_bounds=np.array([[0,2]]),station_positions=np.array([[39.,-119.]]),station_index=np.array([0,0]),phase=np.array([1,2]),travel_time_table=tab,distance_grid=x,depth_grid=z,cross_correlation=np.ones(2),differential_times=np.zeros(2));q=initial.copy();q[:,3]=[.1,-.1];m=physical_metrics(d,q);np.testing.assert_allclose([m['p_mean_absolute_residual_s'],m['s_mean_absolute_residual_s']],[.2,.2],atol=1e-15);assert m['maximum_component_centroid_offset_km']==0 and m['maximum_component_mean_origin_correction_s']==0

def test_graph_anchors_are_public_not_fitted_clusters():
 d=load(T/'data/raw_data.npz');labels=observation_components(d);assert len(np.unique(labels))==818;assert sorted(np.bincount(labels))[-2:]==[2,798]
 with np.load(T/'evaluation/baseline_reference.npz') as f:m=physical_metrics(d,f['reconstruction'][0])
 assert m['maximum_component_centroid_offset_km']<5e-6 and m['maximum_component_mean_origin_correction_s']<1e-7
 assert d['differential_times'].dtype==np.float64 and d['cross_correlation'].dtype==np.float64 and d['travel_time_table'].dtype==np.float32
