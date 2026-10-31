import numpy as np

def load(path="data/raw_data.npz"):
 with np.load(path,allow_pickle=False) as f:return {k:f[k][0] for k in f.files}

def algorithm_inputs(data):
 q=data['initial_catalog'];pairs=data['event_pairs'];bounds=data['observation_bounds'];obs_pair=np.repeat(np.arange(len(pairs)),bounds[:,1]-bounds[:,0]);s=data['station_positions'][data['station_index']];mean=(q[pairs[:,0],:2]+q[pairs[:,1],:2])/2;center=mean[obs_pair];cos=np.cos((s[:,0]+center[:,0])/2/(180/3.1415927));distance=(np.sqrt(((s[:,1]-center[:,1])*cos)**2+(s[:,0]-center[:,0])**2)*111.1949266).astype(np.float32)
 return dict(latitude=q[:,0].copy(),longitude=q[:,1].copy(),depth=q[:,2].astype(np.float32),origin_correction=np.zeros(len(q),dtype=np.float32),pair_first=pairs[:,0]+1,pair_second=pairs[:,1]+1,start=bounds[:,0]+1,end=bounds[:,1],delta_time=data['differential_times'].astype(np.float32),cross_correlation=data['cross_correlation'].astype(np.float32),phase=data['phase'],station_latitude=s[:,0],station_longitude=s[:,1],station_distance=distance,travel_times=data['travel_time_table'],distance_grid=data['distance_grid'],depth_grid=data['depth_grid'])
