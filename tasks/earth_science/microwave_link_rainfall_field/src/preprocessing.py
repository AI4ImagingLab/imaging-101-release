"""Input-only author consistency checks; records retain chronological order.

Adapted mathematical workflow from RAINLINK1.31, Aart Overeem2024,
GPL-3.0-or-later. Exact provenance and notices accompany the task.
"""
import numpy as np

def prepare(raw):
 """Return complete valid chronological(N,11) float64 record table.

 Columns frequencyGHz,timestampUTC,Pmin/PmaxdBm,lengthkm,WGS84 endpoints,
 numericID,polarization(unknown0,V1,H2). Missing polarization remains NaN.
 """
 a=np.asarray(raw['link_records']);valid=np.asarray(raw['record_valid'])
 if a.shape!=(1,496180,11) or valid.shape!=a.shape or a.dtype!=np.dtype('float64') or valid.dtype!=np.dtype('uint8') or not np.isfinite(a).all() or np.any((valid!=0)&(valid!=1)):raise ValueError('Invalid complete raw record table')
 d=np.where(valid[0],a[0],np.nan)
 d=d[np.argsort(d[:,1],kind='stable')]
 d=d[(d[:,0]>=12.5)&(d[:,0]<=40.5)]
 keep=np.isfinite(d[:,:10]).all(axis=1)
 _,first=np.unique(d[:,9],return_index=True)
 for identifier in d[np.sort(first),9]:
  i=np.flatnonzero(d[:,9]==identifier)
  _,inverse,count=np.unique(d[i,1],return_inverse=True,return_counts=True)
  keep[i[count[inverse]>1]]=False
  if any(len(np.unique(d[i,c]))>1 for c in [0,4,5,6,7,8]):keep[i]=False
 return d[keep]
