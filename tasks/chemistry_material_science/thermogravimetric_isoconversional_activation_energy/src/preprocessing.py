"""Measured thermograms and the published monotone conversion preprocessing."""
from pathlib import Path
import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import linregress

def load_data(path='data/raw_data.npz'):
    with np.load(Path(path),allow_pickle=False) as z:
        traces=[z[f'trace_{i+1}'][0].copy() for i in range(5)]
    for x in traces:
        if x.ndim!=2 or x.shape[1]!=3 or not np.all(np.isfinite(x)) or np.any(np.diff(x[:,0])<=0):
            raise ValueError('Each trace must have increasing time and three finite columns')
    return traces

def polynomial_derivative(t,y,half_window=.5):
    """Cubic least squares in absolute minutes; nearest-endpoint truncated window.

    Explicit extraction of derivative 0.6.3 SavitzkyGolay conventions.
    """
    result=np.empty(len(t))
    for i in range(len(t)):
        left=np.argmin(np.abs(t-(t[i]-half_window)))
        right=np.argmin(np.abs(t-(t[i]+half_window)))
        w=np.polynomial.polynomial.polyfit(t[left:right+1],y[left:right+1],3)
        result[i]=np.sum([j*w[j]*np.power(t[i],j-1) for j in range(1,4)])
    return result

def prepare(traces,half_window=.5):
    """Return monotone traces, rounded cubic isoconversion tables and metadata."""
    beta=np.array([linregress(x[:,0],x[:,1]).slope for x in traces])
    initial=np.array([x[0,1]+273.15 for x in traces]);monotone=[];segments=[]
    for x in traces:
        temp=x[:,1]+273.15;mask=(temp>300)&(temp<400)
        t=x[mask,0];temp=temp[mask];m=x[mask,2]
        alpha=(m[0]-m)/(m[0]-m[-1]);rate=polynomial_derivative(t,alpha,half_window)
        keep=[0]
        for i in range(len(alpha)):
            if alpha[i]>alpha[keep[-1]]:keep.append(i)
        dr=rate[keep].copy();dr[0]=rate[1]
        monotone.append(np.column_stack([alpha[keep],temp[keep],t[keep],dr]))
        segments.append(np.column_stack([t,temp,m,alpha,rate]))
    alpha=monotone[-1][:,0];tables=[]
    for col in [1,2,3]:
        values=[interp1d(v[:,0],v[:,col],kind='cubic',bounds_error=False,fill_value='extrapolate')(alpha) for v in monotone[:-1]]+[monotone[-1][:,col]]
        tables.append(np.round(np.array(values).T,4))
    advanced=np.arange(alpha[0],alpha[-1],.01)
    at=np.round(np.array([interp1d(v[:,0],v[:,1],kind='cubic',bounds_error=False,fill_value='extrapolate')(advanced) for v in monotone]).T,4)
    az=np.round(np.array([interp1d(v[:,0],v[:,2],kind='cubic',bounds_error=False,fill_value='extrapolate')(advanced) for v in monotone]).T,4)
    pick=(alpha>=.1)&(alpha<=.9);apick=(advanced>=.09)&(advanced<=.9)
    return dict(beta=beta,initial_temperature=initial,monotone=monotone,segments=segments,alpha=alpha[pick],temperature=tables[0][pick],rate=tables[2][pick],advanced_alpha=advanced[apick],advanced_temperature=at[apick],advanced_time=az[apick])
