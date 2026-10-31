"""Review metrics and plotting helpers with no backend mutation."""
import numpy as np
def comparison_metrics(estimate,reference):
 a=np.asarray(estimate).ravel();b=np.asarray(reference).ravel();norm=np.linalg.norm(a)*np.linalg.norm(b);scale=np.ptp(b)
 if not norm or not scale:raise ValueError('Degenerate metric reference')
 return dict(ncc=float(np.dot(a,b)/norm),nrmse=float(np.sqrt(np.mean((a-b)**2))/scale),max_abs=float(np.max(abs(a-b))))
def directional_variation(image):
 a=np.asarray(image);return dict(horizontal=float(np.mean(abs(np.diff(a,axis=1)))),vertical=float(np.mean(abs(np.diff(a,axis=0)))))
def plot_field(axis,image,title,maximum=None):
 from .physics_model import Z_MAX,DX
 im=axis.imshow(image,extent=[0,DX*(image.shape[1]-1),Z_MAX,0],aspect='auto',vmin=0,vmax=maximum,cmap='viridis');axis.set(xlabel='Lateral coordinate / mm',ylabel='Author axial coordinate / mm',title=title);return im
