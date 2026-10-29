"""Review-only line residual and plotted field helpers; no backend mutation."""
import numpy as np
def line_residuals(points,line_ids):
    residual=np.empty(len(points))
    for k in np.unique(line_ids):
        mask=line_ids==k;q=points[mask];q=q-q.mean(axis=0)
        _,_,vh=np.linalg.svd(q,full_matrices=False)
        residual[mask]=q@vh[-1]
    return residual
