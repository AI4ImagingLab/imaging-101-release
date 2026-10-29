import numpy as np
def comparison_metrics(pdf,reference):
    return {'nrmse':float(np.sqrt(np.mean((pdf-reference)**2))/np.ptp(reference)),
            'ncc':float(np.dot(pdf,reference)/(np.linalg.norm(pdf)*np.linalg.norm(reference)))}
