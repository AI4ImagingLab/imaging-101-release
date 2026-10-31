"""Inference diagnostics without selecting a Matplotlib backend."""
import numpy as np
def path_agreement(estimate,reference,valid):return float(np.mean(np.asarray(estimate)[valid]==np.asarray(reference)[valid]))
def transition_dwell_frames(transition):return 1/(1-np.diag(transition))
def model_comparison(log_likelihood,frames,states):
    parameters=2*states+(states-1)+states*(states-1)
    return dict(log_likelihood_per_frame=float(log_likelihood/frames),parameter_count=parameters,bic_style_plugin_score=float(-2*log_likelihood+parameters*np.log(frames)))
