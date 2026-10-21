"""Author Fourier-sideband reconstruction, normalized by independent empty field."""
from .preprocessing import padded_spectrum
from .physics_model import sideband_peak,aperture,recover_field,unwrap_for_review
def reconstruct(data,size_fraction=.5,kind='smooth disk'):
    spectrum=padded_spectrum(data['hologram']);carrier=sideband_peak(spectrum)
    window=aperture(spectrum.shape,carrier,size_fraction,kind)
    obj,filtered=recover_field(spectrum,window,carrier,data['hologram'].shape)
    bg_spectrum=padded_spectrum(data['empty_hologram'])
    bg,_=recover_field(bg_spectrum,window,carrier,data['empty_hologram'].shape)
    field=obj/bg
    import numpy as np
    phase_difference=np.angle(obj)-np.angle(bg)
    return {'reconstruction':field[None]}, {'object_field':obj,'background_field':bg,'fft_origin':spectrum,'fft_filtered':filtered,'window':window,'carrier':carrier,'phase_difference':phase_difference,'unwrapped':unwrap_for_review(phase_difference)}
