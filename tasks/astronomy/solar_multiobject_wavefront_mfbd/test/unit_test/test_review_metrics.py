"""Known physical visualization statistics and backend neutrality."""
import importlib
import numpy as np
import pytest


def test_rms_contrast_known_case():
    from src.visualization import contrast
    assert contrast(np.array([1., 3.])) == pytest.approx(50.)
    assert contrast(np.ones((8, 8))) == 0.
    with pytest.raises(ValueError):
        contrast(np.zeros((8, 8)))


def test_spectral_power_parseval_and_pixel_units():
    from src.visualization import radial_power
    x = np.arange(32)
    image = 1 + .1*np.cos(2*np.pi*3*x/32)[None, :]*np.ones((32, 1))
    frequency, power = radial_power(image)
    peak = int(np.argmax(power))
    assert frequency[peak] == pytest.approx(3.5/(32*.059))
    assert np.all(power >= 0)
    radius = np.hypot(*np.meshgrid(np.fft.fftfreq(32, d=.059), np.fft.fftfreq(32, d=.059)))
    labels = np.floor(radius*(32*.059)).astype(int)
    counts = np.bincount(labels.ravel())
    assert np.sum(power*counts) == pytest.approx(np.var(image))


def test_importing_main_does_not_change_backend():
    import matplotlib
    before = matplotlib.get_backend()
    importlib.import_module('src.main')
    assert matplotlib.get_backend() == before
