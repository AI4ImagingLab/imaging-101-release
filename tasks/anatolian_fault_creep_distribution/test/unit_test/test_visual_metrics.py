import numpy as np
import pytest
from src.visualization import reference_metrics,signed_norm


def test_vector_extent_contains_all_tips():
    import matplotlib.pyplot as plt
    from src.visualization import vector_extent
    fig,ax=plt.subplots()
    vector_extent(ax,np.array([[0.,0.],[1.,1.]]),np.array([[5.,-2.],[-3.,4.]]),.25)
    assert ax.get_xlim()[0]<-11 and ax.get_xlim()[1]>20
    assert ax.get_ylim()[0]<-8 and ax.get_ylim()[1]>17
    plt.close(fig)


def test_known_cosine_and_range_metrics():
    a=np.array([1.,2.,4.]);b=np.array([1.,2.,3.])
    m=reference_metrics(a,b)
    assert m['ncc']==pytest.approx(17./np.sqrt(21.*14.))
    assert m['nrmse']==pytest.approx(np.sqrt(1./3.)/2.)
    assert reference_metrics(np.zeros(3),b)['ncc']==0.
    with pytest.raises(ValueError):reference_metrics(a,np.ones(3))


def test_signed_colour_zero_neutral_and_backend_unchanged():
    norm=signed_norm(np.array([-2.,1.]))
    assert norm.vmin==-2. and norm.vmax==2. and norm(0.)==.5
    zero=signed_norm(np.zeros(3));assert zero.vmin==-zero.vmax and zero(0.)==.5
    import matplotlib,importlib
    before=matplotlib.get_backend();import src.main
    importlib.reload(src.main)
    assert matplotlib.get_backend()==before
