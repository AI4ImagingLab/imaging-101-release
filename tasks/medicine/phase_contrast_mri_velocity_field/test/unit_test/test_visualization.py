import numpy as np
from matplotlib.figure import Figure
from src.visualization import plot_component_means

def test_full_temporal_curve_and_direction_association():
    # Each direction/time has a distinct constant spatial field: its known
    # mean is independent of any source reduction/plot implementation.
    t=np.arange(7)[:,None];expected=t+10*np.arange(3)[None]
    field=np.broadcast_to(expected[:,:,None,None,None],(7,3,2,3,4)).astype(float)
    source=expected+.01;ax=Figure().subplots()
    means=plot_component_means(ax,field,source)
    np.testing.assert_array_equal(means,expected)
    for d in range(3):
        np.testing.assert_array_equal(ax.lines[2*d].get_ydata(),expected[:,d]*100)
        np.testing.assert_array_equal(ax.lines[2*d+1].get_ydata(),source[:,d]*100)
        assert len(ax.lines[2*d].get_xdata())==7
        assert ax.lines[2*d].get_color()==ax.lines[2*d+1].get_color()

def test_three_distinct_directions_are_not_three_time_samples():
    expected=np.arange(27)[:,None]*np.array([[1.,-2.,3.]])
    field=np.broadcast_to(expected[:,:,None,None,None],(27,3,1,1,1))
    ax=Figure().subplots();plot_component_means(ax,field,expected)
    assert all(len(line.get_ydata())==27 for line in ax.lines)
