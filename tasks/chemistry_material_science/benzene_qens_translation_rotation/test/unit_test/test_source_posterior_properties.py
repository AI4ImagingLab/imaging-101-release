"""Statistical summaries only: no seed/chain or stochastic-value matching."""
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]

def test_full_source_joint_posterior_summary_properties():
    with np.load(ROOT/'notebooks/reference_outputs/source_joint_posterior_summary.npz') as f:
        levels=f['percentile_levels']
        np.testing.assert_array_equal(levels,[.15,2.5,16,84,97.5,99.85])
        for n,nq,ne in [('360',12,92),('197',14,169)]:
            mean=f['mean_'+n];bands=f['bands_'+n]
            assert mean.shape==(1,nq,ne) and bands.shape==(1,nq,6,ne)
            assert np.isfinite(mean).all() and np.isfinite(bands).all()
            assert np.all(np.diff(bands,axis=2)>=0)
        centers=f['parameter_histogram_centers'];density=f['parameter_histogram_density']
        assert centers.shape==density.shape==(31,100)
        assert np.isfinite(centers).all() and np.isfinite(density).all() and np.all(density>=0)
        widths=np.diff(centers,axis=1)
        assert np.all(widths>0)
        # Histogram is a density estimate. Its integrated normalization, rather
        # than its random draw-dependent bin heights, is the required property.
        np.testing.assert_allclose(np.sum(density,axis=1)*widths[:,0],1,rtol=1e-10)
