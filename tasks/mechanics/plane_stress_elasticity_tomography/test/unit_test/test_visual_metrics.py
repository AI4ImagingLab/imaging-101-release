import numpy as np
import pytest
from src.visualization import quality_metrics,comparison_metrics
from src.preprocessing import load_data
from evaluation.judge import array_metrics
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]

def test_visual_metric_known_case():
    reference=np.array([1.,2.,4.]);estimate=np.array([1.,3.,4.])
    actual=quality_metrics(estimate,reference)
    assert actual['nrmse_vs_ref']==pytest.approx(np.sqrt(1/3)/3)
    assert actual['ncc_vs_ref']==pytest.approx(23/np.sqrt(26*21))
    assert actual==array_metrics(estimate,reference)

@pytest.mark.parametrize('reference',[[1.,1.],[0.,0.]])
def test_visual_metric_rejects_degenerate_reference(reference):
    with pytest.raises(ValueError):quality_metrics(np.array([1.,2.]),reference)
