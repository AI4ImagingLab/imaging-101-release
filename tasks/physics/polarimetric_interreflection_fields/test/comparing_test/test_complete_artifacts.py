from pathlib import Path
import json
import numpy as np
ROOT=Path(__file__).resolve().parents[2]

def test_actual_complete_output_preserves_all_original_fields_and_frozen_parity():
    with np.load(ROOT/'output/reconstruction.npz',allow_pickle=False) as z:a=z['reconstruction'][0]
    with np.load(ROOT/'evaluation/baseline_reference.npz',allow_pickle=False) as z:b=z['reconstruction'][0]
    assert a.shape==b.shape==(5,2048,2448) and np.isfinite(a).all()
    for x,y in zip(a,b):
        assert np.linalg.norm(x-y)/np.linalg.norm(y)<=1e-8
        assert abs(x-y).max()/max(1,abs(y).max())<=1e-8
    assert np.any(a[0]<0) and np.any(a[2]<0) and np.any(a[4]<0)

def test_active_Notebook_computed_full_reference_review_and_all_six_cells():
    path=ROOT/'notebooks/polarimetric_interreflection_fields.ipynb';nb=json.loads(path.read_text(encoding='utf-8'))
    counts=[c.get('execution_count') for c in nb['cells'] if c['cell_type']=='code']
    assert counts==list(range(1,7))
    assert not [o for c in nb['cells'] for o in c.get('outputs',[]) if o.get('output_type')=='error']
    with np.load(ROOT/'notebooks/reference_outputs/fresh_review.npz',allow_pickle=False) as z:
        assert np.max(z['channel_relative_l2'])<=1e-8
        assert z['fresh_rms_map'].shape==z['fresh_nine_rms_map'].shape==(2048,2448)
        np.testing.assert_allclose(z['fresh_forward_relative_l2'],0.15243698953044157,rtol=1e-10)
        np.testing.assert_allclose(z['fresh_nine_basis_relative_l2'],0.02948077533200727,rtol=1e-10)
        assert z['fresh_nine_basis_relative_l2']<z['fresh_forward_relative_l2']

def test_generated_Judge_report_uses_all_five_field_metrics_and_full_forward():
    report=json.loads((ROOT/'output/judge_report.json').read_text(encoding='utf-8'))
    assert report['status']=='pass' and report['passed'] and not report['errors']
    assert len(report['metrics'])==6
    assert .15<report['metrics']['observation_relative_l2']<.16
    assert all(t['comparison']=='<=' for t in report['thresholds'].values())
