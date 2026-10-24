from pathlib import Path
import importlib.util
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('standalone_judge',ROOT/'evaluation/judge.py')
judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)


def sample():
    return {'slip_rate':np.linspace(1.,2.,1299)[None],
            'basin_coefficient':np.array([[3.]]),'datum_offset':np.arange(5,dtype=float)[None]}


@pytest.mark.parametrize('field,value',[
    ('main_slip_ncc_boundary',True),('main_slip_ncc_boundary','0.9'),
    ('main_slip_ncc_boundary',-1.01),('main_slip_ncc_boundary',1.01),
    ('main_slip_ncc_boundary',float('nan')),
    ('main_slip_nrmse_boundary',False),('main_slip_nrmse_boundary','0.1'),
    ('main_slip_nrmse_boundary',-.01),
    ('observation_weighted_rms_boundary',-.01),
    ('absolute:deep_slip',True),('absolute:basin_coefficient',-.01),
    ('absolute:datum4',float('inf'))])
def test_invalid_evaluator_boundary_configuration(tmp_path,field,value):
    import json
    config=json.loads((ROOT/'evaluation/metrics.json').read_text())
    if field.startswith('absolute:'):config['absolute_boundaries'][field.split(':')[1]]=value
    else:config[field]=value
    (tmp_path/'metrics.json').write_text(json.dumps(config))
    p=tmp_path/'prediction.npz';np.savez(p,**sample())
    np.savez(tmp_path/'baseline_reference.npz',**sample())
    result=judge.evaluate(p,directory=tmp_path)
    assert result['status']=='error' and not result['passed']
    assert result['metrics']=={} and 'boundary' in result['errors'][0].lower()


def test_independent_known_metric_case():
    r=sample();p={k:2*v for k,v in r.items()}
    m=judge.field_metrics(p,r)
    expected=np.sqrt(np.mean(r['slip_rate'][0,:1292]**2))/np.ptp(r['slip_rate'][0,:1292])
    assert m['main_slip_ncc']==pytest.approx(1.,abs=2e-15)
    assert m['main_slip_nrmse']==pytest.approx(expected,rel=2e-15)
    assert m['basin_coefficient_maxabs']==3.
    assert m['datum4_maxabs']==4.


@pytest.mark.parametrize('case',['missing_key','wrong_shape','integer','float32','complex','nan','inf','object'])
def test_invalid_prediction_fields(tmp_path,case):
    a=sample()
    if case=='missing_key':del a['datum_offset']
    elif case=='wrong_shape':a['slip_rate']=a['slip_rate'][0]
    elif case in ('integer','float32','complex','object'):
        a['slip_rate']=a['slip_rate'].astype({'integer':int,'float32':np.float32,'complex':complex,'object':object}[case])
    else:a['slip_rate'][0,0]=np.nan if case=='nan' else np.inf
    p=tmp_path/'input.npz';np.savez(p,**a)
    result=judge.evaluate(p)
    assert result['status']=='error' and not result['passed']
    assert result['metrics']=={} and result['errors']


def test_missing_prediction_and_corrupt_npz(tmp_path):
    assert judge.evaluate(tmp_path/'absent.npz')['status']=='error'
    p=tmp_path/'corrupt.npz';p.write_bytes(b'not npz')
    assert judge.evaluate(p)['status']=='error'


def test_actual_truth_precedence_and_no_invalid_truth_fallback(tmp_path):
    baseline=sample();truth={k:v+1. for k,v in baseline.items()}
    np.savez(tmp_path/'baseline_reference.npz',**baseline)
    np.savez(tmp_path/'ground_truth.npz',**truth)
    selected=judge.load_reference(tmp_path)
    assert np.array_equal(selected['slip_rate'],truth['slip_rate'])
    assert judge.field_metrics(baseline,selected)['main_slip_nrmse']>0
    np.savez(tmp_path/'ground_truth.npz',slip_rate=truth['slip_rate'])
    with pytest.raises(ValueError,match='Missing required key'):
        judge.load_reference(tmp_path)


def test_degenerate_main_reference_rejected():
    r=sample();r['slip_rate'][:]=1.
    with pytest.raises(ValueError,match='Degenerate'):judge.field_metrics(r,r)


def test_report_write_error_exit2(tmp_path,capsys):
    p=tmp_path/'pred.npz';np.savez(p,**sample())
    obstacle=tmp_path/'file';obstacle.write_text('occupied')
    assert judge.main(['--prediction',str(p),'--report',str(obstacle/'report.json')])==2
    assert 'Could not write report' in capsys.readouterr().err


def test_unavailable_hidden_assets_are_errors(tmp_path):
    p=tmp_path/'pred.npz';np.savez(p,**sample())
    result=judge.evaluate(p,directory=tmp_path)
    assert result['status']=='error' and result['metrics']=={}


def test_native_cli_exit0_exit1_exit2_and_all_groups(tmp_path):
    with np.load(ROOT/'evaluation/baseline_reference.npz',allow_pickle=False) as z:
        reference={key:z[key] for key in judge.SHAPES}
    prediction=tmp_path/'pred.npz';report=tmp_path/'nested/report.json'
    np.savez(prediction,**reference)
    assert judge.main(['--prediction',str(prediction),'--report',str(report)])==0
    import json
    measured=json.loads(report.read_text())
    assert measured['passed'] and len(measured['metrics'])==12
    for key,index in [('slip_rate',0),('slip_rate',1292),('slip_rate',1297),('slip_rate',1298),('basin_coefficient',0),('datum_offset',0),('datum_offset',1),('datum_offset',2),('datum_offset',3),('datum_offset',4)]:
        poor={k:v.copy() for k,v in reference.items()};poor[key][0,index]+=1.
        np.savez(prediction,**poor)
        assert judge.main(['--prediction',str(prediction),'--report',str(report)])==1
    assert judge.main(['--prediction',str(tmp_path/'missing.npz'),'--report',str(report)])==2
