"""Independent scientific examples and standalone evaluator contract checks."""
from pathlib import Path
import importlib.util
import json
import shutil
import subprocess
import sys
import numpy as np
import pytest

TASK = Path(__file__).resolve().parents[2]


@pytest.fixture
def evaluator(tmp_path):
    folder = tmp_path / 'evaluation'
    folder.mkdir()
    for name in ['judge.py', 'metrics.json', 'ground_truth.npz']:
        shutil.copyfile(TASK / 'evaluation' / name, folder / name)
    with np.load(folder / 'ground_truth.npz', allow_pickle=False) as archive:
        p, q = archive['positions'][0].copy(), archive['quarter_turns'][0].copy()
    spec = importlib.util.spec_from_file_location('pose_contract_test_module', folder / 'judge.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, folder, p, q


CASES = [
    'common_C4_0','common_C4_1','common_C4_2','common_C4_3','big_endian','native',
    'missing_key','extra_key','missing_batch','wrong_turn_shape','float_positions',
    'float_nan','complex_positions','unsigned_positions','int32_positions','object_positions',
    'uint64_turns','negative_coordinates','nonzero_origin','negative_turns','out_of_range_turns',
    'collision','huge_coordinate','wrong_extent','missing_prediction','corrupt_archive',
    'poor_output','incompatible_gauges','twenty_corner_errors','twentyfive_corner_errors',
    'nan_boundary','bool_boundary','relaxed_boundary','undefined_metric',
    'missing_reference','malformed_reference',
]


@pytest.mark.parametrize('case', CASES)
def test_scientific_pose_and_input_contract(case, evaluator, tmp_path):
    module, folder, p, q = evaluator
    arrays = dict(positions=p[None], quarter_turns=q[None])
    expected = 2
    if case.startswith('common_C4_'):
        k = int(case[-1]); a = p.copy()
        for _ in range(k):
            a = np.column_stack((-a[:,1], a[:,0])); a -= a.min(axis=0)
        arrays = dict(positions=a[None], quarter_turns=((q+k)%4)[None]); expected = 0
    elif case == 'big_endian':
        arrays = {key:value.astype('>i8') for key,value in arrays.items()}; expected = 0
    elif case == 'native':
        expected = 0
    elif case == 'missing_key': arrays.pop('quarter_turns')
    elif case == 'extra_key': arrays['reconstruction'] = np.zeros(1)
    elif case == 'missing_batch': arrays['positions'] = p
    elif case == 'wrong_turn_shape': arrays['quarter_turns'] = q[None,:,None]
    elif case == 'float_positions': arrays['positions'] = p[None].astype(float)
    elif case == 'float_nan': arrays['positions'] = np.full((1,432,2), np.nan)
    elif case == 'complex_positions': arrays['positions'] = p[None].astype(complex)
    elif case == 'unsigned_positions': arrays['positions'] = p[None].astype(np.uint64)
    elif case == 'int32_positions': arrays['positions'] = p[None].astype(np.int32)
    elif case == 'object_positions': arrays['positions'] = p[None].astype(object)
    elif case == 'uint64_turns': arrays['quarter_turns'] = q[None].astype(np.uint64)
    elif case == 'negative_coordinates': arrays['positions'] = (p-1)[None]
    elif case == 'nonzero_origin': arrays['positions'] = (p+1)[None]
    elif case == 'negative_turns': arrays['quarter_turns'] = np.full((1,432),-1,np.int64)
    elif case == 'out_of_range_turns': arrays['quarter_turns'] = np.full((1,432),4,np.int64)
    elif case in {'collision','huge_coordinate','wrong_extent'}:
        a=p.copy()
        if case == 'collision': a[0]=a[1]
        elif case == 'huge_coordinate': a[0,0]=np.iinfo(np.int64).max
        else: a[0,0]=18
        arrays['positions']=a[None]
    elif case in {'missing_prediction','corrupt_archive'}:
        arrays=None
    elif case == 'poor_output':
        arrays=dict(positions=np.column_stack((np.arange(432)%18,np.arange(432)//18))[None].astype(np.int64),
                    quarter_turns=np.zeros((1,432),np.int64));expected=1
    elif case == 'incompatible_gauges':
        arrays['quarter_turns']=((q+1)%4)[None];expected=1
    elif case in {'twenty_corner_errors','twentyfive_corner_errors'}:
        height=4 if case=='twenty_corner_errors' else 5
        bad=(p[:,0]<height)&(p[:,1]<5);qq=q.copy();qq[bad]=(qq[bad]+1)%4
        arrays['quarter_turns']=qq[None];expected=0 if height==4 else 1
    elif case.endswith('_boundary') or case=='undefined_metric':
        config=json.loads((folder/'metrics.json').read_text(encoding='utf8'))
        if case=='undefined_metric':config['undefined_boundary']=0.95
        else:config['joint_piece_accuracy_boundary']={'nan_boundary':float('nan'),'bool_boundary':True,'relaxed_boundary':0.9}[case]
        (folder/'metrics.json').write_text(json.dumps(config),encoding='utf8')
    elif case=='missing_reference':
        # Move only this test's copied reference, preserving the actual task asset.
        (folder/'ground_truth.npz').rename(folder/'unavailable.npz')
    elif case=='malformed_reference':
        np.savez(folder/'ground_truth.npz', positions=p[None])
    else:
        raise AssertionError('Undefined test case')
    prediction=tmp_path/'reconstruction.npz'
    if arrays is not None:np.savez(prediction,**arrays)
    elif case=='corrupt_archive':prediction.write_bytes(b'not a zip archive')
    result,code=module.evaluate(prediction,folder)
    assert code==expected
    assert result['status']=={0:'pass',1:'fail',2:'error'}[expected]
    assert result['passed'] is (expected==0)
    if expected==2:assert result['metrics']=={} and result['errors']
    else:assert not result['errors']
    json.dumps(result,allow_nan=False)
    if case.startswith('common_C4_') or case in {'big_endian','native'}:
        assert result['metrics']['joint_piece_accuracy']==1
        assert result['metrics']['oriented_neighbor_accuracy']==1
    if case=='incompatible_gauges':
        candidates=result['alignment_candidates']
        assert max(x['metrics']['position_accuracy'] for x in candidates)==1
        assert max(x['metrics']['orientation_accuracy'] for x in candidates)==1
        assert result['metrics']['joint_piece_accuracy']==0
    if case=='twenty_corner_errors':
        assert result['metrics']['joint_piece_accuracy']==412/432
        assert result['metrics']['oriented_neighbor_accuracy']==782/822


@pytest.mark.parametrize('expected',[0,1,2,'write_error'])
def test_standalone_foreign_working_directory(expected,evaluator,tmp_path):
    _,folder,p,q=evaluator
    prediction=tmp_path/'submission.npz'
    if expected==1:
        p=np.column_stack((np.arange(432)%18,np.arange(432)//18)).astype(np.int64)
        q=np.zeros(432,np.int64)
    if expected!=2:np.savez(prediction,positions=p[None],quarter_turns=q[None])
    report=folder if expected=='write_error' else tmp_path/'new_report_parent'/'judge_report.json'
    command=[sys.executable,str(folder/'judge.py'),'--prediction',str(prediction),'--report',str(report)]
    process=subprocess.run(command,cwd=tmp_path.parent,capture_output=True,text=True)
    assert process.returncode==(2 if expected=='write_error' else expected)
    if expected=='write_error':
        assert 'Could not write evaluator report' in process.stderr
    else:
        body=json.loads(report.read_text(encoding='utf8'))
        assert body['status']=={0:'pass',1:'fail',2:'error'}[expected]
