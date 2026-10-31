from pathlib import Path
import importlib.util,json,os,shutil
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('judge_schema',ROOT/'evaluation/judge.py');judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)
KEYS=['pixel_objective_ratio_boundary', 'pixel_objective_absolute_floor', 'forward_relative_l2_boundary']
BAD=[True,False,'1','',[],[1],{},None,float('nan'),float('inf'),-float('inf'),-1,0]

def assets(tmp_path):
 (tmp_path/'data').mkdir();os.link(ROOT/'data/raw_data.npz',tmp_path/'data/raw_data.npz');shutil.copytree(ROOT/'evaluation',tmp_path/'evaluation');return json.loads((tmp_path/'evaluation/metrics.json').read_text(encoding='utf8'))
def run(tmp_path):
 return judge.evaluate(ROOT/'output/reconstruction.npz',tmp_path)

@pytest.mark.parametrize('key',KEYS)
@pytest.mark.parametrize('bad',BAD)
def test_malformed_numeric_config(tmp_path,key,bad):
 cfg=assets(tmp_path)
 if key=='objective_boundary' and type(cfg[key]) is list:cfg[key][0]=bad
 else:cfg[key]=bad
 (tmp_path/'evaluation/metrics.json').write_text(json.dumps(cfg),encoding='utf8');report,code=run(tmp_path);assert code==2 and report['status']=='error' and report['metrics']=={} and report['errors'];json.dumps(report,allow_nan=False)

@pytest.mark.parametrize('key',KEYS)
def test_missing_required_boundary(tmp_path,key):
 cfg=assets(tmp_path);del cfg[key];(tmp_path/'evaluation/metrics.json').write_text(json.dumps(cfg),encoding='utf8');report,code=run(tmp_path);assert code==2 and report['metrics']=={}

def test_valid_config_and_missing_config_file(tmp_path):
 assets(tmp_path);assert run(tmp_path)[1]==0;(tmp_path/'evaluation/metrics.json').unlink();report,code=run(tmp_path);assert code==2 and report['metrics']=={}

@pytest.mark.parametrize('bad',BAD)
def test_truth_boundary_numeric_schema(tmp_path,bad):
 cfg=assets(tmp_path);shutil.copy2(tmp_path/'evaluation/baseline_reference.npz',tmp_path/'evaluation/ground_truth.npz');cfg['truth_relative_l2_boundary']=bad;(tmp_path/'evaluation/metrics.json').write_text(json.dumps(cfg),encoding='utf8');report,code=run(tmp_path);assert code==2 and report['metrics']=={};json.dumps(report,allow_nan=False)
