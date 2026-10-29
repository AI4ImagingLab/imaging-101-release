"""Canonical named boundaries are active evaluator inputs, not unused aliases."""
from pathlib import Path
import json,shutil,importlib.util
import numpy as np,pytest
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location("canonical_boundary_judge",ROOT/"evaluation/judge.py")
judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)
def test_canonical_configuration_has_one_boundary_source():
    config=json.loads((ROOT/"evaluation/metrics.json").read_text(encoding="utf-8"))
    assert "thresholds" not in config
    assert set(k[:-9] for k in config if k.endswith("_boundary"))==set(config["metric_comparisons"])
@pytest.mark.parametrize("change",["boundary_changed","boundary_missing","comparison_missing","unexpected_metric"])
def test_named_boundary_consumption_and_schema_errors(tmp_path,monkeypatch,change):
    case=tmp_path/"case";assets=case/"evaluation";assets.mkdir(parents=True);(case/"data").mkdir()
    for f in (ROOT/"evaluation").glob("*.npz"):shutil.copyfile(f,assets/f.name)
    shutil.copyfile(ROOT/"data/raw_data.npz",case/"data/raw_data.npz")
    config=json.loads((ROOT/"evaluation/metrics.json").read_text(encoding="utf-8"))
    metric="observation_chi2_per_sample"
    if change=="boundary_changed":config[metric+"_boundary"]=.5
    if change=="boundary_missing":del config[metric+"_boundary"]
    if change=="comparison_missing":del config["metric_comparisons"][metric]
    if change=="unexpected_metric":
        config["metric_comparisons"]["undefined_metric"]="<=";config["undefined_metric_boundary"]=1.
    (assets/"metrics.json").write_text(json.dumps(config),encoding="utf-8")
    monkeypatch.setattr(judge,"ASSETS",assets)
    report,code=judge.evaluate(ROOT/"output/reconstruction.npz")
    if change=="boundary_changed":
        assert code==1 and report["status"]=="fail" and report["metrics"]
        assert report["thresholds"][metric]["boundary"]==.5
    else:assert code==2 and report["status"]=="error" and report["metrics"]=={}
