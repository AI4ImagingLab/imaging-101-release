from pathlib import Path
import importlib.util,shutil,json,subprocess,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('independent_bini_judge',ROOT/'evaluation/judge.py');judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)


def test_independently_known_metric_cases():
    reference=np.array([0.,1.,2.]);estimate=np.array([0.,2.,4.]);ncc,nrmse=judge.metrics(estimate,reference)
    np.testing.assert_allclose(ncc,1);np.testing.assert_allclose(nrmse,np.sqrt(5/3)/2)
    assert judge.metrics(np.zeros(3),reference)[0]==0
    np.testing.assert_allclose(judge.metrics(-reference,reference)[0],-1,atol=1e-15)
    try:judge.metrics(reference,np.ones(3))
    except ValueError:pass
    else:assert False


def test_visual_metrics_on_same_independent_cases():
    from src.visualization import quality
    reference=np.array([0.,1.,2.]);estimate=np.array([0.,2.,4.])
    np.testing.assert_allclose(quality(estimate,reference),[1.,np.sqrt(5/3)/2])
    np.testing.assert_allclose(quality(np.zeros(3),reference),[0.,np.sqrt(5/3)/2])


def test_strict_contract_all_quantities(tmp_path):
    with np.load(ROOT/'evaluation/baseline_reference.npz',allow_pickle=False) as f:good={k:f[k].copy() for k in f.files}
    p=tmp_path/'submission.npz';np.savez(p,**good);assert judge.evaluate(p)[1]==0
    for key in good:
        bad={k:v.copy() for k,v in good.items()};bad[key]*=-1;np.savez(p,**bad)
        report,code=judge.evaluate(p);assert code==1 and report['metrics']
        for mode in ['missing','shape','dtype','complex','nan','inf']:
            bad={k:v.copy() for k,v in good.items()}
            if mode=='missing':del bad[key]
            elif mode=='shape':bad[key]=bad[key][0]
            elif mode=='dtype':bad[key]=bad[key].astype('float32')
            elif mode=='complex':bad[key]=bad[key].astype('complex128')
            else:bad[key].flat[0]=np.nan if mode=='nan' else np.inf
            np.savez(p,**bad);report,code=judge.evaluate(p);assert code==2 and not report['metrics'] and report['errors']
    assert judge.evaluate(tmp_path/'missing.npz')[1]==2
    p.write_bytes(b'broken');assert judge.evaluate(p)[1]==2
    assets=tmp_path/'assets';shutil.copytree(ROOT/'evaluation',assets)
    (assets/'baseline_reference.npz').unlink();assert judge.evaluate(p,assets)[1]==2
    np.savez(p,**good);np.savez(assets/'ground_truth.npz',**good);(assets/'baseline_reference.npz').write_bytes(b'bad')
    assert judge.evaluate(p,assets)[1]==0
    config=json.loads((assets/'metrics.json').read_text(encoding='utf8'));config['nrmse_boundary']=0
    (assets/'metrics.json').write_text(json.dumps(config),encoding='utf8');assert judge.evaluate(p,assets)[1]==2


def test_foreign_cwd_and_report_failure(tmp_path):
    command=[sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(ROOT/'evaluation/baseline_reference.npz'),'--report',str(tmp_path/'nested/report.json')]
    result=subprocess.run(command,cwd=tmp_path,capture_output=True,text=True);assert result.returncode==0
    block=tmp_path/'file';block.write_text('block',encoding='utf8');command[-1]=str(block/'report.json')
    result=subprocess.run(command,cwd=tmp_path,capture_output=True,text=True);assert result.returncode==2 and 'Cannot write report' in result.stderr
