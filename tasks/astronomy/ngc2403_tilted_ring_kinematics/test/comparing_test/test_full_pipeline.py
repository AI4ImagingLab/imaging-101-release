from pathlib import Path
import json,os,subprocess,sys,time
import numpy as np
T=Path(__file__).resolve().parents[2]


def test_complete_standard_cli_against_independent_original(tmp_path):
    archive=Path(os.environ.get('IMAGING_TEST_LOGDIR',str(tmp_path)))
    archive.mkdir(parents=True,exist_ok=True)
    commands=[('main',[sys.executable,'-m','src.main']),
              ('judge',[sys.executable,'evaluation/judge.py','--prediction','output/reconstruction.npz','--report','output/judge_report.json'])]
    for name,command in commands:
        started=time.perf_counter()
        with (archive/(name+'_stdout.log')).open('wb') as stdout,(archive/(name+'_stderr.log')).open('wb') as stderr:
            process=subprocess.run(command,cwd=T,stdout=stdout,stderr=stderr,timeout=7200,shell=False)
        record=dict(command=command,cwd=str(T),returncode=process.returncode,runtime_seconds=time.perf_counter()-started)
        (archive/(name+'_process.json')).write_text(json.dumps(record,indent=2),encoding='utf-8')
        assert process.returncode==0,(record,(archive/(name+'_stderr.log')).read_text(encoding='utf-8'))
    report=json.loads((T/'output/judge_report.json').read_text(encoding='utf-8'))
    assert report['passed'] and report['status']=='pass' and report['observation_consistency_evaluated']
    assert set(report['metrics'])=={'rotation_range_nrmse','dispersion_range_nrmse','position_angle_wrapped_rms_deg','full_cube_rms_Jy_beam'}
    with np.load(T/'output/reconstruction.npz',allow_pickle=False) as result:
        assert set(result.files)=={'rotation','dispersion','position_angle'}
        for key in result.files:assert result[key].shape==(1,41) and np.isfinite(result[key]).all()
