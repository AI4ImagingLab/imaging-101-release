from pathlib import Path
import subprocess,sys
import numpy as np
from src.preprocessing import load_data,window_waveforms,distance_weights
from src.physics_model import design_matrix
from src.solvers import solve
FIXTURE=Path(__file__).resolve().parents[1]/'fixtures/upstream.npz'

def test_independent_upstream_stage_and_pipeline_parity():
    f=np.load(FIXTURE,allow_pickle=False);d=load_data();r=solve(d)
    np.testing.assert_array_equal(window_waveforms(d['waveforms']).ravel(),f['output_d'])
    np.testing.assert_array_equal(np.repeat(distance_weights(d['station_distance']),450),f['output_weights'])
    for i,depth in enumerate([10.,12.,20.]):
        G=design_matrix(d['fundamental_greens'][i],d['station_azimuth'])
        np.testing.assert_allclose(G,f['output_G_'+str(depth)],rtol=1e-12,atol=1e-20)
        m=f[f'output_tensor{i}__m']*1e-7
        expected=np.array([[m[0],m[3],m[4]],[m[3],m[1],m[5]],[m[4],m[5],m[2]]])
        np.testing.assert_allclose(r['tensors'][i],expected,rtol=1e-10,atol=1e4)
        np.testing.assert_allclose(r['variance_reduction'][i],f[f'output_tensor{i}_total_VR'],atol=1e-10)

def test_cli_and_independent_judge(tmp_path):
    p=subprocess.run([sys.executable,'-m','src.main'],capture_output=True,text=True)
    assert p.returncode==0,p.stderr
    report=tmp_path/'nested/report.json'
    p=subprocess.run([sys.executable,'evaluation/judge.py','--prediction','output/reconstruction.npz','--report',str(report)],capture_output=True,text=True)
    assert p.returncode==0,p.stderr


def test_independent_deviatoric_comparison_parity():
    f=np.load(FIXTURE.parent/'upstream_deviatoric.npz',allow_pickle=False);d=load_data();r=solve(d,degree=5)
    for i,depth in enumerate([10.,12.,20.]):
        np.testing.assert_allclose(design_matrix(d['fundamental_greens'][i],d['station_azimuth'],5),f['output_G_'+str(depth)],rtol=1e-12,atol=1e-20)
        m=f[f'output_tensor{i}__m']*1e-7
        expected=np.array([[m[0],m[3],m[4]],[m[3],m[1],m[5]],[m[4],m[5],m[2]]])
        np.testing.assert_allclose(r['tensors'][i],expected,rtol=1e-10,atol=1e4)
        np.testing.assert_allclose(r['variance_reduction'][i],f[f'output_tensor{i}_total_VR'],atol=1e-10)
