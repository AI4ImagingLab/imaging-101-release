from pathlib import Path
import ctypes,numpy as np,pytest
from src.preprocessing import camera_parameters
from src.physics_model import project,rays,detect,estimate_otf,predict,refine,reconstruct_frame,library,stereo_diagnostics,residual
ROOT=Path(__file__).resolve().parents[2]
@pytest.mark.parametrize('camera',range(4))
def test_source_geometry(data,camera):
    v=np.load(ROOT/'test/fixtures/native_geometry.npz')['values'];v=v[v[:,0]==camera];cal=camera_parameters(data)[camera]
    np.testing.assert_allclose(project(v[:,2:5],cal),v[:,5:7],rtol=0,atol=1e-10)
    np.testing.assert_allclose(rays(v[:,5:7],cal),v[:,7:13],rtol=0,atol=1e-9)
@pytest.mark.parametrize('camera',range(4))
def test_all_source_detection_and_otf(data,stages,camera):
    for f in range(5):np.testing.assert_allclose(detect(data['observations'][0,f,camera]),stages[f'detected_centers_{camera*5+f}'],rtol=0,atol=1e-8)
    expected=np.r_[stages[f'otf_peak_{camera}'].ravel(),stages[f'otf_fitted_coefficients_{camera}'].ravel()]
    np.testing.assert_allclose(estimate_otf(data['observations'][0,:5,camera]),expected,rtol=1e-10,atol=1e-12)
def test_source_shake(data,stages):
    cal=camera_parameters(data);otf=np.array([np.r_[stages[f'otf_peak_{c}'].ravel(),stages[f'otf_fitted_coefficients_{c}'].ravel()]for c in range(4)]);images=np.array([stages[f'shake_input_image_{c}']for c in range(4)])
    p,s,f=refine(stages['shake_input_objects_0'],images,cal,otf)
    np.testing.assert_allclose(p,stages['shake_output_objects_0'],rtol=0,atol=1e-6)
    np.testing.assert_allclose(s,stages['shake_output_scores_flags_0'][:,0],rtol=1e-10,atol=1e-10);np.testing.assert_array_equal(f,stages['shake_output_scores_flags_0'][:,1])
@pytest.mark.parametrize('frame',range(4))
def test_full_initial_ipr(data,stages,frame):
    cal=camera_parameters(data);otf=np.array([np.r_[stages[f'otf_peak_{c}'].ravel(),stages[f'otf_fitted_coefficients_{c}'].ravel()]for c in range(4)])
    p=reconstruct_frame(data['observations'][0,frame],cal,otf);np.testing.assert_allclose(p,stages[f'initial_frame_ipr_{frame}'],rtol=0,atol=1e-6)
def test_both_candidate_competitions_and_second_shake(data,stages):
    cal=camera_parameters(data);otf=np.array([np.r_[stages[f'otf_peak_{c}'].ravel(),stages[f'otf_fitted_coefficients_{c}'].ravel()]for c in range(4)])
    for case in range(2):
        points=[stages[f'match_input_cam{c}_{case}']for c in range(4)];candidates,selected,positions=stereo_diagnostics(points,cal,.8 if case==0 else 1.2)
        np.testing.assert_array_equal(candidates[:,:4],stages[f'match_candidates_{case}'][:,:4]);np.testing.assert_array_equal(selected,stages[f'match_selected_{case}']);np.testing.assert_allclose(positions,stages[f'shake_input_objects_{case}'],rtol=0,atol=1e-6)
    first=stages['shake_output_objects_0'];first_flags=stages['shake_output_scores_flags_0'][:,1];images=np.array([stages[f'shake_input_image_{c}']for c in range(4)])
    np.testing.assert_array_equal(images,data['observations'][0,0])
    for c in range(4):np.testing.assert_allclose(detect(images[c]),stages[f'detected_centers_{20+c}'],rtol=0,atol=1e-8)
    images=residual(first[first_flags==0],images,cal,otf);p,s,f=refine(stages['shake_input_objects_1'],images,cal,otf)
    np.testing.assert_allclose(p,stages['shake_output_objects_1'],rtol=0,atol=1e-6);np.testing.assert_allclose(s,stages['shake_output_scores_flags_1'][:,0],rtol=1e-10,atol=1e-10);np.testing.assert_array_equal(f,stages['shake_output_scores_flags_1'][:,1])
def test_independent_otf_design_normal_equations(stages):
    for c in range(4):
        design=stages[f'otf_design_{c}'];observed=stages[f'otf_log_observation_{c}'];known=stages[f'otf_fitted_coefficients_{c}'];solved=np.linalg.lstsq(design,observed,rcond=None)[0];np.testing.assert_allclose(solved,known,rtol=1e-10,atol=1e-12)
def test_sixteen_original_lms_histories(stages):
    for i in range(16):np.testing.assert_allclose(predict(stages[f'predict_history_{i}']),stages[f'predict_next_{i}'].ravel(),rtol=0,atol=1e-10)
def test_original_displacement_grid(stages):
    a=stages['initial_frame_ipr_0'];b=stages['initial_frame_ipr_1'];out=np.empty((51**3,3));assert library().numeric_field(a,len(a),b,len(b),out)==0
    expected=np.load(ROOT/'test/fixtures/initial_displacement_csv.npz')['field'];np.testing.assert_allclose(out,expected,rtol=0,atol=5.1e-10)
