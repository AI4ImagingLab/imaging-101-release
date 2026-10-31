from pathlib import Path
import numpy as np
from src.preprocessing import prepare_data
from src.solvers import fit_card
FIX=Path(__file__).resolve().parents[1]/"fixtures/upstream_stages.npz"
TASK=Path(__file__).resolve().parents[2]
def test_original_preprocessing():
    a=np.load(FIX,allow_pickle=False);p=prepare_data(TASK/"data/raw_data.npz")
    np.testing.assert_allclose(p["X"],a["XinputIn"],rtol=2e-13,atol=2e-13)
    np.testing.assert_allclose(p["basis"],a["UIn"],rtol=3e-12,atol=3e-12)
    np.testing.assert_allclose(p["kernel"],a["WIn"],rtol=2e-13,atol=2e-13)
def test_original_updates_and_final():
    a=np.load(FIX,allow_pickle=False)
    for steps in [1,2,5,6,20,1000]:
        r=fit_card(a["XinputIn"],a["UIn"],a["WIn"],phi=.99,max_iter=steps)
        prefix="uniform_7" if steps==1000 else f"iter_{steps}"
        for key in ["V","b","lambda","sigma_e2","Obj"]:
            np.testing.assert_allclose(np.asarray(r[key]).ravel(),a[f"{prefix}_{key}"].ravel(),rtol=2e-8,atol=2e-8)


def test_qualified_normalized_selection():
    from src.solvers import reconstruct
    from src.physics_model import car_log_normalizer
    a=np.load(FIX,allow_pickle=False)
    out,results=reconstruct({"X":a["XinputIn"],"basis":a["UIn"],"kernel":a["WIn"]})
    scores=np.array([r["normalized_score"] for r in results])
    assert np.argmax(scores)==4
    np.testing.assert_allclose(out["mixing_weights"][0],a["uniform_5_V"],rtol=2e-8,atol=2e-8)
    np.testing.assert_allclose(scores,[178472.50985298184,178482.88976011093,178503.96612738873,178521.09015501669,178531.26893347173,178524.42919506005,178493.18282642617],rtol=4e-12,atol=1e-6)

def test_full_cli_pipeline(tmp_path):
    import subprocess,sys
    r=subprocess.run([sys.executable,"-m","src.main"],cwd=TASK,capture_output=True)
    assert r.returncode==0
    r=subprocess.run([sys.executable,str(TASK/"evaluation/judge.py"),"--prediction",str(TASK/"output/reconstruction.npz"),"--report",str(tmp_path/"report.json")],cwd=tmp_path,capture_output=True)
    assert r.returncode==0


def test_native_observation_conversion_and_reference_basis():
    from src.preprocessing import reference_basis
    a=np.load(FIX,allow_pickle=False)
    with np.load(TASK/"data/raw_data.npz",allow_pickle=False) as z:
        np.testing.assert_array_equal(z["spatial_counts"][0],a["spatial_count"])
        np.testing.assert_array_equal(z["single_cell_counts"][0],a["sc_count"])
        np.testing.assert_array_equal(z["spatial_coordinates"][0],a["spatial_location"])
        np.testing.assert_array_equal(z["cell_type_index"][0],a["sc_type"].ravel())
        np.testing.assert_array_equal(z["cell_sample_index"][0],a["sc_sample"].ravel())
        B,_=reference_basis(z["single_cell_counts"][0],z["cell_type_index"][0],z["cell_sample_index"][0])
        np.testing.assert_allclose(B,a["basis_full"],rtol=3e-12,atol=3e-12)
        C=z["spatial_counts"][0];np.testing.assert_array_equal(C[(C>0).sum(1)>5],a["counts_qc"])
        for key in ["spatial_gene_names","single_cell_gene_names"]:
            assert z[key].dtype.kind=="U" and np.all(z[key]!="")
