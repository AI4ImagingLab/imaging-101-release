from pathlib import Path
import json,subprocess,sys
import numpy as np
import pytest
from src.preprocessing import prepare_stack,correct_bleaching,bleaching_envelope,DT,PIXEL_SIZE,FIRST_FRAME

def test_background_subtraction_after_transient():
    frames=np.full((1,2501,2,2),18,dtype=np.uint16);frames[:,:1500]=0;frames[0,1500:,0,0]=20
    out=prepare_stack(frames);assert out.shape==(1001,2,2) and out.dtype==np.uint16
    assert np.all(out[:,0,0]==2) and np.all(out[:,1,1]==0)
    with pytest.raises(ValueError):prepare_stack(frames.astype(float))
    with pytest.raises(ValueError):prepare_stack(frames[:,:2000])

def test_bleaching_known_constant_envelope_and_invalid_input():
    j=np.arange(2000);envelope=bleaching_envelope(j,.4,100.,.5,1000.,.6)
    assert envelope[0]==1 and np.all(envelope>0)
    with pytest.raises(ValueError):correct_bleaching(np.ones(999))
    with pytest.raises(ValueError):correct_bleaching(np.zeros(1001))

def test_readme_constants_and_main_import_backend():
    root=Path(__file__).resolve().parents[2];text=(root/'README.md').read_text(encoding='utf8')
    for token in (str(DT),str(PIXEL_SIZE),str(FIRST_FRAME),'0.19','2.1'):
        assert token in text
    run=subprocess.run([sys.executable,'-c','import matplotlib; b=matplotlib.get_backend(); import src.main; assert matplotlib.get_backend()==b'],cwd=root,capture_output=True,text=True)
    assert run.returncode==0,run.stderr
