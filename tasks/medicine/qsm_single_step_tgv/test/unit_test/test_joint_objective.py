from pathlib import Path
import importlib.util
import numpy as np
sp=importlib.util.spec_from_file_location('j',Path(__file__).resolve().parents[2]/'evaluation/judge.py');j=importlib.util.module_from_spec(sp);sp.loader.exec_module(j)
def test_independent_constant_witness_and_eta_energy():
 shape=(5,5,5);mask=np.ones(shape,bool);chi=np.zeros(shape);eta=np.ones(shape);w=np.zeros((3,*shape));w[0]=2
 s=2*np.pi*42.58*2.89362*.025
 expected=.5*125/s**2+.00075*27*2
 assert abs(j.joint_tgv_objective(chi,eta,w,mask)-expected)<1e-14
