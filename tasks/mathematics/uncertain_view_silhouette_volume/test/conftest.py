from pathlib import Path
import sys,os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
torch.set_num_threads(1);torch.set_num_interop_threads(1)
