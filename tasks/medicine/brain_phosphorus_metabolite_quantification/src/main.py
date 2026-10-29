"""Standard entry point: python -m src.main."""
from pathlib import Path
import time
import numpy as np
from .solvers import reconstruct

ROOT=Path(__file__).resolve().parents[1]
OUTPUT_KEYS=('reconstruction','frequency_hz','linewidth_hz','phase_rad')


def run(data_path=None,output_path=None):
    """Fit the full bound FID, save all physical quantities and return diagnostics."""
    start=time.perf_counter()
    data_path=ROOT/'data/raw_data.npz' if data_path is None else Path(data_path)
    output_path=ROOT/'output/reconstruction.npz' if output_path is None else Path(output_path)
    with np.load(data_path,allow_pickle=False) as archive:
        fid=archive['measurements'][0]
    result=reconstruct(fid)
    output_path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output_path,**{k:result[k][None] for k in OUTPUT_KEYS})
    result['runtime_seconds']=time.perf_counter()-start
    return result


if __name__=='__main__':
    result=run()
    print(f"Fit all 16 resonances; objective={result['objective']:.12g}; "
          f"runtime={result['runtime_seconds']:.3f}s; all four arrays saved")
