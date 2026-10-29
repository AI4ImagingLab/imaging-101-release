"""Standard standalone reconstruction: python -m src.main."""
from pathlib import Path
import time
import numpy as np
from .preprocessing import preprocess
from .solvers import reconstruct

ROOT = Path(__file__).resolve().parents[1]


def run(data_path=None, output_path=None):
    """Reconstruct the bound observations; write exactly the two public arrays.

    Returns the diagnostic dictionary, including actual elapsed wall time.
    Explicit paths resolve relative to the caller; defaults are task-relative.
    """
    start = time.perf_counter()
    data_path = ROOT/'data/raw_data.npz' if data_path is None else Path(data_path)
    output_path = ROOT/'output/reconstruction.npz' if output_path is None else Path(output_path)
    with np.load(data_path, allow_pickle=False) as archive:
        raw = {key: archive[key][0] for key in archive.files}
    result = reconstruct(preprocess(raw))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, reconstruction=result['reconstruction'][None],
                        standard_error=result['standard_error'][None])
    result['runtime_seconds'] = time.perf_counter()-start
    return result


if __name__ == '__main__':
    result = run()
    print(f"Saved {result['reconstruction'].size} signed density samples and uncertainties; "
          f"selected smoothing={result['selected_smoothing']:g}; "
          f"runtime={result['runtime_seconds']:.3f} s")
