"""Read public observations without hidden assets or implicit conversions."""
from pathlib import Path
import numpy as np
def load_data(path="data/raw_data.npz"):
    """Return a dict of unbatched copies, retaining stored dtypes and units."""
    with np.load(Path(path),allow_pickle=False) as archive:
        result={key:archive[key][0].copy() for key in archive.files}
    if not all(np.all(np.isfinite(a)) for a in result.values()):
        raise ValueError("Non-finite public data")
    return result
