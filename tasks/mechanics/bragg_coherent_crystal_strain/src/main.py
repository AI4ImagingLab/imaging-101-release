"""Run the complete two-volume, 200-start-per-volume research reconstruction."""
from pathlib import Path
import argparse
import json
import time
import numpy as np
from .preprocessing import load_observations
from .solvers import reconstruct


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/raw_data.npz")
    parser.add_argument("--output", default="output/reconstruction.npz")
    parser.add_argument("--device", default="cupy")
    parser.add_argument("--measurements", default=None,
                        help="Optional maintainer NPZ with fresh per-start measurements")
    args = parser.parse_args(argv)
    counts, _ = load_observations(args.data)
    started = time.perf_counter()

    def progress(position, start, error, seconds):
        print(f"position={position + 1}/2 start={start + 1}/200 "
              f"squared_amplitude_error={error:.8g} seconds={seconds:.2f}", flush=True)

    result, coherence, histories = reconstruct(counts, device=args.device, progress=progress)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, reconstruction=result.astype(np.complex128),
                        coherence=coherence.astype(np.float64))
    if args.measurements is not None:
        measurements = Path(args.measurements)
        measurements.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(measurements,
                            errors=np.asarray([h["errors"] for h in histories], dtype=np.float64),
                            selected=np.asarray([h["selected"] for h in histories], dtype=np.int64),
                            peak_shifts=np.asarray([h["peak_shift"] for h in histories], dtype=np.int64),
                            intensity_thresholds=np.asarray([h["intensity_threshold"] for h in histories], dtype=np.float64),
                            padding=np.asarray([h["padding"] for h in histories], dtype=np.int64),
                            seconds=np.asarray(time.perf_counter()-started, dtype=np.float64))
    print(json.dumps({"seconds": time.perf_counter() - started, "output": str(target)}))


if __name__ == "__main__":
    main()
