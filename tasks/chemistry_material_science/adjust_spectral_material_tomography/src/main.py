"""Complete real-data spectral self-calibration and joint material inference."""
from pathlib import Path
import argparse
import time
import numpy as np
from .preprocessing import load_observations, correct_counts, calibrated_dictionary
from .physics_model import geometry_vectors, build_system_matrix, sirt
from .solvers import adjust
from .linear_algebra import SourceBlas


def run_pipeline(data_path='data/raw_data.npz', output_path='output/reconstruction.npz',
                 backend='cuda_ordered', verbose=True, review_assets_path=None):
    """Execute all128-channel100-step SIRT and65-channel1000-step ADJUST."""
    start = time.perf_counter()
    observation = load_observations(data_path)
    data = correct_counts(observation['counts'], observation['dark'], observation['flat'])
    matrix = build_system_matrix(geometry_vectors(observation['angles']))
    t0 = time.perf_counter()
    calibration_backend = 'cuda_ordered' if backend == 'cuda_ordered' else 'cpu'
    channels = sirt(matrix, data, iterations=100, backend=calibration_backend, verbose=verbose)
    sirt_seconds = time.perf_counter() - t0
    dictionary, quartz = calibrated_dictionary(observation['dictionary'], channels)
    with SourceBlas():
        result = adjust(data[:, 29:94], matrix, dictionary, backend=backend, verbose=verbose)
    maps = np.stack([result['maps'][:, i].reshape(255, 255, order='F') for i in range(4)])
    submission = {'reconstruction': maps[None], 'spectra': result['spectra'][None],
                  'dictionary_weights': result['weights'][None]}
    if output_path is not None:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target, **submission)
    returned = {'submission': submission, 'result': result, 'matrix': matrix,
            'observations': observation, 'corrected_data': data,
            'spectral_channels': channels, 'quartz': quartz, 'dictionary': dictionary,
            'sirt_seconds': sirt_seconds, 'runtime_seconds': time.perf_counter() - start}
    if review_assets_path is not None:
        target = Path(review_assets_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target, spectral_channels=channels, quartz=quartz,
            dictionary=dictionary, history=result['history'],
            feasibility_history=result['feasibility_history'],
            solver_runtime_seconds=result['runtime_seconds'], sirt_seconds=sirt_seconds,
            runtime_seconds=returned['runtime_seconds'])
    return returned


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='data/raw_data.npz')
    parser.add_argument('--output', default='output/reconstruction.npz')
    parser.add_argument('--backend', choices=('cpu', 'cpu_sparse', 'cuda', 'cuda_sparse', 'cuda_ordered'),
                        default='cuda_ordered')
    parser.add_argument('--review-assets', default=None,
                        help='Optional freshly computed maintainer notebook arrays')
    args = parser.parse_args(argv)
    result = run_pipeline(args.data, args.output, args.backend,
                          review_assets_path=args.review_assets)
    print(f"Saved {args.output}; full runtime={result['runtime_seconds']:.3f}s; "
          f"iterations={len(result['result']['history'])}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
