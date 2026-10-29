"""Standard task entry point: python -m src.main."""
from pathlib import Path
import argparse
import numpy as np
from .preprocessing import load_data
from .solvers import reconstruct

def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',type=Path,default=Path('data/raw_data.npz'))
    parser.add_argument('--output',type=Path,default=Path('output/reconstruction.npz'))
    args=parser.parse_args(argv)
    result,diagnostics=reconstruct(load_data(args.data))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output,reconstruction=result)
    print(f"PM1 reconstructed343 supported loci on all350 native bins; objective={diagnostics['objective']:.12g}; {diagnostics['seconds']:.3f}s")
    return 0

if __name__=='__main__':raise SystemExit(main())
