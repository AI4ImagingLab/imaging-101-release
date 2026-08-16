"""Cheap environment probe: report array backend availability.
Run from the run root with the run-local python.
"""
import numpy as np

def main():
    print('numpy', np.__version__)
    try:
        import cupy as cp
        print('cupy', cp.__version__)
        try:
            n = int(cp.cuda.runtime.getDeviceCount())
            print('cuda_devices', n)
        except Exception as e:
            print('cuda_devices_error', repr(e))
    except Exception:
        print('cupy unavailable -> will run on numpy (slower, same result)')

if __name__ == '__main__':
    main()
