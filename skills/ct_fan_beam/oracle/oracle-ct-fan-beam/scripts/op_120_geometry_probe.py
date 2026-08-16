#!/usr/bin/env python
"""Smoke-check fan-beam geometry conventions against provided arrays.

Usage:
  python scripts/op_120_geometry_probe.py \
      public/.../case_000/input_data/raw_data.npz \
      public/.../case_000/params_data.json
"""
import sys
import json
import numpy as np


def main(npz_path, params_path):
    d = np.load(npz_path)
    p = json.load(open(params_path))
    N = p['image_size']; n_det = p['n_det']
    D_sd = p['source_to_isocenter_pixels']; D_dd = p['isocenter_to_detector_pixels']
    det_width = 2.0 * (D_sd + D_dd) * np.tan(np.arctan(N / 2 / D_sd))
    det_spacing = det_width / n_det
    det_pos = (np.arange(n_det) - (n_det - 1) / 2.0) * det_spacing
    af = np.linspace(0, 2 * np.pi, p['n_angles_full'], endpoint=False)
    sr = p['short_scan_range_deg'] * np.pi / 180
    asr = np.linspace(0, sr, p['n_angles_short'], endpoint=False)
    print('det_width', det_width, 'det_spacing', det_spacing,
          'eff_spacing', det_spacing * D_sd / (D_sd + D_dd))
    print('pad_len', int(2 ** np.ceil(np.log2(2 * (n_det - 1)))))
    print('det_pos match:', np.allclose(det_pos, d['det_pos'], atol=1e-3))
    print('angles_full match:', np.allclose(af, d['angles_full'], atol=1e-4))
    print('angles_short match:', np.allclose(asr, d['angles_short'], atol=1e-4))


if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) != 2:
        print('provide raw_data.npz and params_data.json paths')
        sys.exit(1)
    main(args[0], args[1])
