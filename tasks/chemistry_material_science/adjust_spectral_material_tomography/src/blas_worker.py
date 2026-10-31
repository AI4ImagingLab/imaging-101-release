"""Generic float64 Linux BLAS worker using Python standard-library binary IPC.

This process implements no acquisition, objective, projection or inverse method.
"""
from __future__ import annotations

import ctypes
import json
import math
import os
import struct
import sys


def main():
    source, destination = sys.stdin.buffer, sys.stdout.buffer
    library = ctypes.CDLL(os.environ.get('IMAGING_BLAS_LIBRARY', 'libblas.so.3'))
    pointer = ctypes.POINTER(ctypes.c_double)
    gemm = library.cblas_dgemm
    gemm.argtypes = [ctypes.c_int] * 6 + [ctypes.c_double, pointer, ctypes.c_int,
        pointer, ctypes.c_int, ctypes.c_double, pointer, ctypes.c_int]
    gemm.restype = None
    dot = library.cblas_ddot
    dot.argtypes = [ctypes.c_int, pointer, ctypes.c_int, pointer, ctypes.c_int]
    dot.restype = ctypes.c_double
    syrk = library.cblas_dsyrk
    syrk.argtypes = [ctypes.c_int] * 5 + [ctypes.c_double, pointer, ctypes.c_int,
        ctypes.c_double, pointer, ctypes.c_int]
    syrk.restype = None
    library.openblas_get_config.restype = ctypes.c_char_p
    library.openblas_set_num_threads(4)

    def read_exact(count):
        value = bytearray()
        while len(value) < count:
            part = source.read(count - len(value))
            if not part:
                raise EOFError('truncated BLAS request')
            value.extend(part)
        return value

    def values(count):
        buffer = read_exact(count * 8)
        return buffer, (ctypes.c_double * count).from_buffer(buffer)

    while True:
        header = source.read(32)
        if not header:
            break
        if len(header) < 32:
            header += read_exact(32 - len(header))
        operation, m, n, k = struct.unpack('<4Q', header)
        if operation == 0:
            break
        if max(m, n, k, m * k, k * n, m * n) >= 2**31:
            raise ValueError('BLAS request exceeds LP64 dimensions')
        if operation == 3:
            result = json.dumps({'config': library.openblas_get_config().decode('ascii'),
                'threads': library.openblas_get_num_threads(),
                'byteorder': sys.byteorder}).encode('ascii')
        elif operation == 1:
            ab, a = values(m * k)
            bb, b = values(k * n)
            c = (ctypes.c_double * (m * n))()
            gemm(102, 111, 111, m, n, k, 1., a, m, b, k, 0., c, m)
            result = bytes(c)
        elif operation == 2:
            ab, a = values(k)
            bb, b = values(k)
            result = struct.pack('<d', dot(k, a, 1, b, 1))
        elif operation == 4:
            ab, a = values(k)
            c = (ctypes.c_double * 1)()
            syrk(102, 121, 112, 1, k, 1., a, k, 0., c, 1)
            result = bytes(c)
        elif operation == 5:
            ab, a = values(2)
            result = struct.pack('<d', math.pow(a[0], a[1]))
        else:
            raise ValueError('unknown BLAS operation')
        destination.write(struct.pack('<Q', len(result)))
        destination.write(result)
        destination.flush()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
