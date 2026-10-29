"""Explicit optional Linux-BLAS execution context for source numeric fidelity."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

import numpy as np

_current = ContextVar('spectral_tomography_blas', default=None)
EXPECTED_CONFIG = 'OpenBLAS 0.3.26 NO_LAPACKE DYNAMIC_ARCH NO_AFFINITY Haswell MAX_THREADS=64'
LONG_CONTRACTION = 4096


class SourceBlas:
    """Run generic DGEMM/DDOT/DSYRK and scalar power with owned cleanup.

    Windows uses WSL Python3 and its system BLAS. IMAGING_WSL_DISTRO selects
    the distribution; optional IMAGING_BLAS_LIBRARY_PATH supplies its generic
    library directory. Native Linux uses the current Python interpreter.
    """

    def __init__(self, timeout=120.):
        self.timeout = float(timeout)
        self.process = None
        self.executor = None
        self.token = None
        self.stderr = None

    def __enter__(self):
        if _current.get() is not None:
            raise RuntimeError('nested SourceBlas contexts are unsupported')
        worker = Path(__file__).with_name('blas_worker.py').resolve()
        environment = os.environ.copy()
        extra = os.environ.get('IMAGING_BLAS_LIBRARY_PATH')
        if os.name == 'nt':
            command = ['wsl']
            distro = os.environ.get('IMAGING_WSL_DISTRO')
            if distro:
                command += ['-d', distro]
            converted = subprocess.run(command + ['--', 'wslpath', '-u', '-a', worker.as_posix()],
                capture_output=True, text=True, encoding='utf-8', errors='replace',
                check=True, timeout=30).stdout.strip()
            command += ['--', 'env']
            if extra:
                command += ['LD_LIBRARY_PATH=' + extra]
            if os.environ.get('IMAGING_BLAS_LIBRARY'):
                command += ['IMAGING_BLAS_LIBRARY=' + os.environ['IMAGING_BLAS_LIBRARY']]
            command += ['python3', converted]
        else:
            command = [sys.executable, str(worker)]
            if extra:
                environment['LD_LIBRARY_PATH'] = extra
        self.stderr = tempfile.TemporaryFile()
        self.executor = ThreadPoolExecutor(max_workers=1)
        try:
            self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=self.stderr, env=environment)
            self.info = json.loads(self._request(3, 0, 0, 0))
            if self.info != {'config': EXPECTED_CONFIG, 'threads': 4, 'byteorder': 'little'}:
                raise RuntimeError('unvalidated BLAS backend: ' + str(self.info))
            self.token = _current.set(self)
            return self
        except BaseException:
            self.close()
            raise

    def _read(self, count):
        value = bytearray()
        while len(value) < count:
            part = self.process.stdout.read(count - len(value))
            if not part:
                raise RuntimeError('BLAS worker terminated unexpectedly')
            value.extend(part)
        return value

    def _exchange(self, operation, m, n, k, arrays):
        if self.process.poll() is not None:
            raise RuntimeError('BLAS worker unavailable')
        self.process.stdin.write(struct.pack('<4Q', operation, m, n, k))
        for value in arrays:
            self.process.stdin.write(np.asarray(value, dtype='<f8').tobytes(order='F'))
        self.process.stdin.flush()
        count = struct.unpack('<Q', self._read(8))[0]
        expected = m * n * 8 if operation == 1 else 8 if operation in (2, 4, 5) else count
        if count != expected or count > 2**31:
            raise RuntimeError('invalid BLAS response length')
        return self._read(count)

    def _request(self, operation, m, n, k, *arrays):
        future = self.executor.submit(self._exchange, operation, m, n, k, arrays)
        try:
            return future.result(timeout=self.timeout)
        except BaseException as error:
            self.process.terminate()
            self.process.wait(timeout=10)
            self.stderr.seek(0)
            diagnostic = self.stderr.read().decode('utf-8', errors='replace')
            raise RuntimeError('generic BLAS IPC failed: ' + diagnostic) from error

    def matmul(self, left, right):
        if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[0]:
            raise ValueError('matmul requires compatible two-dimensional arrays')
        m, k = left.shape
        n = right.shape[1]
        if k < LONG_CONTRACTION:
            return np.matmul(left, right, out=np.empty((m, n), dtype=np.float64, order='F'))
        raw = self._request(1, m, n, k, left, right)
        return np.frombuffer(raw, dtype='<f8').reshape(m, n, order='F').copy(order='F')

    def dot(self, left, right):
        if left.shape != right.shape:
            raise ValueError('dot shapes must agree')
        if left is right:
            raw = self._request(4, 0, 0, left.size, left.ravel(order='F'))
        else:
            raw = self._request(2, 0, 0, left.size, left.ravel(order='F'), right.ravel(order='F'))
        return struct.unpack('<d', raw)[0]

    def power(self, base, exponent):
        """Evaluate the selected Linux libm scalar power, with no imaging logic."""
        operands = np.asarray([base, exponent], dtype=np.float64)
        raw = self._request(5, 0, 0, 2, operands)
        return struct.unpack('<d', raw)[0]

    def close(self):
        if self.token is not None:
            _current.reset(self.token)
            self.token = None
        if self.process is not None:
            if self.process.poll() is None:
                try:
                    self.process.stdin.write(struct.pack('<4Q', 0, 0, 0, 0))
                    self.process.stdin.flush()
                    self.process.wait(timeout=10)
                except (OSError, subprocess.TimeoutExpired):
                    self.process.terminate()
                    self.process.wait(timeout=10)
            try:
                self.process.stdin.close()
            except OSError:
                pass
            self.process.stdout.close()
        if self.executor is not None:
            self.executor.shutdown(wait=True, cancel_futures=True)
        if self.stderr is not None:
            self.stderr.close()

    def __exit__(self, *exception):
        self.close()


def current_backend():
    return _current.get()
