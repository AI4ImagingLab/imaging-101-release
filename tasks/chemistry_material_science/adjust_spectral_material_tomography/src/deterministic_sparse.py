"""General float32/float64 CSR multiplication with explicit ordered scalar reductions."""
from __future__ import annotations

import numpy as np

_CODE = r'''
extern "C" __global__ void ordered_csr_product(
    const int* offsets, const int* columns, const SCALAR* coefficients,
    const SCALAR* right, SCALAR* result, int rows, int quantities) {
    long long position = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (position >= (long long)rows * quantities) return;
    int row = position / quantities;
    int quantity = position % quantities;
    SCALAR accumulator = 0.0;
    for (int entry = offsets[row]; entry < offsets[row + 1]; ++entry) {
        SCALAR product = MULTIPLY(coefficients[entry],
            right[(long long)columns[entry] * quantities + quantity]);
        accumulator = ADD(accumulator, product);
    }
    result[position] = accumulator;
}
'''


class OrderedCsr:
    """Own one CSR device copy and return source-compatible F-order host arrays.

    One thread computes one output element in CSR entry order. Explicit rounded
    multiply/add prevents atomic reductions and fused multiply-add reassociation.
    Importing this module allocates no device context or arrays.
    """

    def __init__(self, matrix):
        import cupy as cp
        csr = matrix.tocsr()
        if csr.dtype not in (np.dtype('float32'), np.dtype('float64')):
            raise TypeError('ordered CSR requires float32 or float64 coefficients')
        self.dtype = csr.dtype
        limit = np.iinfo(np.int32).max
        if max(*csr.shape, csr.nnz) > limit:
            raise ValueError('ordered CSR exceeds supported int32 matrix dimensions')
        self.shape = csr.shape
        self.cp = cp
        self.pool = cp.cuda.MemoryPool()
        scalar, multiply, add = ('float', '__fmul_rn', '__fadd_rn') if self.dtype == np.float32 else ('double', '__dmul_rn', '__dadd_rn')
        code = f'#define SCALAR {scalar}\n#define MULTIPLY {multiply}\n#define ADD {add}\n' + _CODE
        self.kernel = cp.RawKernel(code, 'ordered_csr_product', options=('--fmad=false',))
        self.offsets = self.columns = self.coefficients = None
        with cp.cuda.using_allocator(self.pool.malloc):
            self.offsets = cp.asarray(csr.indptr, dtype=cp.int32)
            self.columns = cp.asarray(csr.indices, dtype=cp.int32)
            self.coefficients = cp.asarray(csr.data, dtype=self.dtype)

    def multiply(self, values):
        values = np.asarray(values)
        if values.dtype != self.dtype:
            raise TypeError('ordered CSR dense dtype must match matrix dtype')
        if values.ndim != 2 or values.shape[0] != self.shape[1]:
            raise ValueError('ordered CSR requires a compatible two-dimensional operand')
        if self.offsets is None:
            raise RuntimeError('ordered CSR operator is closed')
        columns = values.shape[1]
        if columns > np.iinfo(np.int32).max:
            raise ValueError('ordered CSR dense operand has too many columns')
        if not columns or not self.shape[0]:
            return np.empty((self.shape[0], columns), dtype=self.dtype, order='F')
        cp = self.cp
        with cp.cuda.using_allocator(self.pool.malloc):
            right = cp.asarray(np.ascontiguousarray(values))
            result = cp.empty((self.shape[0], columns), dtype=self.dtype)
            self.kernel(((result.size + 255) // 256,), (256,),
                (self.offsets, self.columns, self.coefficients, right, result,
                 np.int32(self.shape[0]), np.int32(columns)))
            host = cp.asnumpy(result)
        return np.asfortranarray(host)

    def close(self):
        self.offsets = self.columns = self.coefficients = None
        self.pool.free_all_blocks()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
