import time
import numpy as np
import math
from numba import cuda
from matrix_operations import matmul_packed32_shared, matmul_packed32, TPB
from matrix_convert import to_gpu, from_gpu, to_type, from_type

np.random.seed(1)


def run_uint32(size=10000, times=50, sparsity=0.7):
    A, B = np.random.randint(0, 100, (size,) * 2) > (sparsity * 100)
    C = np.zeros_like(A)

    matrices = {'A': A, 'B': B, 'C': C}
    matrices = to_type(matrices, 'uint32')
    matrices = to_gpu(matrices)

    is_changed = cuda.device_array((1,), dtype=bool)

    blockspergrid = tuple(int(math.ceil(A.shape[i] / TPB[i])) for i in (0, 1))

    print('Begin multuplying without shared memory')
    begin_no_shared = time.time()
    for i in range(times):
        matmul_packed32[blockspergrid, TPB](A, B, C, is_changed)
    end_no_shared = time.time()

    print('Begin multuplying with shared memory')
    begin_shared = time.time()
    for i in range(times):
        matmul_packed32_shared[blockspergrid, TPB](A, B, C, is_changed)
    end_shared = time.time()

    matrices = from_gpu(matrices)
    matrices = from_type(matrices)

    shared_average = (end_shared - begin_shared) / float(times)
    not_shared_average = (end_no_shared - begin_no_shared) / float(times)

    print('For shared memory average is {}'.format(shared_average))
    print('For not shared memory average is {}'.format(not_shared_average))


if __name__ == '__main__':
    print('Testing uint32 multiplication')

    for size in (100, 1000, 100000):
        print('-> Size of matrix {}'.format((size,) * 2))
        run_uint32(size, times=50, sparsity=0.7)