import numpy as np
import numba
from logger import log
from matrix_operations import TPB, BOOL_TYPE_VALUES, BYTE_TYPE_VALUES, INT_TYPE_VALUES, shared_memory


def to_type(matrices, type):
    for key, matrix in matrices.items():
        assert matrix.dtype == np.bool, "Passed matrices aren't bool"
        if matrix.dtype in BOOL_TYPE_VALUES:
            log('Matrices already bool, returning them..')
            return matrices
        elif type in BYTE_TYPE_VALUES:
            if shared_memory:
                size = 8 * TPB[0]
                matrix = np.pad(matrix, [(0, (size - matrix.shape[0]) % size), (0, (size - matrix.shape[1]) % size)], 'constant')
            matrix = np.packbits(matrix, axis=-1)
            matrices[key] = matrix
        elif type in INT_TYPE_VALUES:
            if shared_memory:
                size = 32 * TPB[0]
                matrix = np.pad(matrix, [(0, (size - matrix.shape[0]) % size), (0, (size-matrix.shape[1]) % size)], 'constant')
            matrix = np.pad(matrix, [(0, 0), (0, (32 - matrix.shape[1] % 32) % 32)], 'constant').astype(np.uint32)
            packed_matrix = sum(matrix[:, i::32] << (31 - i) for i in range(32))
            matrices[key] = packed_matrix
        else:
            raise ValueError('Matrix type {} is not supported'.format(type))
    return matrices


def from_type(matrices):
    for key, matrix in matrices.items():
        if matrix.dtype in BOOL_TYPE_VALUES:
            log('Matrices already bool, returning them..')
            return matrices
        else:
            raise ValueError('Matrix type {} is not supported'.format(matrix.dtype))
    log('Matrices are copied to GPU')
    return matrices


def to_gpu(matrices):
    for nonterminal, matrix in matrices.items():
        matrices[nonterminal] = numba.cuda.to_device(matrix)
    return matrices


def from_gpu(matrices):
    for nonterminal, matrix in matrices.items():
        matrices[nonterminal] = matrix.copy_to_host()
    log('Matrices are copied from GPU')
    return matrices