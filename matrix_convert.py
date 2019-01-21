import numpy as np
import numba
from logger import log

BOOL_TYPE_VALUES = ['bool', np.bool, numba.boolean]


def to_type(matrices, type):
    for key, matrix in matrices.items():
        assert matrix.dtype == np.bool, "Passed matrices aren't bool"
        if matrix.dtype in BOOL_TYPE_VALUES:
            log('Matrices already bool, returning them..')
            return matrices
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
    return matrices


def to_gpu(matrices):
    for nonterminal, matrix in matrices.items():
        matrices[nonterminal] = matrix.copy_to_host()
    return matrices


def from_gpu(matrices):
    for nonterminal, matrix in matrices.items():
        matrices[nonterminal] = numba.cuda.to_device(matrix)
    return matrices