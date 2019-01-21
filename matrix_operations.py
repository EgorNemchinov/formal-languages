import math
import numpy as np
from numba import cuda
import numba

gpu = False
shared_memory = False
TPB = (16, 16)
sB_size8 = TPB[0] * 8
sB_size32 = TPB[1] * 32
BOOL_TYPE_VALUES = ['bool', np.bool, numba.boolean]


def update_matrix(matrices, head, body):
    if gpu:
        update_matrix_gpu(matrices, head, body)
    else:
        update_matrix_cpu(matrices, head, body)


def update_matrix_cpu(matrices, head, body):
    head_mat = matrices[head]
    body_first_mat, body_second_mat = matrices[body[0]], matrices[body[1]]
    if head_mat.dtype in BOOL_TYPE_VALUES:
        new_matrix = head_mat + body_first_mat.dot(body_second_mat)
        matrices[head] = new_matrix
        return np.any(new_matrix != head_mat)
    else:
        raise ValueError('Matrix type {} on CPU is not supported'.format(head_mat.dtype))


def update_matrix_gpu(matrices, head, body):
    head_mat = matrices[head]
    body_first_mat, body_second_mat = matrices[body[0]], matrices[body[1]]
    is_changed = cuda.device_array((1,), dtype=bool)

    blockspergrid = (int(math.ceil(body_first_mat.shape[i] / TPB[i])) for i in (0, 1))

    if str(head_mat.dtype) == 'bool':
        matmul[blockspergrid, TPB](body_first_mat, body_second_mat, head_mat, is_changed)
        return is_changed[0]
    elif str(head_mat.dtype) == 'uint8':
        if shared_memory:
            raise NotImplementedError('Shared memory on type {} on GPU is not supported'.format(head_mat.dtype))
        else:
            matmul_packed8[blockspergrid, TPB](body_first_mat, body_second_mat, head_mat, is_changed)
        return is_changed[0]
    elif str(head_mat.dtype) == 'uint32':
        if shared_memory:
            raise NotImplementedError('Shared memory on type {} on GPU is not supported'.format(head_mat.dtype))
        else:
            matmul_packed32[blockspergrid, TPB](body_first_mat, body_second_mat, head_mat, is_changed)
        return is_changed[0]
    else:
        raise ValueError('Type {} on GPU is not supported'.format(head_mat.dtype))


@cuda.jit
def matmul(A, B, C, is_changed):
    """Perform matrix multiplication of C = A * B
    """
    row, col = cuda.grid(2)
    if row < C.shape[0] and col < C.shape[1]:
        tmp = False
        for k in range(A.shape[1]):
            tmp = tmp or (A[row, k] and B[k, col])
        if tmp and not C[row, col]:
            is_changed[0] = True
            C[row, col] = tmp


@cuda.jit
def matmul_packed8(A, B, C, is_changed):
    row, col = cuda.grid(2)
    size = 8
    if row >= C.shape[0] or col >= C.shape[1]:
        return
    value = 0
    for k in range(A.shape[1]):
        cur_value_A = A[row, k]
        for j in range(size - 1, -1, -1):
            if cur_value_A & 1:
                value |= (B[k * size + j, col])
            cur_value_A >>= 1
    old_value = C[row, col]
    new_value = old_value | value
    if new_value != old_value:
        is_changed[0] = True
        C[row, col] = new_value


@cuda.jit
def matmul_packed32(A, B, C, is_changed):
    row, col = cuda.grid(2)
    size = 32
    if row >= C.shape[0] or col >= C.shape[1]:
        return
    value = 0
    for k in range(A.shape[1]):
        cur_value_A = A[row, k]
        for j in range(size - 1, -1, -1):
            if cur_value_A & 1:
                value |= (B[k * size + j, col])
            cur_value_A >>= 1
    old_value = C[row, col]
    new_value = old_value | value
    if new_value != old_value:
        C[row, col] = new_value
        is_changed[0] = True



