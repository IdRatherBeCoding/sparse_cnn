import numpy as np
import tensorflow as tf
from sparse_cnn_tensorflow.sparse_data_tensor import SparseDataTensor

def filter_positions(row, col, height, width, f):
    # slide filter around this point to generate valid output positions
    for irow in range(max(row - f + 1, 0), min(row + 1, height - f + 1)):
        for icol in range(max(col - f + 1, 0), min(col + 1, width - f + 1)):
            yield (irow, icol)


# c_in will be in range(n_in)
def position_in_filter(filter_row, filter_col, site_row, site_col, f, c_in, n_in):
    return c_in + (site_col - filter_col) * n_in + (site_row - filter_row) * f * n_in


def build_h_out_and_Q(H_in, M_in, dense_shape, f, n_in, ground_state):
    """
    Build H_out and intermediate Q from https://arxiv.org/pdf/1505.02890.pdf.

    Iterate through H_in and create list of active sites in output.

    Parameters
    ----------
    H_in : numpy.ndarray
        Dimensions a_in x 2. Rows provide coordinates of active-sites in dense matrix. 
    M_in : numpy.ndarray
        Dimensions a_in x n_in. "Each row corresponds to the vector at one of the active spatial locations".
    dense_shape : tuple
        Dimensions of dense tensor: n_H, n_W, n_C (n_C is the same as n_out)
    f : int
        Size of filter (f x f).
    n_in : int
        Number of input channels.
    ground_state : numpy.ndarray

    Returns
    -------
    (np.ndarray, np.ndarray)
        Tuple elemnts:
        1. The a_out x 2 matrix of active-site positions in M_out.
        2. The a_out x (f^2 * n_in) Q matrix.
    """

    height = dense_shape[0]
    width = dense_shape[1]

    output_sites = {}
    # enumerate all output active sites and store the row numbers in H_out and Q
    i_out = 0
    for [row, col] in H_in:
        for i, j in filter_positions(row, col, height, width, f):
            if (i, j) not in output_sites:
                output_sites[(i, j)] = i_out
                i_out += 1

    a_out = i_out
    h_out = np.empty((a_out, 2), dtype=H_in.dtype)
    q = np.empty((a_out, f * f, n_in), dtype=M_in.dtype)
    for (i_gs, gs) in enumerate(ground_state):
        q[:, :, i_gs] = gs
    q = q.reshape((a_out, f * f * n_in))

    # enumerate all output active sites again, filling in values as we go.
    for idx, [row, col] in enumerate(H_in):
        values = M_in[idx]
        for i, j in filter_positions(row, col, height, width, f):
            i_out = output_sites[(i, j)]
            h_out[i_out, 0] = i
            h_out[i_out, 1] = j
            for i_val, value in enumerate(values):
                # todo: we should be able to reshape q and assign values to a slice.
                # q could be reshaped back to a_out x f*f*n_in at the end.
                q[i_out, position_in_filter(i, j, row, col, f, i_val, n_in)] = value

    return (h_out, q)


# W has shape (f*f*n_in, n_out)
# tf.nn.conv_2d has W shape (batch_size, f, f, n_in, n_out)
# so gs_n should have comparable ordering for product.
def next_ground_state(W, gs_in, f):
    gs_1 = tf.reshape(gs_in, (-1, 1))
    gs_n = tf.tile(gs_1, [f * f, 1])
    return tf.reshape(tf.matmul(W, gs_n, transpose_a=True), [-1])


def sparse_conv_2d(sparse_input, W, f, n_out, b):
    # b should just be 1-D tensor of weights, length n_out

    H_in = sparse_input.H
    M_in = sparse_input.M
    dense_shape = sparse_input.dense_shape
    n_in = dense_shape[2]
    ground_state = sparse_input.ground_state

    output_spatial_shape = (dense_shape[0] - f + 1, dense_shape[1] - f + 1)

    H_out, Q = tf.py_func(build_h_out_and_Q,
                          [H_in, M_in, dense_shape, f, n_in, ground_state],
                          [H_in.dtype, M_in.dtype])

    M_out = tf.add(tf.matmul(Q, W), b)

    output_dense_shape = (output_spatial_shape[0], output_spatial_shape[1], n_out)

    output_ground_state = next_ground_state(W, ground_state, f) + b

    return SparseDataTensor(H_out, M_out, output_dense_shape, output_ground_state)