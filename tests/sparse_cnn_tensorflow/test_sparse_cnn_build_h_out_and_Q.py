from sparse_cnn_tensorflow.sparse_cnn import build_h_out_and_Q
from sparse_cnn_tensorflow.sparse_data_value import SparseDataValue
import numpy as np


def test_h_q_n_in_2_nonzero_ground_state():
    x_dense = np.array([
        [[1.7, 0.7], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        [[0.0, 0.0], [0.0, 4.1], [0.0, 0.0], [0.0, 0.0]],
        [[0.0, 0.0], [0.0, 0.0], [7.9, 0.9], [4.8, 0.8]],
        [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
        ], dtype=np.float32)

    np.random.seed(102)

    x_s = SparseDataValue(x_dense, ground_state=np.array([0.5, 1.78]).astype(np.float32))

    n_in = 2
    f = 2

    H_out, Q = build_h_out_and_Q(x_s.H, x_s.M, x_s.dense_shape, f, n_in, x_s.ground_state)

    expected_H = np.array([(0, 0), (0, 1), (1, 0), (1, 1), (1, 2), (2, 1), (2, 2)])
    expected_Q = np.array([
        (1.7, 0.7, 0.5, 1.78, 0.5, 1.78, 0.0, 4.1),
        (0.5, 1.78, 0.5, 1.78, 0.0, 4.1, 0.5, 1.78),
        (0.5, 1.78, 0.0, 4.1, 0.5, 1.78, 0.5, 1.78),
        (0.0, 4.1, 0.5, 1.78, 0.5, 1.78, 7.9, 0.9),
        (0.5, 1.78, 0.5, 1.78, 7.9, 0.9, 4.8, 0.8),
        (0.5, 1.78, 7.9, 0.9, 0.5, 1.78, 0.5, 1.78),
        (7.9, 0.9, 4.8, 0.8, 0.5, 1.78, 0.5, 1.78),
    ])

    np.testing.assert_array_equal(H_out, expected_H)
    np.testing.assert_array_almost_equal(Q, expected_Q, decimal=6)
