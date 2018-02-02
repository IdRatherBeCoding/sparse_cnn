from sparse_cnn_tensorflow.sparse_data_value import SparseDataValue
import numpy as np


def test_simple_dense_int_array():
    dense = np.array(
        [
            [[1, 2], [0, 0], [0, 0]],
            [[0, 0], [5, 6], [6, 7]],
            [[0, 0], [0, 0], [0, 0]],
            [[0, 0], [0, 0], [16, 17]]
        ]
    )
    sparse = SparseDataValue(dense)
    assert sparse.dense_shape == (4, 3, 2)
    np.testing.assert_array_equal(sparse.ground_state, np.zeros(2, dtype=np.int64))
    np.testing.assert_array_equal(sparse.H, np.array([[0, 0], [1, 1], [1, 2], [3, 2]]))
    np.testing.assert_array_equal(sparse.M, np.array([[1, 2], [5, 6], [6, 7], [16, 17]]))


def test_simple_dense_float32_array():
    dense = np.array(
        [
            [[1.432, 2.654], [0, 0], [0, 0]],
            [[0, 0], [5.327, 6.777], [6.112, 7.123]],
            [[0, 0], [0, 0], [0, 0]],
            [[0, 0], [0, 0], [16.853, 17.352]]

        ]
    ).astype(np.float32)
    sparse = SparseDataValue(dense)
    assert sparse.dense_shape == (4, 3, 2)
    np.testing.assert_array_equal(sparse.ground_state, np.zeros(2, dtype=np.int64))
    np.testing.assert_array_equal(sparse.H, np.array([[0, 0], [1, 1], [1, 2], [3, 2]]))
    np.testing.assert_array_almost_equal(
        sparse.M, np.array([[1.432, 2.654], [5.327, 6.777], [6.112, 7.123], [16.853, 17.352]])
    )


def test_simple_sparse_to_dense():
    dense = np.array(
        [
            [[1.432, 2.654], [0, 0], [0, 0]],
            [[0, 0], [5.327, 6.777], [6.112, 7.123]],
            [[0, 0], [0, 0], [0, 0]],
            [[0, 0], [0, 0], [16.853, 17.352]]

        ]
    ).astype(np.float32)
    sparse = SparseDataValue(dense)
    np.testing.assert_almost_equal(dense, sparse.to_dense())


def test_large_dense_to_dense():
    np.random.seed(1)
    dense = np.random.rand(128, 128, 3).astype(np.float32)
    sparse = SparseDataValue(dense)
    np.testing.assert_almost_equal(dense, sparse.to_dense())

# not supporting dense -> sparse for nonzero ground states
# def test_simple_sparse_ground_state():
#     dense = np.array(
#         [
#             [[1.432, 2.654], [0.5, 7.1], [0.5, 7.1]],
#             [[0.5, 7.1], [5.327, 6.777], [6.112, 7.123]],
#             [[0.5, 7.1], [0.5, 7.1], [0.5, 7.1]],
#             [[0.5, 7.1], [0.5, 7.1], [16.853, 17.352]]
#
#         ]
#     ).astype(np.float32)
#     sparse = SparseDataValue(dense, ground_state=np.array([0.5, 7.1]))
#     np.testing.assert_almost_equal(dense, sparse.to_dense())
#     np.testing.assert_array_equal(sparse.H, np.array([[0, 0], [1, 1], [1, 2], [3, 2]]))

def test_sparse_components():
    expected_dense = np.array(
        [
            [[1.432, 2.654], [0, 0], [0, 0]],
            [[0, 0], [5.327, 6.777], [6.112, 7.123]],
            [[0, 0], [0, 0], [0, 0]],
            [[0, 0], [0, 0], [16.853, 17.352]]

        ]
    ).astype(np.float32)

    H_in = np.array([[0, 0], [1, 1], [1, 2], [3, 2]])
    M_in = np.array([[1.432, 2.654], [5.327, 6.777], [6.112, 7.123], [16.853, 17.352]])
    dense_shape = (4, 3, 2)

    sparse = SparseDataValue(H=H_in, M=M_in, dense_shape=dense_shape)

    np.testing.assert_equal(sparse.H, H_in)
    np.testing.assert_almost_equal(sparse.M, M_in)
    np.testing.assert_equal(sparse.ground_state, np.zeros(2))
    assert sparse.dense_shape == dense_shape

    np.testing.assert_almost_equal(expected_dense, sparse.to_dense(), decimal=5)

def test_sparse_components_nonzero_ground_state():
    expected_dense = np.array(
        [
            [[1.432, 2.654], [0.5, 7.1], [0.5, 7.1]],
            [[0.5, 7.1], [5.327, 6.777], [6.112, 7.123]],
            [[0.5, 7.1], [0.5, 7.1], [0.5, 7.1]],
            [[0.5, 7.1], [0.5, 7.1], [16.853, 17.352]]

        ]
    ).astype(np.float32)

    H_in = np.array([[0, 0], [1, 1], [1, 2], [3, 2]])
    M_in = np.array([[1.432, 2.654], [5.327, 6.777], [6.112, 7.123], [16.853, 17.352]])
    dense_shape = (4, 3, 2)
    ground_state = np.array([0.5, 7.1])

    sparse = SparseDataValue(H=H_in, M=M_in, dense_shape=dense_shape, ground_state=ground_state)

    np.testing.assert_equal(sparse.H, H_in)
    np.testing.assert_almost_equal(sparse.M, M_in)
    np.testing.assert_equal(sparse.ground_state, ground_state)
    assert sparse.dense_shape == dense_shape

    np.testing.assert_almost_equal(expected_dense, sparse.to_dense(), decimal=5)

test_sparse_components()