import numpy as np
import tensorflow as tf
from sparse_cnn_tensorflow.sparse_data_value import SparseDataValue
from sparse_cnn_tensorflow.sparse_data_tensor import SparseDataTensor
from sparse_cnn_tensorflow.sparse_cnn import sparse_conv_2d


def run_sparse_and_dense(sparse: SparseDataValue, n_out, f, np_seed: int = None, b: np.ndarray = None):
    dense = sparse.to_dense()

    if np_seed:
        np.random.seed(np_seed)

    n_in = sparse.dense_shape[2]

    sparse_tensor = SparseDataTensor(
        tf.constant(sparse.H),
        tf.constant(sparse.M),
        sparse.dense_shape,
        tf.constant(sparse.ground_state))

    W_np = np.random.rand(f * f * n_in, n_out)
    W = tf.constant(W_np, dtype=tf.float32)
    if b is None:
        b = tf.constant(np.random.rand(n_out), dtype=tf.float32)

    output_sparse_tensor = sparse_conv_2d(sparse_tensor, W, f, n_out, b)

    # calculate output of sparse_conv_2d and convert to dense representation
    with tf.Session() as sess:
        scnn_output_dense = \
            SparseDataValue(
                H=sess.run(output_sparse_tensor.H), M=sess.run(output_sparse_tensor.M),
                dense_shape=output_sparse_tensor.dense_shape, ground_state=sess.run(output_sparse_tensor.ground_state)
            ).to_dense()

    # now perform the same convolution with tf.nn.conv2d

    X_4d = tf.reshape(tf.constant(dense, dtype=tf.float32),
                      (1, dense.shape[0], dense.shape[1], dense.shape[2]))
    W_4d = tf.reshape(W, (f, f, n_in, n_out))

    conv_1_tf = tf.nn.conv2d(X_4d, W_4d, strides=[1, 1, 1, 1], padding="VALID")
    dense_cnn_output_tf = tf.nn.bias_add(conv_1_tf, b)

    with tf.Session() as sess:
        dense_cnn_output = sess.run(dense_cnn_output_tf)

    return scnn_output_dense.reshape(dense_cnn_output.shape), dense_cnn_output


def test_sparse_conv_2d_n_in_2_nonzero_ground_state():
    x_dense = np.array([
        [[1.7, 0.7], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        [[0.0, 0.0], [0.0, 4.1], [0.0, 0.0], [0.0, 0.0]],
        [[0.0, 0.0], [0.0, 0.0], [7.9, 0.9], [4.8, 0.8]],
        [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
    ], dtype=np.float32)

    x_sparse = SparseDataValue(x_dense, ground_state=np.array([0.5, 1.78]).astype(np.float32))

    scnn_output, tf_cnn_output = run_sparse_and_dense(x_sparse, 13, 2, np_seed=1)

    np.testing.assert_almost_equal(scnn_output, tf_cnn_output, decimal=6)


def test_sparse_conv_2d_128x128x3():

    np.random.seed(123)
    x_dense = np.random.rand(128, 128, 3).astype(np.float32)
    # make a sparse region
    x_dense[range(51, 74), range(85, 108), :] = 0.0

    x_sparse = SparseDataValue(x_dense, ground_state=np.array([4.23, 2.5355, 13.322]).astype(np.float32))

    scnn_output, tf_cnn_output = run_sparse_and_dense(x_sparse, 8, 5, np_seed=2)

    np.testing.assert_almost_equal(scnn_output, tf_cnn_output, decimal=7)


def test_sparse_conv_2d_sparse_large_filter():

    x_dense = np.zeros((12, 12, 3)).astype(np.float32)
    # set a few values
    x_dense[4, 1] = [13.52, 3.1, 0.44]
    x_dense[9, 2] = [0.0, 158.97, 12.58]
    x_dense[3, 5] = [45.7, 93.1, 456.14]

    x_sparse = SparseDataValue(x_dense, ground_state=np.array([0.5, 12.1, 5.6]).astype(np.float32))

    scnn_output, tf_cnn_output = run_sparse_and_dense(x_sparse, 1, 7, np_seed=123)

    np.testing.assert_almost_equal(scnn_output, tf_cnn_output, decimal=4)


def test_sparse_conv_2d_very_large_very_sparse():

    x_dense = np.zeros((1280, 1280, 3)).astype(np.float32)
    # set a few values
    x_dense[4, 1] = [13.52, 3.1, 0.44]
    x_dense[9, 2] = [0.0, 158.97, 12.58]
    x_dense[3, 5] = [45.7, 93.1, 456.14]

    x_sparse = SparseDataValue(x_dense, ground_state=np.array([0.5, 0.5, 0.5]).astype(np.float32))

    scnn_output, tf_cnn_output = run_sparse_and_dense(x_sparse, 1, 7, np_seed=123)

    np.testing.assert_almost_equal(scnn_output, tf_cnn_output, decimal=4)
