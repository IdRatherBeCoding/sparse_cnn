from sparse_cnn_tensorflow.sparse_cnn import next_ground_state
import numpy as np
import tensorflow as tf

def test_n_in_1():
    W = tf.constant(np.array([
        [1.0, 4.5, 2.3, 6.4],
        [4.2, 2.6, 6.4, 4.3]
    ]).T)

    gs = tf.constant(np.array([0.55]))

    with tf.Session() as sess:
        result = sess.run(next_ground_state(W, gs, 2))

    expected = np.array([
        [(1.0 + 4.5 + 2.3 + 6.4) * 0.55],
        [(4.2 + 2.6 + 6.4 + 4.3) * 0.55]
    ]).reshape(-1)

    np.testing.assert_almost_equal(result, expected, decimal=6)


def test_n_in_2():
    W = tf.constant(np.array([
        [1.0, 4.5, 2.3, 6.4, 1.1, 2.2, 3.3, 4.4],
        [4.2, 2.6, 6.4, 4.3, 9.9, 8.8, 7.7, 6.6]
    ]).T)

    gs = tf.constant(np.array([0.55, 0.92]))

    with tf.Session() as sess:
        result = sess.run(next_ground_state(W, gs, 2))

    expected = np.array([
        [(1.0 + 2.3 + 1.1 + 3.3) * 0.55 + (4.5 + 6.4 + 2.2 + 4.4) * 0.92],
        [(4.2 + 6.4 + 9.9 + 7.7) * 0.55 + (2.6 + 4.3 + 8.8 + 6.6) * 0.92]
    ]).reshape(-1)

    np.testing.assert_almost_equal(result, expected, decimal=6)
