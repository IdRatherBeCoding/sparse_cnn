import tensorflow as tf
from sparse_cnn_tensorflow.sparse_data_value import SparseDataValue

class SparseDataTensor(object):
    def __init__(self, H, M, dense_shape, ground_state):
        self.H = H
        self.M = M
        self.dense_shape = dense_shape
        self.ground_state = ground_state

    def to_value(self, sess: tf.Session):
        H, M, gs = sess.run([self.H, self.M, self.ground_state])
        return SparseDataValue(H=H, M=M, dense_shape=self.dense_shape, ground_state=gs)