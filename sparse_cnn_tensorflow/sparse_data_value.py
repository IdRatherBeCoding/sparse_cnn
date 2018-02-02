import tensorflow as tf
import numpy as np
from typing import Tuple

class SparseDataValue(object):
    """
    Sparse representation of dense tensor value. Differs to tf.SparseTensorValue representation because only
    spatial indices are stored; it is assumed that all channels have the same set of active sites.
    """
    def __init__(self,
                 dense: np.ndarray = None,
                 H: np.ndarray = None,
                 M: np.ndarray = None,
                 dense_shape=None,
                 ground_state=None):
        """
        Instantiate SparseDataValue by providing either `dense`, the full numpy.ndarray,
        or the sparse components: H, M, dense_shape, ground_state.
        
        Args:
            dense (numpy.ndarray): The dense representation of the tensor.
            H (numpy.ndarray): Dimensions a_in x 2. Rows provide coordinates of active-sites in dense matrix.
            M (numpy.ndarray): Dimensions a_in x n_in. "Each row corresponds to the vector at one of the active spatial
                               locations".
            dense_shape (Tuple[int, int]): Dimensions of dense tensor: n_H, n_W, n_in
            ground_state (numpy.ndarray): 1-D array of length n_in. Specifies ground-state value to assign to non-active
                                          sites of each channel.
        """

        self.dense_shape = dense_shape if dense_shape else dense.shape

        if ground_state is None:
            self.ground_state = np.zeros(self.dense_shape[2], dtype=dense.dtype if dense is not None else M.dtype)
        else:
            if len(ground_state) != self.dense_shape[2]:
                raise ValueError("One ground-state value must be provided for each channel")
            self.ground_state = ground_state

        if dense is not None:
            self._from_dense(dense)
        else:
            self.H = H
            self.M = M

    # todo: this only works for dense input with zeros for non-active sites.
    def _from_dense(self, dense):
        # reduce to purely spatial representation: take max of absolute values along channels axis
        # dense should be np.array of shape (n_H, n_W, n_c)
        dense_spatial = np.amax(np.abs(dense), axis=2)

        indx_col, indx_row = np.nonzero(dense_spatial)
        H_in = np.array(list(zip(indx_col, indx_row)))

        dense_flat = dense.reshape((-1, dense.shape[-1]))
        ravelled_indices = np.ravel_multi_index((indx_col, indx_row), dense_spatial.shape)
        M_in = dense_flat[ravelled_indices, :]

        self.H = H_in
        self.M = M_in

    def to_dense(self):
        (n_H, n_W, n_C) = self.dense_shape
        dense_flat = np.empty((n_H * n_W, n_C), dtype=self.M.dtype)
        for i, gs in enumerate(self.ground_state):
            dense_flat[:, i].fill(gs)
        for row, [i, j] in enumerate(self.H):
            dense_flat[np.ravel_multi_index((i, j), (n_H, n_W))] = self.M[row]
        return dense_flat.reshape(self.dense_shape)
