from sparse_cnn_tensorflow.sparse_cnn import position_in_filter

# channels should be in fastest moving index to match tf.nn.conv2d weight ordering
# for a given filter position, overlapping an active site, we want to know the active-site's position in the
# filter array. I.e. we want know which weight element to multiply by the active_site.

def test_position_in_filter_first():
    ic0 = position_in_filter(0, 0, 0, 0, 2, 0, 3)
    ic1 = position_in_filter(0, 0, 0, 0, 2, 1, 3)
    ic2 = position_in_filter(0, 0, 0, 0, 2, 2, 3)

    assert (ic0, ic1, ic2) == (0, 1, 2)


def test_position_in_filter_last():
    ic0 = position_in_filter(0, 0, 1, 1, 2, 0, 3)
    ic1 = position_in_filter(0, 0, 1, 1, 2, 1, 3)
    ic2 = position_in_filter(0, 0, 1, 1, 2, 2, 3)

    assert (ic0, ic1, ic2) == (9, 10, 11)
