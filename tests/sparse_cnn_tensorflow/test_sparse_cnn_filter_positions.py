from sparse_cnn_tensorflow.sparse_cnn import filter_positions


def test_filter_positions_top_left():
    result = filter_positions(0, 0, 5, 5, 2)
    expected = [(0, 0)]
    assert list(result) == expected


def test_filter_positions_top_right():
    result = filter_positions(0, 4, 5, 5, 2)
    expected = [(0, 3)]
    assert list(result) == expected


def test_filter_positions_bottom_left():
    result = filter_positions(4, 0, 5, 5, 2)
    expected = [(3, 0)]
    assert list(result) == expected


def test_filter_positions_bottom_right():
    result = filter_positions(4, 4, 5, 5, 2)
    expected = [(3, 3)]
    assert list(result) == expected


def test_filter_positions_first_row_second_col():
    result = filter_positions(0, 1, 5, 5, 2)
    expected = [(0, 0), (0, 1)]
    assert list(result) == expected


def test_filter_positions_second_row_first_col():
    result = filter_positions(1, 0, 5, 5, 2)
    expected = [(0, 0), (1, 0)]
    assert list(result) == expected


def test_filter_positions_last_row_second_last_col():
    result = filter_positions(4, 3, 5, 5, 2)
    expected = [(3, 2), (3, 3)]
    assert list(result) == expected


def test_filter_positions_second_last_row_last_col():
    result = filter_positions(3, 4, 5, 5, 2)
    expected = [(2, 3), (3, 3)]
    assert list(result) == expected


def test_filter_positions_middle():
    result = filter_positions(2, 2, 5, 5, 2)
    expected = [(1, 1), (1, 2), (2, 1), (2, 2)]
    assert list(result) == expected


def test_filter_positions_same_size_top_left():
    result = filter_positions(0, 0, 5, 5, 5)
    expected = [(0, 0)]
    assert list(result) == expected


def test_filter_positions_same_size_middle():
    result = filter_positions(2, 2, 5, 5, 5)
    expected = [(0, 0)]
    assert list(result) == expected


def test_filter_positions_same_size_bottom_right():
    result = filter_positions(4, 4, 5, 5, 5)
    expected = [(0, 0)]
    assert list(result) == expected


def test_filter_positions_f7_top_left():
    result = filter_positions(0, 0, 12, 12, 7)
    expected = [(0, 0)]
    assert list(result) == expected


def test_filter_positions_f7_near_middle():
    result = filter_positions(5, 5, 12, 12, 7)
    expected = []
    for i in range(6):
        for j in range(6):
            expected.append((i, j))
    assert list(result) == expected

