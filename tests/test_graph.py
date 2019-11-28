from text_rank.graph import filter_pos


def test_filter_pos_adj():
    token = {'pos': 'ADJ'}
    assert filter_pos(token) is True


def test_filter_pos_cd():
    token = {'pos': 'CD'}
    assert filter_pos(token) is True


def test_filter_pos_noun():
    token = {'pos': 'NNP'}
    assert filter_pos(token) is True


def test_filter_pos_j():
    token = {'pos': 'JJ'}
    assert filter_pos(token) is True

def test_filter_pos_other():
    token = {'pos': 'G'}
    assert filter_pos(token) is False
