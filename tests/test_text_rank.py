import sys
import random
from unittest.mock import patch
import pytest
from text_rank.text_rank import sum_edges, accum, text_rank
from text_rank.graph import AdjacencyList


@pytest.fixture
def args():
    niter = random.randint(0, 100)
    dampening = random.random()
    quiet = random.choice([True, False])
    seed = None if random.choice([True, False]) else random.randint(0, sys.maxsize)
    return {'niter': niter, 'dampening': dampening, 'quiet': quiet, 'seed': seed}


@pytest.mark.skip(reason="Because we import the text_rank function into the __init__ we can't mock things inside the text_rank file")
def test_text_rank_dispatch_list(args):
    with patch('text_rank.text_rank.text_rank_list') as rank_patch:
        graph = MagicMock(spec=AdjacencyList)
        text_rank(graph, **args)
        rank_patch.assert_called_once_with(graph, **args)


@pytest.mark.skip(reason="Because we import the text_rank function into the __init__ we can't mock things inside the text_rank file")
def test_text_rank_dispatch_matrix(args):
    with patch('text_rank.text_rank.text_rank_matrix') as rank_patch:
        graph = MagicMock(spec=AdjacencyMatrix)
        text_rank(graph, **args)
        rank_patch.assert_called_once_with(graph, **args)


def test_sum_edges():
    def test():
        edges = {}
        total = random.randint(0, 100)
        i = 0
        while total > 0:
            weight = random.randint(1, 5)
            weight = min(weight, total)
            edges[i] = weight
            i += 1
        score = sum_edges(edges)
        assert score == total
