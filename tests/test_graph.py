import random
from unittest.mock import MagicMock
from text_rank.graph import Vertex, Graph, AdjacencyList, AdjacencyMatrix
from utils import rand_str


def test_vertex_degree_in():
    v = Vertex(None)
    gold = random.randint(0, 100)
    for i in range(gold):
        v._edges_in[i] = None
    assert v.degree_in == gold


def test_vertex_degree_out():
    v = Vertex(None)
    gold = random.randint(0, 100)
    for i in range(gold):
        v._edges_out[i] = None
    assert v.degree_out == gold


def test_graph_get_node_etr():
    g = Graph({})
    g.label2idx = MagicMock()
    node = rand_str()
    g[node]
    g.label2idx.__getitem__.assert_called_once_with(node)


def test_graph_get_node_int():
    g = Graph({})
    g.idx2label = MagicMock()
    node = random.randint(0, 10)
    g[node]
    g.idx2label.__getitem__.assert_called_once_with(node)


def test_graph_contains_etr():
    g = Graph({})
    node = n2 = rand_str()
    while n2 == node:
        n2 = rand_str()
    g.label2idx = {node: None}
    assert node in g
    assert n2 not in g


def test_graph_contains_etr():
    g = Graph({})
    node = n2 = random.randint(0, 10)
    while n2 == node:
        n2 = random.randint(0, 10)
    g.idx2label = {node: None}
    assert node in g
    assert n2 not in g


def test_add_node_list():
    pass


def test_add_node_matrix():
    pass


def test_density():
    pass


def test_edge_count_list():
    pass


def test_edge_count_matrix():
    pass


def test_vertex_count_list():
    pass


def test_vertex_count_matrix():
    pass


def test_keyword_graph():
    pass


def test_sentence_graph():
    pass
