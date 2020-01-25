import math
import random
from copy import deepcopy
from itertools import combinations
from unittest.mock import MagicMock, PropertyMock
import pytest
import numpy as np
from text_rank.graph import Vertex, Graph, AdjacencyList, AdjacencyMatrix
from utils import rand_str


def random_graph(p=None, min_vert=1000, max_vert=2000):
    p = random.random() if p is None else p
    G = random.choice([AdjacencyList, AdjacencyMatrix])
    verts = [str(i) for i in range(random.randint(min_vert, max_vert))]
    graph = G(verts)
    for src, tgt in combinations(verts, 2):
        if src == tgt:
            continue
        if random.random() <= p:
            graph.add_edge(src, tgt, 1)
            graph.add_edge(tgt, src, 1)
    return graph


def random_graphs(p=None, min_vert=10, max_vert=100):
    p = random.random() if p is None else p
    verts = [str(i) for i in range(random.randint(min_vert, max_vert))]
    graph = AdjacencyMatrix(verts)
    graph2 = AdjacencyList(verts)
    for src, tgt in combinations(verts, 2):
        if src == tgt:
            continue
        if random.random() <= p:
            graph.add_edge(src, tgt, 1)
            graph2.add_edge(src, tgt, 1)
            graph.add_edge(tgt, src, 1)
            graph2.add_edge(tgt, src, 1)
    return graph, graph2


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


def test_vertex_equal_match():
    label = rand_str()
    edges_in = {random.randint(0, 100): random.random() for _ in range(random.randint(0, 10))}
    edges_out = {random.randint(0, 100): random.random() for _ in range(random.randint(0, 10))}
    v1 = Vertex(label)
    v1._edges_out = edges_out
    v1._edges_in = edges_in

    v2 = Vertex(label)
    v2._edges_out = edges_out
    v2._edges_in = edges_in

    assert v1 == v2


def test_vertex_equal_match_same_object():
    label = rand_str()
    edges_in = {random.randint(0, 100): random.random() for _ in range(random.randint(0, 10))}
    edges_out = {random.randint(0, 100): random.random() for _ in range(random.randint(0, 10))}
    v1 = Vertex(label)
    v1._edges_out = edges_out
    v1._edges_in = edges_in

    v2 = v1

    assert v1 == v2


def test_vertex_equal_mismatch_values():
    label = label2 = rand_str()
    while label2 == label:
        label2 = rand_str()
    edges_in = {random.randint(0, 100): random.random() for _ in range(random.randint(0, 10))}
    edges_out = {random.randint(0, 100): random.random() for _ in range(random.randint(0, 10))}
    v1 = Vertex(label)
    v1._edges_out = edges_out
    v1._edges_in = edges_in

    v2 = Vertex(label2)
    v2._edges_out = edges_out
    v2._edges_in = edges_in

    assert v1 != v2


def test_vertex_equal_mismatch_edges_in():
    label = rand_str()
    edges_in = edges_in2 = {random.randint(0, 100): random.random() for _ in range(random.randint(0, 10))}
    while edges_in2 == edges_in:
        edges_in2 = {random.randint(0, 100): random.random() for _ in range(random.randint(0, 100))}
    edges_out = {random.randint(0, 100): random.random() for _ in range(random.randint(0, 10))}
    v1 = Vertex(label)
    v1._edges_out = edges_out
    v1._edges_in = edges_in

    v2 = Vertex(label)
    v2._edges_out = edges_out
    v2._edges_in = edges_in2

    assert v1 != v2


def test_vertex_equal_mismatch_edges_out():
    label = rand_str()
    edges_in = {random.randint(0, 100): random.random() for _ in range(random.randint(0, 10))}
    edges_out = edges_out2 = {random.randint(0, 100): random.random() for _ in range(random.randint(0, 10))}
    while edges_out2 == edges_out:
        edges_out2 = {random.randint(0, 100): random.random() for _ in range(random.randint(0, 100))}
    v1 = Vertex(label)
    v1._edges_out = edges_out
    v1._edges_in = edges_in

    v2 = Vertex(label)
    v2._edges_out = edges_out2
    v2._edges_in = edges_in

    assert v1 != v2


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


def test_graph_forces_contiguous_idx():
    pop = random.randint(100, 200)
    vertex_idxs = [1, 2, 3]
    while all(p != c - 1 for p, c in zip(vertex_idxs, vertex_idxs[1:])):
        vertex_idxs = sorted(random.sample(range(pop), k=pop // random.randint(2, 10)))
    vertices = {rand_str(): vi for vi in vertex_idxs}
    with pytest.raises(ValueError):
        g = Graph(vertices)


def test_add_vertex_label_none():
    label2idx = {rand_str(): i for i in range(random.randint(1, 10))}
    graph = Graph(deepcopy(label2idx))
    idx = graph._add_vertex(None)
    assert idx == len(label2idx)
    assert graph.idx2label[idx] == str(idx)


def test_add_vertex_label_reuse():
    label2idx = {rand_str(): i for i in range(random.randint(1, 10))}
    graph = Graph(deepcopy(label2idx))
    with pytest.raises(ValueError):
        idx = graph._add_vertex(random.choice(list(label2idx.keys())))


def test_add_vertex_label():
    label2idx = {rand_str(): i for i in range(random.randint(1, 10))}
    graph = Graph(deepcopy(label2idx))
    gold_label, idx = rand_str(), len(label2idx)
    label2idx[gold_label] = idx
    idx = graph._add_vertex(gold_label)
    assert graph.label2idx == label2idx
    assert graph.idx2label == {i: l for l, i in label2idx.items()}


def test_add_vertex_list_out_of_order():
    graph = AdjacencyList({})
    gold_position = random.randint(10, 100)
    graph._add_vertex = MagicMock(return_value=gold_position)
    label = rand_str()
    with pytest.raises(ValueError):
        graph.add_vertex(label)
        graph._add_vertex.assert_called_once_with(label)


def test_add_vertex_list():
    verts = [rand_str() for _ in range(random.randint(1, 10))]
    graph = AdjacencyList(verts)
    gold_position = len(verts)
    graph._add_vertex = MagicMock(return_value=gold_position)
    gold_label = rand_str()
    idx = graph.add_vertex(gold_label)
    graph._add_vertex.assert_called_once_with(gold_label)
    assert idx == gold_position
    assert graph.vertices[gold_position] == Vertex(gold_label)


def test_add_node_matrix_out_of_order():
    graph = AdjacencyMatrix({})
    gold_position = random.randint(10, 100)
    graph._add_vertex = MagicMock(return_value=gold_position)
    label = rand_str()
    with pytest.raises(ValueError):
        graph.add_vertex(label)
        graph._add_vertex.assert_called_once_with(label)


def test_add_node_matrix():
    verts = [rand_str() for _ in range(random.randint(1, 10))]
    graph = AdjacencyMatrix(verts)
    graph._adjacency_matrix = np.random.rand(*graph._adjacency_matrix.shape)
    gold_position = len(verts)
    graph._add_vertex = MagicMock(return_value=gold_position)
    gold_label = rand_str()
    idx = graph.add_vertex(gold_label)
    graph._add_vertex.assert_called_once_with(gold_label)
    assert idx == gold_position
    assert (graph.adjacency_matrix[:, idx] == 0).all()
    assert (graph.adjacency_matrix[idx, :] == 0).all()


def test_add_edge_list():
    graph = AdjacencyList([rand_str() for _ in range(random.randint(10, 100))])
    source = random.randint(0, graph.vertex_count - 1)
    target = random.randint(0, graph.vertex_count - 1)
    weight = random.random()

    source_req = random.choice([source, graph[source]])
    target_req = random.choice([target, graph[target]])

    graph.add_edge(source_req, target_req, weight)

    assert graph.vertices[source].edges_out[target] == weight
    assert graph.vertices[target].edges_in[source] == weight


def test_add_edge_matrix():
    graph = AdjacencyMatrix([rand_str() for _ in range(random.randint(10, 100))])
    source = random.randint(0, graph.vertex_count - 1)
    target = random.randint(0, graph.vertex_count - 1)
    weight = random.random()

    source_req = random.choice([source, graph[source]])
    target_req = random.choice([target, graph[target]])

    graph.add_edge(source_req, target_req, weight)

    assert graph.adjacency_matrix[source, target] == weight


def test_density_half():
    graph = Graph({})
    edge_mock = PropertyMock(return_value=10)
    vertex_mock = PropertyMock(return_value=5)
    type(graph).edge_count = edge_mock
    type(graph).vertex_count = vertex_mock
    assert graph.density == 0.5


def test_density_bounds():
    verts = [rand_str() for _ in range(random.randint(2, 10))]
    G = random.choice([AdjacencyList, AdjacencyMatrix])
    graph = G(verts)
    assert graph.density == 0
    for src, tgt in combinations(verts, 2):
        if src == tgt:
            continue
        graph.add_edge(src, tgt, 1)
        graph.add_edge(tgt, src, 1)
    assert graph.density == 1
    graph = random_graph()
    assert 0 <= graph.density <= 1


def test_density():
    p = random.random()
    graph = random_graph(p)
    assert math.isclose(graph.density, p, rel_tol=1e-2)


def test_vertex_count():
    g1, g2 = random_graphs()
    assert g1.vertex_count == g2.vertex_count


def test_vertex_count_list():
    graph = AdjacencyList({})
    verts = random.randint(15, 30)
    for vert in range(verts):
        graph.add_vertex(str(vert))
    assert graph.vertex_count == verts


def test_vertex_count_matrix():
    graph = AdjacencyMatrix({})
    verts = random.randint(15, 30)
    for vert in range(verts):
        graph.add_vertex(str(vert))
    assert graph.vertex_count == verts


def test_edge_count():
    g1, g2 = random_graphs()
    assert g1.edge_count == g2.edge_count


def test_edge_count_list():
    graph = AdjacencyList({})
    gold = random.randint(10, 100)
    verts = random.randint(15, 30)
    vertices = [Vertex(str(i)) for i in range(verts)]
    for _ in range(gold):
        added = False
        while not added:
            src, tgt = np.random.randint(0, verts, size=(2,))
            if src == tgt:
                continue
            src_vert = vertices[src]
            tgt_vert = vertices[tgt]
            if src not in tgt_vert.edges_in and tgt not in src_vert.edges_out:
                src_vert.edges_out[tgt] = 1
                tgt_vert.edges_in[src] = 1
                added = True
    graph._vertices = vertices
    assert graph.edge_count == gold


def test_edge_count_matrix():
    graph = AdjacencyMatrix({})
    gold = random.randint(10, 100)
    verts = random.randint(15, 30)
    edges = np.zeros((verts, verts))
    for _ in range(gold):
        added = False
        while not added:
            src, tgt = np.random.randint(0, verts, size=(2,))
            if src == tgt:
                continue
            if edges[src, tgt] == 0:
                edges[src, tgt] = 1
                added = True
    graph._adjacency_matrix = edges
    assert graph.edge_count == gold
