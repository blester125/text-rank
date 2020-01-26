import sys
import math
import random
from copy import deepcopy
from itertools import combinations
from unittest.mock import patch, MagicMock
import pytest
import numpy as np
from text_rank.graph import AdjacencyList, AdjacencyMatrix, Vertex
from text_rank.text_rank import (
    sum_edges,
    accumulate_score,
    text_rank,
    text_rank_init_list,
    text_rank_init_matrix,
    text_rank_update_list,
    text_rank_update_matrix,
    text_rank_output_list,
    text_rank_output_matrix,
)
from utils import rand_str


TRIALS = 100


def test_sum_edges():
    def test():
        edges = {}
        gold_total = random.randint(0, 100)
        total = gold_total
        i = 0
        while total > 0:
            weight = random.randint(1, 5)
            weight = min(weight, total)
            edges[i] = weight
            i += 1
            total -= weight
        score = sum_edges(edges)
        assert score == gold_total

    for _ in range(TRIALS):
        test()


def test_accumulate_score():
    vertex = MagicMock(edges_in={0: 0.5, 1: 0.6, 4: 0.1})
    denom = [0.1, 0.7, 0.4, 0.0, 0.3]
    ws = [0.9, 0.2, 0.5, 0.1, 0.4]
    gold = 4.8047619047619055
    assert math.isclose(accumulate_score(vertex, ws, denom), gold)


def test_init_list_shapes():
    graph = MagicMock(vertices=[MagicMock() for _ in range(random.randint(10, 100))])
    ws, denom = text_rank_init_list(graph)
    assert len(ws) == len(denom) == len(graph.vertices)


def test_init_list_ws_inits():
    seed = 1337
    gold_ws = [
        0.6177528569514706,
        0.5332655736050008,
        0.36584835924937553,
        0.5857873539022715,
        0.16568728368878083,
        0.8243737469076314,
        0.38370480861420864,
        0.7896128249156874,
        0.9217265098962596,
        0.30763338797950246,
    ]
    graph = MagicMock(vertices=[MagicMock() for _ in range(len(gold_ws))])
    ws, _ = text_rank_init_list(graph, seed=seed)
    for w, gw in zip(ws, gold_ws):
        assert math.isclose(w, gw)


def test_init_list_d_norms():
    graph = MagicMock(vertices=[MagicMock() for _ in range(random.randint(10, 100))])
    gold = [random.random() for _ in range(len(graph.vertices))]
    values = deepcopy(gold)
    zero_indices = random.sample(range(len(gold)), len(gold) // 2)
    for i in zero_indices:
        gold[i] = 1
        values[i] = 0
    with patch("text_rank.text_rank_module.sum_edges") as sum_patch:
        sum_patch.side_effect = values
        _, denom = text_rank_init_list(graph)
        for d, g in zip(denom, gold):
            assert math.isclose(d, g)


def test_init_matrix_shapes():
    verts = random.randint(10, 100)
    graph = MagicMock(vertex_count=verts, adjacency_matrix=np.random.rand(verts, verts))
    ws, denom = text_rank_init_matrix(graph)
    assert len(ws) == len(denom) == verts


def test_init_matrix_ws_inits():
    seed = 1337
    gold_ws = np.array(
        [
            0.6177528569514706,
            0.5332655736050008,
            0.36584835924937553,
            0.5857873539022715,
            0.16568728368878083,
            0.8243737469076314,
            0.38370480861420864,
            0.7896128249156874,
            0.9217265098962596,
            0.30763338797950246,
        ]
    )
    graph = MagicMock(vertex_count=len(gold_ws), adjacency_matrix=np.random.rand(len(gold_ws), len(gold_ws)))
    ws, _ = text_rank_init_matrix(graph, seed=seed)
    np.testing.assert_allclose(ws, gold_ws)


def test_init_matrix_d_norms():
    graph = MagicMock(vertices=[MagicMock() for _ in range(random.randint(10, 100))])
    gold = [random.random() for _ in range(len(graph.vertices))]
    values = deepcopy(gold)
    zero_indices = random.sample(range(len(gold)), len(gold) // 2)
    for i in zero_indices:
        gold[i] = 1
        values[i] = 0
    with patch("text_rank.text_rank_module.sum_edges") as sum_patch:
        sum_patch.side_effect = values
        _, denom = text_rank_init_matrix(graph)
        for d, g in zip(denom, gold):
            assert math.isclose(d, g)


def test_update_list():
    graph = MagicMock(vertices=[MagicMock() for _ in range(random.randint(10, 100))])
    dampening = random.random()
    golds = [random.random() for _ in graph.vertices]
    update_values = [(g - 1 + dampening) / dampening for g in golds]
    with patch("text_rank.text_rank_module.accumulate_score") as acc_patch:
        acc_patch.side_effect = update_values
        ws = text_rank_update_list(graph, None, None, dampening)
        for w, g in zip(ws, golds):
            math.isclose(w, g)


def test_update_matrix():
    verts = np.random.randint(10, 100)
    ws = np.random.rand(verts)
    adj = np.random.rand(verts, verts)
    denom = np.random.rand(verts, 1)
    graph = MagicMock(adjacency_matrix=adj)
    dampening = np.random.rand()
    gold = (1 - dampening) + dampening * np.sum(ws.reshape((-1, 1)) * (adj / denom), axis=0)

    res = text_rank_update_matrix(graph, ws, denom, dampening)

    np.testing.assert_allclose(res, gold)


def test_output_list():
    v = random.randint(2, 10)
    verts = [rand_str() for _ in range(v)]
    graph = MagicMock(vertices=[Vertex(v) for v in verts])
    ws = np.random.rand(v)
    gold_labels = [verts[i] for i in np.argsort(ws)[::-1]]
    gold_score = np.sort(ws)[::-1]
    res = text_rank_output_list(graph, ws)
    for gl, gs, (l, s) in zip(gold_labels, gold_score, res):
        assert l == gl
        assert math.isclose(s, gs)


def test_output_matrix():
    v = random.randint(2, 10)
    verts = [rand_str() for _ in range(v)]
    graph = MagicMock(label2idx={v: None for v in verts})
    ws = np.random.rand(v)
    gold_labels = [verts[i] for i in np.argsort(ws)[::-1]]
    gold_score = np.sort(ws)[::-1]
    res = text_rank_output_matrix(graph, ws)
    for gl, gs, (l, s) in zip(gold_labels, gold_score, res):
        assert l == gl
        assert math.isclose(s, gs)


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


def test_text_rank():
    g1, g2 = random_graphs()
    g1_out = text_rank(g1, seed=1234)
    g2_out = text_rank(g2, seed=1234)
    for (g1_label, g1_score), (g2_label, g2_score) in zip(g1_out, g2_out):
        assert g1_label == g2_label
        math.isclose(g1_score, g2_score)
