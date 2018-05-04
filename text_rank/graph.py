import re
from math import log
from itertools import combinations
import numpy as np
#from scipy.sparse import lil_matrix, random


class Vertex:
    def __init__(self, value, index):
        self.value = value
        self.index = index
        self.edges_out = {}
        self.edges_in = {}


def filter_pos(token):
    if (
        not re.match(r"^[NJ]", token['pos']) and
        token['pos'] != 'ADJ' and
        token['pos'] != 'CD'
    ):
        return False
    return True


def overlap(s1, s2):
    intersection = len(set(s1) & set(s2))
    norm = log(len(s1)) + log(len(s2))
    return intersection / norm


def keyword_graph(tokens, winsz=2):
    vertices = []
    vocab = {}

    tokens = list(filter(filter_pos, tokens))
    for i in range(len(tokens)):
        token = tokens[i]
        if token['term'] in vocab:
            idx = vocab[token['term']]
        else:
            idx = len(vocab)
            vocab[token['term']] = idx
        if idx >= len(vertices):
            vertices.append(token['term'])

    graph = np.zeros((len(vertices), len(vertices)))
    for i in range(len(tokens)):
        token = tokens[i]
        source_idx = vocab[token['term']]
        source_node = vertices[source_idx]
        min_ = max(0, i - winsz)
        max_ = min(len(tokens) - 1, i + winsz)
        for j in range(min_, max_):
            other = tokens[j]
            if i == j:
                continue
            target_idx = vocab[other['term']]
            graph[source_idx, target_idx] = 1
            graph[target_idx, source_idx] = 1
    return vertices, graph


def keyword_graph_list(tokens, winsz=2):
    vertices = []
    vocab = {}

    tokens = list(filter(filter_pos, tokens))
    for i in range(len(tokens)):
        token = tokens[i]
        if token['term'] in vocab:
            idx = vocab[token['term']]
        else:
            idx = len(vocab)
            vocab[token['term']] = idx
        if idx >= len(vertices):
            vertices.append(Vertex(token['term'], idx))

    for i in range(len(tokens)):
        token = tokens[i]
        source_idx = vocab[token['term']]
        source_node = vertices[source_idx]
        min_ = max(0, i - winsz)
        max_ = min(len(tokens) - 1, i + winsz)
        for j in range(min_, max_):
            other = tokens[j]
            if i == j:
                continue
            target_idx = vocab[other['term']]
            target_node = vertices[target_idx]
            source_node.edges_out[target_idx] = 1
            source_node.edges_in[target_idx] = 1
            target_node.edges_out[source_idx] = 1
            target_node.edges_in[source_idx] = 1
    return vertices


def sentence_graph(sentences, sim=overlap):
    vertices = [None] * len(sentences)
    graph = np.zeros((len(sentences), len(sentences)))
    for i, j in combinations(range(len(sentences)), 2):
        vertices[i] = sentences[i]
        vertices[j] = sentences[j]
        graph[i, j] = sim(sentences[i], sentences[j])
        graph[j, i] = sim(sentences[j], sentences[i])
    return vertices, graph


def sentence_graph_list(sentences, sim=overlap):
    vertices = [None] * len(sentences)
    for i, j in combinations(range(len(sentences)), 2):
        score_ij = sim(sentences[i], sentences[j])
        score_ji = sim(sentences[j], sentences[i])
        vertices[i] = vertices[i] or Vertex(sentences[i], i)
        vertices[j] = vertices[j] or Vertex(sentences[j], j)
        vertices[i].edges_out[j] = score_ji
        vertices[i].edges_in[j] = score_ij
        vertices[j].edges_out[i] = score_ji
        vertices[j].edges_in[i] = score_ij
    return vertices


# def sparse_graph(sentences, sim=overlap):
#     graph = lil_matrix((len(sentences), len(sentences)), dtype=np.float64)
#     vertices = [None] * len(sentences)
#     for i, j in combinations(range(len(sentences)), 2):
#         graph[i, j] = sim(sentences[i], sentences[j])
#         graph[j, i] = sim(sentences[j], sentences[i])
#         vertices[i] = vertices[i] or Vertex(sentences[i], i)
#         vertices[j] = vertices[i] or Vertex(sentences[j], j)
#     return vertices, graph


def show(vertices):
    import networkx as nx
    import matplotlib.pyplot as plt
    from networkx.drawing.nx_agraph import graphviz_layout,to_agraph
    import pygraphviz as pgv
    G = nx.MultiDiGraph()
    for vertex in vertices.values():
        G.add_node(vertex.index)
        for edge, weight in vertex.edges_out.items():
            G.add_edge(vertex.index, edge, weight)
        for edge, weight in vertex.edges_in.items():
            G.add_edge(edge, vertex.index, weight)
    A = to_agraph(G)
    print(A)
    A.layout('dot')
    A.draw('test.png')


def print_graph(g):
    for v in g.values():
        print(f"Vertex: {v.index}")
        print("\tOutbound Edges:")
        for idx, weight in v.edges_out.items():
            print(f"\t\t{v.index}->{idx}: {weight}")
        print("\tInbound Edges:")
        for idx, weight in v.edges_in.items():
            print(f"\t\t{idx}->{v.index}: {weight}")
