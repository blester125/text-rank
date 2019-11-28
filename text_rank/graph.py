import re
from math import log
from itertools import combinations
from collections import defaultdict
import numpy as np


class Vertex:
    def __init__(self, value):
        self.value = value
        self.edges_out = {}
        self.edges_in = {}

    def __str__(self):
        return "V(term={}, degree_in={}, degree_out={})".format(
            self.value, self.index, len(self.edges_in), len(self.edges_out)
        )


class Graph:
    def __init__(self, vocab):
        self.word2idx = vocab
        self.idx2word = {v: k for k, v in vocab.items()}

    def __getitem__(self, token):
        if isinstance(token, int):
            return self.idx2word[token]
        return self.word2idx[token]

    def add_edge(source, target, weight):
        raise NotImplementedError

    @property
    def density(self):
        raise NotImplementedError

    @property
    def edge_count(self):
        raise NotImplementedError

    @property
    def vertex_count(self):
        raise NotImplementedError


class AdjacencyList(Graph):
    def __init__(self, vocab):
        super().__init__(vocab)
        self.vertices = [Vertex(k) for k in self.word2idx.keys()]

    def add_edge(self, source, target, weight):
        source_idx = source if isinstance(source, int) else self[source]
        target_idx = target if isinstance(target, int) else self[target]
        source_node = self.vertices[source_idx]
        target_node = self.vertices[target_idx]
        source_node.edges_out[target_idx] = weight
        target_node.edges_in[source_idx] = weight


class AdjacencyMatrix(Graph):
    def __init__(self, vocab):
        super().__init__(vocab)
        self.adjacency_matrix = np.zeros((len(vocab), len(vocab)))

    def add_edge(self, source, target, weight):
        source = source if isinstance(source, int) else self[source]
        target = target if isinstance(target, int) else self[target]
        self.adjacency_matrix[source, target] = weight


def filter_pos(token):
    if not re.match(r"^[NJ]", token["pos"]) and token["pos"] != "ADJ" and token["pos"] != "CD":
        return False
    return True


def overlap(s1, s2):
    s1 = s1.split()
    s2 = s2.split()
    intersection = len(set(s1) & set(s2))
    norm = log(len(s1)) + log(len(s2))
    return intersection / norm


def build_vocab(tokens):
    vocab = defaultdict(lambda: len(vocab))
    for token in tokens:
        vocab[token]
    return {k: i for k, i in vocab.items()}


def keyword_graph(tokens, winsz=2, sim=lambda x, y: 1, GraphType=AdjacencyMatrix):
    tokens = list(map(lambda x: x["term"], filter(filter_pos, tokens)))
    vocab = build_vocab(tokens)
    graph = GraphType(vocab)

    for i, token in enumerate(tokens):
        source_idx = graph[token]
        min_ = max(0, i - winsz)
        max_ = min(len(tokens) - 1, i + winsz)
        for j in range(min_, max_):
            other = tokens[j]
            if i == j:
                continue
            target_idx = graph[other]
            graph.add_edge(source_idx, target_idx, sim(token, other))
            graph.add_edge(target_idx, source_idx, sim(other, token))
    return graph


def sentence_graph(sentences, sim=overlap, GraphType=AdjacencyMatrix):
    vocab = build_vocab(sentences)
    graph = GraphType(vocab)

    for src, tgt in combinations(sentences, 2):
        graph.add_edge(src, tgt, sim(src, tgt))
        graph.add_edge(tgt, src, sim(tgt, src))
    return graph


def show(vertices):
    import networkx as nx
    import matplotlib.pyplot as plt
    from networkx.drawing.nx_agraph import graphviz_layout, to_agraph
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
    A.layout("dot")
    A.draw("test.png")


def print_graph(g):
    for v in g.values():
        print(f"Vertex: {v.index}")
        print("\tOutbound Edges:")
        for idx, weight in v.edges_out.items():
            print(f"\t\t{v.index}->{idx}: {weight}")
        print("\tInbound Edges:")
        for idx, weight in v.edges_in.items():
            print(f"\t\t{idx}->{v.index}: {weight}")
