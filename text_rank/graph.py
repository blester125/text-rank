from itertools import combinations
import numpy as np
from text_rank.utils import filter_pos, build_vocab, overlap


class Vertex:
    def __init__(self, value):
        self.value = value
        self.edges_out = {}
        self.edges_in = {}

    @property
    def degree_in(self):
        return len(self.edges_in)

    @property
    def degree_out(self):
        return len(self.edges_out)

    def __str__(self):
        return f"V(term={self.value}, in={self.degree_in}, out={self.degree_out})"


class Graph:
    def __init__(self, vocab):
        self.label2idx = vocab
        self.idx2label = {v: k for k, v in vocab.items()}

    def __getitem__(self, token):
        if isinstance(token, int):
            return self.idx2label[token]
        return self.label2idx[token]

    def add_edge(source, target, weight):
        raise NotImplementedError

    @property
    def density(self):
        return self.edge_count / (self.vertex_count * (self.vertex_count - 1))

    @property
    def edge_count(self):
        raise NotImplementedError

    @property
    def vertex_count(self):
        raise NotImplementedError

    def __str__(self):
        return f"G(V={self.vertex_count}, E={self.edge_count}, D={self.density})"

    def print_graph(self):
        raise NotImplementedError

    def to_dot(self):
        raise NotImplementedError


class AdjacencyList(Graph):
    def __init__(self, vocab):
        super().__init__(vocab)
        self.vertices = [Vertex(k) for k in self.label2idx.keys()]

    def add_edge(self, source, target, weight):
        source_idx = source if isinstance(source, int) else self[source]
        target_idx = target if isinstance(target, int) else self[target]
        source_node = self.vertices[source_idx]
        target_node = self.vertices[target_idx]
        source_node.edges_out[target_idx] = weight
        target_node.edges_in[source_idx] = weight

    @property
    def vertex_count(self):
        return len(self.vertices)

    @property
    def edge_count(self):
        return sum(v.degree_out for v in self.vertices)

    def print_graph(self):
        print(str(self))
        for v in self.vertices:
            print(f"\tVertex {self[v.value]}: {v}")
            print(f"\t\tOutbound:")
            for idx, weight in v.edges_out.items():
                print(f"\t\t\t{self[v.value]} -> {idx}: {weight}")
            print(f"\t\tInbound:")
            for idx, weight in v.edges_in.items():
                print(f"\t\t\t{self[v.value]} <- {idx}: {weight}")

    def to_dot(self):
        dot = ["digraph G {"]
        for v in self.vertices:
            dot.append(f'\t{self[v.value]} [label="{v.value}"]')
            for idx, weight in v.edges_out.items():
                dot.append(f'\t{self[v.value]} -> {idx} [label="{weight}"]')
        dot.append("}")
        return "\n".join(dot)


class AdjacencyMatrix(Graph):
    def __init__(self, vocab):
        super().__init__(vocab)
        self.adjacency_matrix = np.zeros((len(vocab), len(vocab)))

    def add_edge(self, source, target, weight):
        source = source if isinstance(source, int) else self[source]
        target = target if isinstance(target, int) else self[target]
        self.adjacency_matrix[source, target] = weight

    @property
    def vertex_count(self):
        return self.adjacency_matrix.shape[0]

    @property
    def edge_count(self):
        return np.sum(self.adjacency_matrix != 0)

    def print_graph(self):
        print(str(self))
        for idx, label in self.idx2label.items():
            print(f"\tVertex {idx}: {label}")
            print(f"\t\tOutbound:")
            for i, weight in enumerate(self.adjacency_matrix[idx, :]):
                if weight == 0.0:
                    continue
                print(f"\t\t\t{idx} -> {i}: {weight}")
            print(f"\t\tInbound:")
            for i, weight in enumerate(self.adjacency_matrix[:, idx]):
                if weight == 0.0:
                    continue
                print(f"\t\t\t{idx} <- {i}: {weight}")

    def to_dot(self):
        dot = ["digraph G {"]
        for idx, label in self.idx2label.items():
            dot.append(f'\t{idx} [label="{label}"]')
            for i, weight in enumerate(self.adjacency_matrix[idx, :]):
                if weight == 0.0:
                    continue
                dot.append(f'\t{idx} -> {i} [label="{weight}"]')
        dot.append("}")
        return "\n".join(dot)


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
