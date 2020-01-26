from itertools import combinations
from collections import defaultdict
from typing import Optional, Dict, Union, Callable, Tuple, Type, List
import numpy as np
from text_rank.utils import overlap, cooccurrence, norm_sentence, norm_token, filter_pos, build_vocab


class Vertex:
    def __init__(self, value: str):
        """A vertex in a graph.

        :param value: The label for this vertex
        """
        self.value = value
        self._edges_out: Dict[int, float] = {}
        self._edges_in: Dict[int, float] = {}

    @property
    def edges_out(self) -> Dict[int, float]:
        """A mapping of target vertex to weight representing the edges with this vertex as the source."""
        return self._edges_out

    @property
    def edges_in(self) -> Dict[int, float]:
        """A mapping of source vertex to weight representing the edges that end at this vertex."""
        return self._edges_in

    @property
    def degree_in(self) -> int:
        """The number of edges that end at this vertex."""
        return len(self.edges_in)

    @property
    def degree_out(self) -> int:
        """The number of edges that start at this vertex."""
        return len(self.edges_out)

    def __str__(self) -> str:
        """A summary of this vertex."""
        return f"V(term={self.value}, in={self.degree_in}, out={self.degree_out})"

    def __eq__(self, other) -> bool:
        if not isinstance(other, Vertex):
            raise TypeError(f"Can only compare to other Vertex objects, got {type(other)}")
        if self is other:
            return True
        if self.value != other.value:
            return False
        if self._edges_out != other._edges_out:
            return False
        if self._edges_in != other._edges_in:
            return False
        return True


class Graph:
    def __init__(self, vertices: Union[Dict[str, int], List[str]]):
        """A directed simple graph.

        :param vertices: A mapping of vertex labels to integer indices or a list of vertex labels.
            If the latter then indices are assigned in order
        """
        if isinstance(vertices, dict):
            if set(vertices.values()) != set(range(len(vertices))):
                raise ValueError("Vertex indices must be contiguous")
            self.label2idx: Dict[str, int] = vertices
        else:
            self.label2idx: Dict[str, int] = {n: i for i, n in enumerate(vertices)}
        self.idx2label: Dict[int, str] = {i: k for k, i in self.label2idx.items()}

    def __getitem__(self, key: Union[str, int]) -> Union[int, str]:
        """Get either the index or vertex label based on the other one.

        :param key: The vertex label or index
        :returns: the vertex index of the label is given or the vertex label if index is given.
        """
        if isinstance(key, int):
            return self.idx2label[key]
        return self.label2idx[key]

    def __contains__(self, key: Union[str, int]) -> bool:
        """Check if the graph has a vertex labeled key.

        :param key: The vertex label or index you are asking about
        :returns: True if the vertex exists, False otherwise
        """
        if isinstance(key, int):
            return key in self.idx2label
        return key in self.label2idx

    def add_vertex(self, label: Optional[str]) -> str:
        """Add a vertex to the graph.

        :param label: The label to give the new vertex.
        :returns: The vertex label
        """
        raise NotImplementedError

    def _add_vertex(self, label: Optional[str]) -> str:
        """Add a vertex to the label2idx with a given label or a new one.

        :param label: The label for the new vertex
        :returns: The label for the new vertex
        """
        if label is None:
            label = str(len(self.label2idx))
        if label in self.label2idx:
            raise ValueError(f"Node labels must be unique, label {label} is already in use.")
        idx = len(self.label2idx)
        self.label2idx[label] = idx
        self.idx2label[idx] = label
        return idx

    def add_edge(self, source: Union[str, int], target: Union[str, int], weight: float = 1.0) -> None:
        """Add an edge to the graph.

        :param source: The vertex label or index of the edge source
        :param target: The vertex label or index of the edge target
        :param weight: The weight to put on the edge
        :raises ValueError: When the source and target node are the same, when the weight is less than zero
        """
        raise NotImplementedError

    @property
    def density(self) -> float:
        """Get the density of the graph.

        The density of a graph is the ratio of edges that the graph has to the number
        it could possibly have, this is bounded by 0 and 1.

        ```math
            D = \frac{|E|}{|V|(|V| - 1)}
        ```
        """
        return self.edge_count / (self.vertex_count * (self.vertex_count - 1))

    @property
    def edge_count(self) -> int:
        """The number of edges in the graph."""
        raise NotImplementedError

    @property
    def vertex_count(self) -> int:
        """The number of vertices in the graph."""
        raise NotImplementedError

    def __str__(self) -> str:
        """A summary of the graph.

        Graph summary includes the number of vertices and edges as well as the
        density of the graph.
        """
        return f"G(V={self.vertex_count}, E={self.edge_count}, D={self.density})"

    def print_graph(self, label_lengths: Optional[int] = None) -> None:
        """Print the graph is a human readable way.

        :param label_length: A cut-off on the length of a single label while printing.
        """
        raise NotImplementedError

    def to_dot(self, directed: bool = False, label_length: Optional[int] = None) -> str:
        """Get a dot representation of the graph.

        The dot graph includes vertex labels and edge weights.

        :param directed: Should the dot representation be directed of not. Most graphs created
            in the package are directed but have the same weight in either direction so we
            can collapse the graph into an undirected weighted graph for cleaner plotting.
            Note: Collapsing this doesn't check that the weights in each direction are the same,
            it just plots a single edge.
        :param label_length: A cut-off on the length allowed for a single label in the printing.

        :returns: The representations of the graph as a dot string.
        """
        raise NotImplementedError


class AdjacencyList(Graph):
    def __init__(self, vertices: Dict[str, int]):
        """A directed simple graph represented as an AdjacencyList.

        Note:
            This is slightly different than a true adjacency list, it is a List of Vertex
            object rather than a list of lists. The Vertex objects help things like labels
            and dicts of the edges. This gives faster lookup for specific edges compared
            to a normal adjacency list.

        :param vertices: A mapping of vertex labels to integer indices or a list of vertex labels.
            If the latter then indices are assigned in order
        """
        super().__init__(vertices)
        self._vertices: List[Vertex] = [Vertex(l) for l in self.label2idx]

    @property
    def vertices(self) -> List[Vertex]:
        """The vertices in this graph."""
        return self._vertices

    def add_vertex(self, label: Optional[str]) -> int:
        """Add a vertex to the graph.

        :param label: The label to give the new vertex.
        :returns: The vertex index
        """
        idx = self._add_vertex(label)
        if idx != len(self.vertices):
            raise ValueError(
                "The added vertex has a label that is out of order, expected: {len(self.vertices)} found: {idx}"
            )
        self.vertices.append(Vertex(label))
        return idx

    def add_edge(self, source: Union[str, int], target: Union[str, int], weight: float = 1.0) -> None:
        """Add an edge to the graph.

        :param source: The vertex label or index of the edge source
        :param target: The vertex label or index of the edge target
        :param weight: The weight to put on the edge
        :raises ValueError: When the source and target node are the same, when the weight is less than zero
        """
        if weight < 0.0:
            raise ValueError(f"Edge weight must be greater than zero, got {weight}")
        source_idx = source if isinstance(source, int) else self[source]
        target_idx = target if isinstance(target, int) else self[target]
        if source_idx == target_idx:
            raise ValueError(f"Self loops are not allowed, found edge with source and target if {source_idx}")
        source_vertex = self.vertices[source_idx]
        target_vertex = self.vertices[target_idx]
        source_vertex.edges_out[target_idx] = weight
        target_vertex.edges_in[source_idx] = weight

    @property
    def vertex_count(self) -> int:
        """The number of vertices in the graph."""
        return len(self.vertices)

    @property
    def edge_count(self) -> int:
        """The number of edges in the graph."""
        return sum(v.degree_out for v in self.vertices)

    def print_graph(self, label_length: Optional[int] = None) -> None:
        """Print the graph is a human readable way.

        :param label_length: A cut-off on the length of a single label while printing.
        """
        print(str(self))
        for v in self.vertices:
            print(f"\tVertex {self[v.value]}: {v.value[:label_length]}")
            print(f"\t\tOutbound:")
            for idx, weight in v.edges_out.items():
                print(f"\t\t\t{self[v.value]} -> {idx}: {weight}")
            print(f"\t\tInbound:")
            for idx, weight in v.edges_in.items():
                print(f"\t\t\t{self[v.value]} <- {idx}: {weight}")

    def to_dot(self, directed: bool = False, label_length: Optional[int] = None) -> str:
        """Get a dot representation of the graph.

        The dot graph includes vertex labels and edge weights.

        :param directed: Should the dot representation be directed of not. Most graphs created
            in the package are directed but have the same weight in either direction so we
            can collapse the graph into an undirected weighted graph for cleaner plotting.
            Note: Collapsing this doesn't check that the weights in each direction are the same,
            it just plots a single edge.
        :param label_length: A cut-off on the length allowed for a single label in the printing.

        :returns: The representations of the graph as a dot string.
        """
        if directed:
            return self._to_directed_dot(label_length)
        return self._to_undirected_dot(label_length)

    def _to_directed_dot(self, label_length: Optional[int] = None) -> str:
        """Get a dot representation of the graph as a directed graph.

        :param label_length: A cut-off on the length allowed for a single label in the printing.

        :returns: The representations of the graph as a dot string.
        """
        dot = ["digraph G {"]
        for v in self.vertices:
            dot.append(f'\t{self[v.value]} [label="{v.value[:label_length]}"];')
            for idx, weight in v.edges_out.items():
                dot.append(f'\t{self[v.value]} -> {idx} [label="{weight}"];')
        dot.append("}")
        return "\n".join(dot)

    def _to_undirected_dot(self, label_length: Optional[int] = None) -> str:
        """Get a dot representation of the graph as a undirected graph.

        Note:
            This doesn't check that graph edges can actually be collapsed into a single edge.

        :param label_length: A cut-off on the length allowed for a single label in the printing.

        :returns: The representations of the graph as a dot string.
        """
        dot = ["graph G {"]
        edges = set()
        for v in self.vertices:
            dot.append(f'\t{self[v.value]} [label="{v.value[:label_length]}"];')
            for idx, weight in v.edges_out.items():
                if (self[v.value], idx) in edges or (idx, self[v.value]) in edges:
                    continue
                dot.append(f'\t{self[v.value]} -- {idx} [label="{weight}"];')
                edges.add((self[v.value], idx))
        dot.append("}")
        return "\n".join(dot)


class AdjacencyMatrix(Graph):
    def __init__(self, vertices: Dict[str, int]):
        """A graph with edges represented as an adjacency matrix.

        Note:
            The format of the adjacency matrix is src is the rows, tgt is the columns, i.e.:
                adjacency_matrix[src, tgt] = weight_src_to_tgt

        :param vertices: A mapping of vertex labels to integer indices or a list of vertex labels.
            If the latter then indices are assigned in order
        """
        super().__init__(vertices)
        self._adjacency_matrix = np.zeros((len(vertices), len(vertices)))

    @property
    def adjacency_matrix(self) -> np.ndarray:
        """The adjacency matrix of graph edges.

        Note:
            adjacency_matrix[x, y] is the edge starting at x and ending at y.
        """
        return self._adjacency_matrix

    def add_vertex(self, label: Optional[str]) -> int:
        """Add a vertex to the graph.

        :param label: The label to give the new vertex.
        :returns: The vertex index
        """
        idx = self._add_vertex(label)
        if idx != self._adjacency_matrix.shape[0]:
            raise ValueError(
                "The added vertex has a label that is out of order, expected: {self.adjacency_matrix.shape[0]} found: {idx}"
            )
        adj = np.zeros((idx + 1, idx + 1))
        adj[: self.adjacency_matrix.shape[0], : self.adjacency_matrix.shape[1]] = self.adjacency_matrix
        self._adjacency_matrix = adj
        return idx

    def add_edge(self, source: Union[str, int], target: Union[str, int], weight: float = 1.0) -> None:
        """Add an edge to the graph.

        :param source: The vertex label or index of the edge source
        :param target: The vertex label or index of the edge target
        :param weight: The weight to put on the edge
        :raises ValueError: When the source and target node are the same, when the weight is less than zero
        """
        if weight < 0.0:
            raise ValueError(f"Edge weight must be greater than zero, got {weight}")
        source_idx = source if isinstance(source, int) else self[source]
        target_idx = target if isinstance(target, int) else self[target]
        if source_idx == target_idx:
            raise ValueError(f"Self loops are not allowed, found edge with source and target if {source_idx}")
        self.adjacency_matrix[source_idx, target_idx] = weight

    @property
    def vertex_count(self) -> int:
        """The number of vertices in the graph."""
        return self.adjacency_matrix.shape[0]

    @property
    def edge_count(self) -> int:
        """The number of edges in the graph."""
        return np.sum(self.adjacency_matrix != 0)

    def print_graph(self, label_length: Optional[int] = None) -> None:
        """Print the graph is a human readable way.

        :param label_length: A cut-off on the length of a single label while printing.
        """
        print(str(self))
        for idx, label in self.idx2label.items():
            print(f"\tVertex {idx}: {label[:label_length]}")
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

    def to_dot(self, directed: bool = False, label_length: Optional[int] = None) -> str:
        """Get a dot representation of the graph.

        The dot graph includes vertex labels and edge weights.

        :param directed: Should the dot representation be directed of not. Most graphs created
            in the package are directed but have the same weight in either direction so we
            can collapse the graph into an undirected weighted graph for cleaner plotting.
            Note: Collapsing this doesn't check that the weights in each direction are the same,
            it just plots a single edge.
        :param label_length: A cut-off on the length allowed for a single label in the printing.

        :returns: The representations of the graph as a dot string.
        """
        if directed:
            return self._to_directed_dot(label_length)
        return self._to_undirected_dot(label_length)

    def _to_directed_dot(self, label_length: Optional[int] = None) -> str:
        """Get a dot representation of the graph as a directed graph.

        :param label_length: A cut-off on the length allowed for a single label in the printing.

        :returns: The representations of the graph as a dot string.
        """
        dot = ["digraph G {"]
        for idx, label in self.idx2label.items():
            dot.append(f'\t{idx} [label="{label[:label_length]}"];')
            for i, weight in enumerate(self.adjacency_matrix[idx, :]):
                if weight == 0.0:
                    continue
                dot.append(f'\t{idx} -> {i} [label="{weight}"];')
        dot.append("}")
        return "\n".join(dot)

    def _to_undirected_dot(self, label_length: Optional[int] = None) -> str:
        """Get a dot representation of the graph as a undirected graph.

        Note:
            This doesn't check that graph edges can actually be collapsed into a single edge.

        :param label_length: A cut-off on the length allowed for a single label in the printing.

        :returns: The representations of the graph as a dot string.
        """
        dot = ["graph G {"]
        for idx, label in self.idx2label.items():
            dot.append(f'\t{idx} [label="{label[:label_length]}"];')
            for i, weight in enumerate(self.adjacency_matrix[idx, :]):
                if weight == 0:
                    continue
                if i >= idx:
                    continue
                dot.append(f'\t{idx} -- {i} [label="{weight}"];')
        dot.append("}")
        return "\n".join(dot)


def keyword_graph(
    tokens: List[Dict[str, str]],
    winsz: int = 2,
    sim: Callable[..., float] = lambda x, y, **kwargs: 1,
    norm: Callable[[str], str] = norm_token,
    filt: Callable[[Dict], Dict] = filter_pos,
    GraphType: Type[Graph] = AdjacencyMatrix,
) -> Tuple[Graph, Dict[str, List[int]]]:
    """Generate a keyword graph where edges are restricted to a window.

    Note:
        This also generates a dict mapping normalized vertex labels to their offsets in the original
        data. This can be used to run text rank on normalized data but return the original strings.
        It also allows the keywords post processing of combining contiguous keywords.

    :param tokens: The text to find keywords in, can be dict that have the keys `surface` and `pos` or
        just a string. If just a string then filtering of tokens is not done.
    :param winsz: The size of a window around each word that it can connect to. This window is calculated
        in the non-filtered token list.
    :param sim: A callable that returns the similarity between two vertices, used to set the weight of the edge.
        The callable should have a signature like:
            sim(
                normed_s1,
                normed_s2,
                context=context,
                raw_s1=raw_s1,
                raw_s2=raw_s2,
                raw_context=raw_context,
                s1_idx=s1_idx,
                s2_idx=s2_idx,
            ) -> float:
        Where normed_s1/2 is the normalized string of two keywords, context is all of the normalized tokens,
        raw_s1/2/context are the prenormalized versions and s1/2_idx are the indices of the keywords in the
        original sentence. This should allow complex similarity functions. The context is the whole list of
        tokens, not just the window around one.
    :param norm: A function the returns a normalized version of the input string. Default implementation
        lowercases string and removes non alpha-numeric characters.
        This is used to unify similar vertices, i.e. `Hurricane` and `hurricane` should be the same vertex
        and will be with normalization.
    :param filt: A function that will filter tokens based on pos tags if the inputs are Dicts.
    :param GraphType: The Graph class to use.

    :returns: The constructed graph and offsets mapping normalized vertex labels to their place in the original text.
    """
    # Unpack `t` in the dict (which adds all keys from t) and over-write the `surface` key
    tokens = [{**t, "normed": norm(t["surface"])} for t in tokens]

    valid_tokens = []
    offsets = defaultdict(list)
    for i, token in enumerate(tokens):
        if filt(token):
            surface = token["normed"]
            offsets[surface].append(i)
            valid_tokens.append(surface)

    vocab = build_vocab(valid_tokens)
    graph = GraphType(vocab)

    surfaces = [t["surface"] for t in tokens]
    normed_surfaces = [t["normed"] for t in tokens]

    for i, source in enumerate(tokens):
        source_surf = source["normed"]
        if source_surf not in graph:
            continue
        source_idx = graph[source_surf]
        min_ = max(0, i - winsz)
        max_ = min(len(surfaces) - 1, i + winsz)
        for j in range(min_, max_):
            if i == j:
                continue
            target = tokens[j]
            target_surf = target["normed"]
            # Don't make an edge to an un-included word or a self loop
            if target_surf not in graph or target_surf == source_surf:
                continue
            target_idx = graph[target_surf]
            graph.add_edge(
                source_idx,
                target_idx,
                sim(
                    source_surf,
                    target_surf,
                    context=normed_surfaces,
                    raw_s1=source["surface"],
                    raw_s2=target["surface"],
                    raw_context=surfaces,
                    s1_idx=source_idx,
                    s2_idx=target_idx,
                ),
            )
            graph.add_edge(
                target_idx,
                source_idx,
                sim(
                    target_surf,
                    source_surf,
                    context=normed_surfaces,
                    raw_s1=target["surface"],
                    raw_s2=source["surface"],
                    raw_context=surfaces,
                    s1_idx=target_idx,
                    s2_idx=source_idx,
                ),
            )
    return graph, offsets


def sentence_graph(
    sentences: List[str],
    sim: Callable[..., float] = overlap,
    norm: Callable[[str], str] = norm_sentence,
    GraphType: Type[Graph] = AdjacencyMatrix,
) -> Tuple[Graph, Dict[str, List[int]]]:
    """Generate a fully connected graph with edges between all sentences.

    Note:
        This also generates a dict mapping normalized vertex labels to their offsets in the original
        data. This can be used to run text rank on normalized data but return the original strings.
        You can also sort the output by offsets to make it maybe more readable?

    :param sentences: The sentences to summarize.
    :param sim: A callable that returns the similarity between two vertices, used to set the weight of the edge.
        The callable should have a signature like:
            sim(
                normed_s1,
                normed_s2,
                raw_s1=raw_s1,
                raw_s2=raw_s2,
                s1_idx=s1_idx,
                s2_idx=s2_idx,
            ) -> float:
        Where normed_s1/2 is the normalized strings of the two sentences, raw_s1/2 is the version of the sentence
        before getting normalized and s1/2_idx is the index of the sentences in the token list. This should
        facilitate both simple and complex similarity functions and also experiments that the actual flow of text
        to determine connections.
    :param norm: A function the returns a normalized version of the input sentence. Default implementation lowercases
        string and removes non alpha-numeric characters.
        This is used so simple similarity functions like the set overlap in the paper work well.
    :param GraphType: The Graph class to use.

    :returns: The constructed graph and offsets mapping normalized vertex labels to their place in the original text.
    """
    offsets = defaultdict(list)
    normed = [norm(sentence) for sentence in sentences]
    for i, norm in enumerate(normed):
        offsets[norm].append(i)

    vocab = build_vocab(normed)
    graph = GraphType(vocab)

    for (i, src), (j, tgt) in combinations(enumerate(normed), 2):
        graph.add_edge(src, tgt, sim(src, tgt, raw_s1=sentences[i], raw_s2=sentences[j], s1_idx=i, s2_idx=j))
        graph.add_edge(tgt, src, sim(tgt, src, raw_s1=sentences[j], raw_s2=sentences[i], s1_idx=j, s2_idx=i))

    return graph, offsets
