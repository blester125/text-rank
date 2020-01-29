import math
import random
from enum import Enum
from operator import itemgetter
from functools import singledispatch
from typing import List, Tuple, Dict, Optional
import numpy as np
from text_rank.graph import Graph, AdjacencyList, AdjacencyMatrix, Vertex


ConvergenceType = Enum("ConvergenceType", "ALL ANY")


def sum_edges(edges: Dict[str, float]) -> float:
    """Calculate the total weight for a collection of edges.

    These edges are represented as a target -> weight mapping and is based on the representation
    we use in the vertex object, the edge doesn't have any information about the source node

    :param edges: A map of edges where the key is the target and the value is the weight
    :returns: The total weight of all edges
    """
    return sum(edges.values())


def accumulate_score(vertex: Vertex, ws: List[float], denom: List[float]):
    """Accumulate the scores from all nodes that have incoming connections to you.

    :param vertex: The node we all looking at
    :param ws: The scores of each node in the graph
    :param denom: The precomputed sum of outgoing edges for each node in the graph
    :returns: The sum of incoming scores weighted by incoming edge strength normalized by outgoing strengths
    """
    return math.fsum([weight / denom[edge] * ws[edge] for edge, weight in vertex.edges_in.items()])


@singledispatch
def text_rank_init(graph: Graph, uniform: bool = False, seed: Optional[int] = None) -> Tuple[List[float], List[float]]:
    """Generate the initial scores for each node and pre-compute the outgoing strength.

    The sum of the weights for outbound edges for a given node doesn't change as text rank runs because
    it is based only on the values in the graph, not on ws for the node so we can pre-compute and reuse
    it instead of always recalculating it.

    :param graph: The graph we will run text rank on
    :param uniform: Should we initialize state vector to have equal prob for each node?
    :param seed: A seed for the RNG if we want reproduceability
    :returns: The initial scores for each node and the edge normalization factor for each node
    """
    raise NotImplementedError


@text_rank_init.register(AdjacencyList)
def text_rank_init_list(
    graph: AdjacencyList, uniform: bool = False, seed: Optional[int] = None
) -> Tuple[List[float], List[float]]:
    random.seed(seed)
    denom = [sum_edges(v.edges_out) for v in graph.vertices]
    # If the sum off all outgoing edges of V_j is 0.0 then the incoming edge from V_j to V_i will be 0.0
    # We can use anything as the denominator and the value will still be zero
    denom = [d if d != 0.0 else 1.0 for d in denom]
    if uniform:
        ws = [1 / len(graph.vertices) for _ in graph.vertices]
    else:
        ws = [random.random() for _ in graph.vertices]
        norm = sum(ws)
        ws = [w / norm for w in ws]
    return ws, denom


@text_rank_init.register(AdjacencyMatrix)
def text_rank_init_matrix(
    graph: AdjacencyMatrix, uniform: bool = False, seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    random.seed(seed)
    if uniform:
        ws = np.ones(graph.vertex_count) / graph.vertex_count
    else:
        ws = np.array([random.random() for _ in range(graph.vertex_count)])
        ws = ws / np.sum(ws)
    denom = np.reshape(np.sum(graph.adjacency_matrix, axis=1), (-1, 1))
    # If the sum off all outgoing edges of V_j is 0.0 then the incoming edge from V_j to V_i will be 0.0
    # We can use anything as the denominator and the value will still be zero
    denom[denom == 0.0] = 1.0
    return ws, denom


@singledispatch
def text_rank_update(graph: Graph, ws: List[float], denom: List[float], damping: float = 0.85) -> List[float]:
    """Calculate new score for each node.

    This is the main step in text rank

    :param graph: The text rank graph
    :param ws: The scores for each node
    :param denom: The outbound weight normalization factor for each node, pre-computed for efficiency
    :param damping: A scalar between 0 and 1. Used to simulate randomly jumping from one vertex to another.

    :returns: The updated scores for each node
    """
    raise NotImplementedError


@text_rank_update.register(AdjacencyList)
def text_rank_update_list(
    graph: AdjacencyList, ws: List[float], denom: List[float], damping: float = 0.85
) -> List[float]:
    updates = [accumulate_score(v, ws, denom) for v in graph.vertices]
    # We collect the updated scores for each node and apply them after. If we were to apply these
    # updates as they happen we would get different results than from the vectorized version used
    # in the adjacency matrix version
    ws = [(1 - damping) + damping * update for update in updates]
    return ws


@text_rank_update.register(AdjacencyMatrix)
def text_rank_update_matrix(
    graph: AdjacencyMatrix, ws: np.ndarray, denom: np.ndarray, damping: float = 0.85
) -> np.ndarray:
    update = np.dot(ws, graph.adjacency_matrix / denom)
    ws = (1 - damping) + damping * update
    return ws


@singledispatch
def text_rank_output(graph: Graph, ws: List[float]) -> List[Tuple[str, float]]:
    raise NotImplementedError


@text_rank_output.register(AdjacencyList)
def text_rank_output_list(graph: AdjacencyList, ws: List[float]) -> List[Tuple[str, float]]:
    norm = sum(ws)
    ws = [w / norm for w in ws]
    return sorted(zip(map(lambda v: v.value, graph.vertices), ws), key=itemgetter(1), reverse=True)


@text_rank_output.register(AdjacencyMatrix)
def text_rank_output_matrix(graph: AdjacencyMatrix, ws: np.ndarray) -> List[Tuple[str, float]]:
    ws = ws / np.sum(ws)
    return sorted(zip(graph.label2idx.keys(), ws), key=itemgetter(1), reverse=True)


def text_rank(
    graph: Graph,
    damping: float = 0.85,
    convergence: float = 0.0001,
    convergence_type: ConvergenceType = ConvergenceType.ALL,
    niter: int = 200,
    uniform: bool = False,
    seed: Optional[int] = None,
) -> List[Tuple[str, float]]:
    """Implementation of text rank from here https://www.aclweb.org/anthology/W04-3252.pdf

    :param graph: The graph we are running text rank on
    :param damping: A scalar between 0 and 1. Used to simulate randomly jumping from one vertex to another.
    :param convergence: An early stopping criteria, when any or all of the node scores change by less than `convergence`
        we stop updating the graph. Set to `0` to turn off early stopping.
    :param convergence_type: Should we stop when all nodes move less than `convergence` or when a single node does
    :param niter: An upper bound on the number of iterations to run
    :param uniform: Should we initialize state vector to have equal prob for each node?
    :param seed: A reproducability seed to initialization of the node scores.

    :returns: Pairs of (node label, scores) sorted by score
    """
    if not 0 <= damping <= 1:
        raise ValueError(f"damping must be between `0` and `1`, got {damping}")
    converge = all if convergence_type is ConvergenceType.ALL else any

    ws_prev, denom = text_rank_init(graph, uniform=uniform, seed=seed)

    for _ in range(niter):
        ws = text_rank_update(graph, ws_prev, denom, damping)
        if converge(abs(p - c) < convergence for p, c in zip(ws_prev, ws)):
            break
        ws_prev = ws

    return text_rank_output(graph, ws)
