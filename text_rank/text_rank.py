import math
import random
from operator import itemgetter
from typing import List, Tuple, Dict, Optional
import numpy as np
from tqdm import tqdm
from text_rank.graph import Graph, AdjacencyList, AdjacencyMatrix, Vertex


def sum_edges(edges: Dict[str, float]) -> float:
    return sum(edges.values())

def accum(vertex: Vertex, denom: List[float], ws: List[float]):
    acc = []
    for edge, weight in vertex.edges_in.items():
        acc.append(weight / denom[edge] * ws[edge])
    return math.fsum(acc)


def text_rank_list(
    graph: AdjacencyList, niter: int = 200, dampening: float = 0.85, quiet: bool = True, seed: Optional[int] = None
) -> List[Tuple[str, float]]:
    vertices = graph.vertices
    denom: List[float] = []
    ws: List[float] = []

    random.seed(seed)
    for vertex in vertices:
        denom.append(sum_edges(vertex.edges_out))
        ws.append(random.random())

    for _ in tqdm(range(niter), disable=quiet):
        updates: List[float] = []
        for vertex in vertices:
            acc = accum(vertex, denom, ws)
            updates.append(acc)
        for i, update in enumerate(updates):
            ws[i] = (1 - dampening) + dampening * update
    return sorted(zip(map(lambda v: v.value, vertices), ws), key=itemgetter(1), reverse=True)


def text_rank_matrix(
    graph: AdjacencyMatrix, niter: int = 200, dampening: float = 0.85, quiet: bool = True, seed: Optional[int] = None
) -> List[Tuple[str, float]]:
    vertices = list(graph.label2idx.keys())
    graph = graph.adjacency_matrix
    np.random.seed(seed)

    ws = np.random.rand(len(vertices), 1)
    denom = np.reshape(np.sum(graph, axis=1), (-1, 1))

    for _ in tqdm(range(niter), disable=quiet):
        update = np.sum(graph / denom * ws, axis=0)
        ws = np.reshape((1 - dampening) + dampening * update, (-1, 1))

    ws = np.reshape(ws, (-1,))
    return sorted(zip(vertices, ws), key=itemgetter(1), reverse=True)


def text_rank(
    graph: Graph, niter: int = 200, dampening: float = 0.85, quiet: bool = True, seed: Optional[int] = None
) -> List[Tuple[str, float]]:
    tr = text_rank_list if isinstance(graph, AdjacencyList) else text_rank_matrix
    return tr(graph, niter=niter, dampening=dampening, quiet=quiet, seed=seed)
