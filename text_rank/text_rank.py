import math
import random
from operator import itemgetter
from functools import singledispatch
from typing import List, Tuple, Dict, Optional
import numpy as np
from tqdm import tqdm
from text_rank.graph import Graph, AdjacencyList, AdjacencyMatrix, Vertex


def sum_edges(edges: Dict[str, float]) -> float:
    return sum(edges.values())


def accum(vertex: Vertex, ws: List[float], denom: List[float]):
    acc = []
    for edge, weight in vertex.edges_in.items():
        acc.append(weight / denom[edge] * ws[edge])
    return math.fsum(acc)


@singledispatch
def text_rank_init(graph: Graph, seed: Optional[int] = None) -> Tuple[List[float], List[float]]:
    raise NotImplementedError


@text_rank_init.register(AdjacencyList)
def _(graph: AdjacencyList, seed: Optional[int] = None) -> Tuple[List[float], List[float]]:
    denom: List[float] = []
    ws: List[float] = []
    random.seed(seed)
    for v in graph.vertices:
        denom.append(sum_edges(v.edges_out))
        ws.append(random.random())
    return ws, denom


@text_rank_init.register(AdjacencyMatrix)
def _(graph: AdjacencyMatrix, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    np.random.seed(seed)
    ws = np.random.rand(len(graph.label2idx), 1)
    denom = np.reshape(np.sum(graph.adjacency_matrix, axis=1), (-1, 1))
    return ws, denom


@singledispatch
def text_rank_update(graph: Graph, ws: List[float], denom: List[float], dampening: float = 0.85) -> List[float]:
    raise NotImplementedError


@text_rank_update.register(AdjacencyList)
def _(graph: AdjacencyList, ws: List[float], denom: List[float], dampening: float = 0.85) -> List[float]:
    updates: List[float] = []
    for v in graph.vertices:
        acc = accum(v, ws, denom)
        updates.append(acc)
    for i, update in enumerate(updates):
        ws[i] = (1 - dampening) + dampening * update
    return ws


@text_rank_update.register(AdjacencyMatrix)
def _(graph: AdjacencyMatrix, ws: np.ndarray, denom: np.ndarray, dampening: float = 0.85) -> np.ndarray:
    update = np.sum(graph.adjacency_matrix / denom * ws, axis=0)
    ws = np.reshape((1 - dampening) + dampening * update, (-1, 1))
    return ws


@singledispatch
def text_rank_output(graph: Graph, ws: List[float]) -> List[Tuple[str, float]]:
    raise NotImplementedError


@text_rank_output.register(AdjacencyList)
def _(graph: AdjacencyList, ws: List[float]) -> List[Tuple[str, float]]:
    return sorted(zip(map(lambda v: v.value, graph.vertices), ws), key=itemgetter(1), reverse=True)


@text_rank_output.register(AdjacencyMatrix)
def _(graph: AdjacencyMatrix, ws: np.ndarray) -> List[Tuple[str, float]]:
    ws = np.reshape(ws, (-1,))
    return sorted(zip(graph.label2idx.keys(), ws), key=itemgetter(1), reverse=True)


def text_rank(
    graph: Graph,
    dampening: float = 0.85,
    convergence: float = 0.0001,
    niter: int = 200,
    quiet: bool = True,
    seed: Optional[int] = None,
) -> List[Tuple[str, float]]:

    ws_prev, denom = text_rank_init(graph, seed)

    for _ in tqdm(range(niter), disable=quiet):
        ws = text_rank_update(graph, ws_prev, denom, dampening)
        if all(abs(p - c) < convergence for p, c in zip(ws_prev, ws)):
            break
        ws_prev = ws

    return text_rank_output(graph, ws)
