import random
from operator import itemgetter
from typing import List, Tuple, Dict
import numpy as np
from tqdm import tqdm
from text_rank.graph import Graph, AdjacencyList, AdjacencyMatrix, Vertex


def sum_edges(edges: Dict[str, float]) -> float:
    return sum(edges.values())


def accum(vertices: List[Vertex], i: int, denorm, ws):
    acc = 0
    for edge in vertices[i].edges_in:
        j = vertices[edge]
        edge_ji = j.edges_out.get(i)
        if edge_ji is not None:
            acc += edge_ji / denorm[edge] * ws[edge]
    return acc


def text_rank_list(
    graph: AdjacencyList, niter: int = 200, dampening: float = 0.85, quiet: bool = True
) -> List[Tuple[str, float]]:
    vertices = graph.vertices
    denorm = {}
    ws = []
    for i, vertex in enumerate(vertices):
        denorm[i] = sum_edges(vertex.edges_out)
        ws.append(random.random())

    for _ in tqdm(range(niter), disable=quiet):
        updates = {}
        for i in range(len(vertices)):
            acc = accum(vertices, i, denorm, ws)
            updates[i] = acc
        for i, update in updates.items():
            ws[i] = (1 - dampening) + dampening * update
    return sorted(zip(map(lambda v: v.value, vertices), ws), key=itemgetter(1), reverse=True)


def text_rank_matrix(
    graph: AdjacencyMatrix, niter: int = 200, dampening: float = 0.85, quiet: bool = True
) -> List[Tuple[str, float]]:
    vertices = list(graph.label2idx.keys())
    graph = graph.adjacency_matrix
    ws = np.random.rand(len(vertices), 1)
    denorm = np.reshape(np.sum(graph, axis=1), (-1, 1))
    for _ in tqdm(range(niter), disable=quiet):
        update = np.sum(graph / denorm * ws, axis=0)
        ws = np.reshape((1 - dampening) + dampening * update, (-1, 1))
    ws = np.reshape(ws, (-1,))
    return sorted(zip(vertices, ws), key=itemgetter(1), reverse=True)


def text_rank(graph: Graph, niter: int = 200, dampening: float = 0.85, quiet: bool = True) -> List[Tuple[str, float]]:
    tr = text_rank_list if isinstance(graph, AdjacencyList) else text_rank_matrix
    return tr(graph, niter=niter, dampening=dampening, quiet=quiet)
