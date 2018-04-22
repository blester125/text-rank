import random
import numpy as np
from tqdm import tqdm


# def sparse_textrank(vertices, graph, niter=200, dampaning=0.85):
#     ws = random(len(vertices), 1, 1)
#     print(ws.shape)
#     denorm = graph.sum(axis=1)
#     print(denorm.shape)
#     for _ in tqdm(range(niter)):
#         update = (graph / denorm).multiply(ws).sum(axis=0)
#         print(update.shape)
#         ws = (1 - dampening) + update.multiply(dampening)
#         print(ws.shape)
#     return sorted(zip(vertices, ws), key=lambda x: x[1], reverse=True)

def sum_edges(edges):
    return sum(edges.values())

def accum(vertices, i, denorm, ws):
    acc = 0
    for edge in vertices[i].edges_in:
        j = vertices[edge]
        edge_ji = j.edges_out.get(i)
        if edge_ji is not None:
            acc += edge_ji / denorm[edge] * ws[edge]
    return acc

def text_rank_list(vertices, niter=200, dampening=0.85, quiet=True):
    denorm = {}
    ws = []
    for i, vertex in enumerate(vertices):
        denorm[i] = sum_edges(vertex.edges_out)
        ws.append(random.random())

    for _ in tqdm(range(niter), disable=quiet):
        updates = {}
        for i  in range(len(vertices)):
            acc = accum(vertices, i, denorm, ws)
            updates[i] = acc
        for i, update in updates.items():
            ws[i] = (1 - dampening) + dampening * update
    return sorted(zip(vertices, ws), key=lambda x: x[1], reverse=True)


def text_rank(vertices, graph, niter=200, dampening=0.85, quiet=True):
    ws = np.random.rand(len(vertices), 1)
    denorm = np.reshape(np.sum(graph, axis=1), (-1, 1))
    for _ in tqdm(range(niter), disable=quiet):
        update = np.sum(graph / denorm * ws, axis=0)
        ws = np.reshape((1 - dampening) + dampening * update, (-1, 1))
    ws = np.reshape(ws, (-1,))
    return sorted(zip(vertices, ws), key=lambda x: x[1], reverse=True)
