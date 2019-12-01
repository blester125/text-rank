from text_rank.utils import overlap
from text_rank.graph import sentence_graph, AdjacencyMatrix
from text_rank.text_rank import text_rank


def summarize(sentences, n_sents=None, dampening=0.85, convergence=0.0001, niter=200, quiet=True, seed=None, sim=overlap, GraphType=AdjacencyMatrix):
    graph = sentence_graph(sentences, sim, GraphType)
    if n_sents is None:
        n_sents = len(sentences) // 3
    sentences = text_rank(graph, convergence, niter, quiet)[:n_sents]
    return [s[0] for s in sentences]
