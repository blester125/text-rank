from text_rank.utils import overlap, norm_sentence
from text_rank.graph import sentence_graph, AdjacencyMatrix
from text_rank.text_rank import text_rank


def summarize(sentences, n_sents=None, dampening=0.85, convergence=0.0001, niter=200, quiet=True, seed=None, sim=overlap, norm=norm_sentence, GraphType=AdjacencyMatrix):
    graph, offsets = sentence_graph(sentences, sim, norm, GraphType)
    if n_sents is None:
        n_sents = len(sentences) // 3
    sentences = text_rank(graph, convergence, niter, quiet)[:n_sents]
    # Look up the normed sentence in offsets, use the string in the original sentences
    # Also sort the sentences by the order they show up in the original
    return [s[0] for s in sentences]
