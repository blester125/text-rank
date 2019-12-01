from operator import itemgetter
from text_rank.utils import filter_pos
from text_rank.graph import keyword_graph, AdjacencyMatrix
from text_rank.text_rank import text_rank


def join_keywords(candidates, tokens):
    candidates = set(candidates)
    candidates = [(token, i) for i, token in enumerate(tokens) if token in candidates]
    candidates = sorted(candidates, key=itemgetter(1))
    keywords = []
    current = []
    prev = None
    for kw, i in candidates:
        if not current or i == prev + 1:
            current.append(kw)
        else:
            keywords.append(" ".join(current))
            current = [kw]
        prev = i
    if current:
        keywords.append(" ".join(current))
    return set(keywords)


def keywords(tokens, n_words=None, winsz=2, dampening=0.85, convergence=0.0001, niter=200, quiet=True, seed=None, sim=lambda x, y: 1, filt=filter_pos, GraphType=AdjacencyMatrix):
    graph = keyword_graph(tokens, winsz, sim, filt, GraphType)
    if n_words is None:
        n_words = len(tokens) // 3
    keywords = text_rank(graph, convergence, niter, quiet)[:n_words]
    keywords = join_keywords([kw[0] for kw in keywords], map(lambda x: x['term'], tokens))
    return keywords
