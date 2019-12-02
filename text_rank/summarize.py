from typing import Optional, Callable, Type
from text_rank.utils import overlap, norm_sentence
from text_rank.text_rank import text_rank, ConvergenceType
from text_rank.graph import sentence_graph, AdjacencyMatrix, Graph


def summarize(
    sentences,
    n_sents: Optional[int] = None,
    dampening: float = 0.85,
    convergence: float = 0.0001,
    convergence_type: ConvergenceType = ConvergenceType.ALL,
    niter: int = 200,
    seed: Optional[int] = None,
    sim: Callable[[str, str], float] = overlap,
    norm: Callable[[str], str] = norm_sentence,
    GraphType: Type[Graph] = AdjacencyMatrix,
):
    graph, offsets = sentence_graph(sentences, sim, norm, GraphType)
    if n_sents is None:
        n_sents = len(sentences) // 3
    selected = text_rank(
        graph, dampening=dampening, convergence=convergence, convergence_type=convergence_type, niter=niter, seed=seed
    )[:n_sents]
    indices = [offsets[s[0]][0] for s in selected]
    return [sentences[i] for i in sorted(indices)]
