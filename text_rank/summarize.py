from typing import Optional, Callable, Type, List
from text_rank.utils import overlap, norm_sentence
from text_rank.text_rank import text_rank, ConvergenceType
from text_rank.graph import sentence_graph, AdjacencyMatrix, Graph


def summarize(
    sentences: List[str],
    nsents: Optional[int] = None,
    keep_order: bool = True,
    damping: float = 0.85,
    convergence: float = 0.0001,
    convergence_type: ConvergenceType = ConvergenceType.ALL,
    niter: int = 200,
    seed: Optional[int] = None,
    sim: Callable[..., float] = overlap,
    norm: Callable[[str], str] = norm_sentence,
    GraphType: Type[Graph] = AdjacencyMatrix,
) -> List[str]:
    """Summarize text.

    :param sentences: The sentences to summarize.
    :param nsents: The number of sentences to use in the summary, If `None` it uses the number of
        sentences to summarize divided by 3
    :param keep_order: Should summary sentences appear in the same order they appear in the in the
        text, if False sentences or ordered by their text rank score.
    :param damping: A scalar between 0 and 1. Used to simulate randomly jumping from one vertex to another.
    :param convergence: An early stopping criteria, when any or all of the node scores change by less than `convergence`
        we stop updating the graph. Set to `0` to turn off early stopping.
    :param convergence_type: Should we stop when all nodes move less than `convergence` or when a single node does
    :param niter: An upper bound on the number of iterations to run
    :param seed: A reproducability seed to initialization of the node scores.
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

    :returns: A list of sentences summarizing the original text.
    """
    graph, offsets = sentence_graph(sentences, sim, norm, GraphType)
    if nsents is None:
        nsents = len(sentences) // 3
    selected = text_rank(
        graph, damping=damping, convergence=convergence, convergence_type=convergence_type, niter=niter, seed=seed,
    )[:nsents]
    indices = [offsets[s[0]][0] for s in selected]
    if keep_order:
        return [sentences[i] for i in sorted(indices)]
    return [sentences[i] for i in indices]
