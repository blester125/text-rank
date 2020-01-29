from itertools import chain
from operator import itemgetter
from typing import Optional, Callable, Type, List, Dict, Set
from text_rank.utils import filter_pos, norm_token
from text_rank.text_rank import text_rank, ConvergenceType
from text_rank.graph import keyword_graph, AdjacencyMatrix, Graph


def join_adjacent_keywords(candidates: List[str], offsets: Dict[str, List[str]]) -> Set[str]:
    """Combine keywords that are adjacent in the source text into a single keyword.

    :param candidates: The keywords selected
    :param offsets: The offsets of the keywords into the original text
    :returns: The keywords created by joining adjacent ones
    """
    candidates = list(chain(*([(candidate, o) for o in offsets[candidate]] for candidate in candidates)))
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


def keywords(
    tokens: List[Dict[str, str]],
    nwords: Optional[int] = None,
    winsz: int = 2,
    damping: float = 0.85,
    convergence: float = 0.0001,
    convergence_type: ConvergenceType = ConvergenceType.ANY,
    niter: int = 200,
    seed: Optional[int] = None,
    sim: Callable[..., float] = lambda x, y, **kwargs: 1,
    norm: Callable[[str], str] = norm_token,
    filt: Callable[[Dict[str, str]], bool] = filter_pos,
    GraphType: Type[Graph] = AdjacencyMatrix,
) -> Set[str]:
    """Find keywords.

    :param tokens: The tokens of the text to pull keywords from
    :param nwords: The number of keywords to extract from the text. If `None` it uses the number
        of tokens in the passage divided by 3. This number isn't exact, after finding keywords it will
        join adjacent ones so it might be possible to get fewer keywords back.
    :param winsz: The size of a window around each word that it can connect to. This window is calculated
        in the non-filtered token list.
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
                context=context,
                raw_s1=raw_s1,
                raw_s2=raw_s2,
                raw_context=raw_context,
                s1_idx=s1_idx,
                s2_idx=s2_idx,
            ) -> float:
        Where normed_s1/2 is the normalized string of two keywords, context is all of the normalized tokens,
        raw_s1/2/context are the prenormalized versions and s1/2_idx are the indices of the keywords in the
        original sentence. This should allow complex similarity functions. The context is the whole list of
        tokens, not just the window around one.
    :param norm: A function the returns a normalized version of the input string. Default implementation
        lowercases string and removes non alpha-numeric characters.
        This is used to unify similar vertices, i.e. `Hurricane` and `hurricane` should be the same vertex
        and will be with normalization.
    :param filt: A function that will filter tokens based on pos tags if the inputs are Dicts.
    :param GraphType: The Graph class to use.

    :returns: The keywords from the passage
    """
    graph, offsets = keyword_graph(tokens, winsz, sim, norm, filt, GraphType)
    if nwords is None:
        nwords = len(tokens) // 3
    keywords = text_rank(
        graph, damping=damping, convergence=convergence, convergence_type=convergence_type, niter=niter, seed=seed
    )[:nwords]
    keywords = join_adjacent_keywords([kw[0] for kw in keywords], offsets)
    return keywords
