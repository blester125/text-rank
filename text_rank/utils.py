import re
from math import log
from typing import Any, Dict, List
from collections import defaultdict


ASCII = re.compile(r"[^a-z0-9]")
POS = re.compile(r"^[NJ]")


def filter_pos(token: Dict[str, str]) -> bool:
    """Should this token be included based on POS tags.

    This says yes to tokens that have pos tags of Adjective, Cardinal Number, any kind of
    noun and any kind of verb.

    :param token: A dict representation of the token with a pos tag in the `pos` key

    :returns: True if the token should be used, False otherwise
    """
    if not POS.match(token["pos"]) and token["pos"] != "ADJ" and token["pos"] != "CD":
        return False
    return True


def cooccurrence(s1: Any, s2: Any, **kwargs) -> int:
    """A similarity function used for keyword graph generation.

    Because the default keyword graphs use cooccurrence instead of weighting the edges
    so we always return 1.

    :returns: 1
    """
    return 1


def overlap(s1: str, s2: str, **kwargs) -> float:
    """A similarity function used for sentences.

    This is the similarity function defined in (Mihalcea and Tarau 2004)

    ```math
        Similarity(S_i, S_j) = \frac{|{w_k|w_k \in S_i & \w_k \in S_j}|}{\log(|S_i|) + \log(|S_j|)}
    ```

    Note:
        There is a weird edge case in this function when both sets have a single word. This results
        in a denominator of 0.0 which causes an error. I am currently forcing the denominator to
        `0.9256056207721526` in this case. This makes since because then both of them are a single
        word, there are two cases, one where the overlap is `0`, this results in a weight of `0` which
        makes sense. The other case is when they match. This results in a `1.0803737332167307` which
        was calculated by fitting this function and finding the value it predicts for `1`. See
        scripts/find-sim-of-one.py for this fitting.

    :param s1: The first item
    :param s2: The second item
    :returns: The similarity between s1 and s2
    """
    s1 = set(s1.split())
    s2 = set(s2.split())
    intersection = len(s1 & s2)
    norm = log(len(s1)) + log(len(s2))
    norm = 0.9256056207721526 if norm == 0.0 else norm
    return intersection / norm


def norm_sentence(sent: str) -> str:
    """Normalize a sentence by lower casing and removing punctuation from each word.

    :param sent: The white spaced joined sentence.
    :returns: The sentence joined with whitespace.
    """
    tokens = sent.split()
    tokens = [norm_token(token) for token in tokens]
    return " ".join(tokens)


def norm_token(token: str) -> str:
    """Normalize a token by lowercasing and removing non-alphanumeric characters.

    :param token: The word to be normalized
    :returns: The word lowercased and punctuation removed.
    """
    token = token.lower()
    return ASCII.sub("", token)


def build_vocab(tokens: List[str]) -> Dict[str, int]:
    """Build a vocab mapping tokens to indices.

    Produced indices are contiguous and a token has the index of where it first appeared in the tokens list.

    :param tokens: The list of tokens to map
    :returns: A mapping from token to index
    """
    vocab = defaultdict(lambda: len(vocab))
    for token in tokens:
        vocab[token]
    return {k: i for k, i in vocab.items()}
