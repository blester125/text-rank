import re
from math import log
from collections import defaultdict


ASCII = re.compile(r"[^a-z0-9]")
POS = re.compile(r"^[NJ]")


def filter_pos(token):
    if not POS.match(token["pos"]) and token["pos"] != "ADJ" and token["pos"] != "CD":
        return False
    return True


def overlap(s1, s2):
    s1 = set(s1.split())
    s2 = set(s2.split())
    intersection = len(s1 & s2)
    norm = log(len(s1)) + log(len(s2))
    norm = 1 if norm == 0.0 else norm
    return intersection / norm


def norm_sentence(sent):
    tokens = sent.split()
    tokens = [norm_token(token) for token in tokens]
    return " ".join(tokens)


def norm_token(token):
    token = token.lower()
    return ASCII.sub('', token)


def build_vocab(tokens):
    vocab = defaultdict(lambda: len(vocab))
    for token in tokens:
        vocab[token]
    return {k: i for k, i in vocab.items()}
