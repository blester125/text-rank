import math
import random
import string
from itertools import chain
from unittest.mock import patch, call
from text_rank.utils import filter_pos, overlap, build_vocab, cooccurrence, norm_token, norm_sentence

# Import random string utility
from utils import rand_str


TRIALS = 100


def test_filter_pos_adj():
    token = {"pos": "ADJ"}
    assert filter_pos(token) is True


def test_filter_pos_cd():
    token = {"pos": "CD"}
    assert filter_pos(token) is True


def test_filter_pos_noun():
    token = {"pos": "NNP"}
    assert filter_pos(token) is True


def test_filter_pos_j():
    token = {"pos": "JJ"}
    assert filter_pos(token) is True


def test_filter_pos_other():
    def test():
        token = {"pos": rand_str()}
        assert filter_pos(token) is False

    for _ in range(TRIALS):
        test()


def test_overlap():
    def test():
        intersection = random.randint(0, 10)
        len_1 = random.randint(intersection, intersection + random.randint(1, 5))
        len_2 = random.randint(intersection, intersection + random.randint(1, 5))
        len_1 = max(1, len_1)
        len_2 = max(1 if len_1 != 1 else 2, len_2)
        union = math.log(len_1) + math.log(len_2)
        union = 1 if union == 0.0 else union

        gold_score = intersection / union

        set1 = set()
        set2 = set()

        while len(set1) < intersection:
            t = rand_str()
            set1.add(t)
            set2.add(t)

        while len(set1) < len_1:
            t = rand_str()
            if t not in set1:
                set1.add(t)

        while len(set2) < len_2:
            t = rand_str()
            if t not in set1 and t not in set2:
                set2.add(t)

        ex1 = list(chain(*[[t] * random.randint(1, 5) for t in set1]))
        ex2 = list(chain(*[[t] * random.randint(1, 5) for t in set2]))

        random.shuffle(ex1)
        random.shuffle(ex2)

        ex1 = " ".join(ex1)
        ex2 = " ".join(ex2)

        score = overlap(ex1, ex2)
        assert math.isclose(score, gold_score)

    for _ in range(TRIALS):
        test()


def test_overlap_match_length_one():
    gold = 1.0803737332167307
    s1 = s2 = rand_str()
    assert math.isclose(overlap(s1, s2), gold)


def test_overlap_mismatch_length_one():
    str1 = str2 = rand_str()
    while str2 == str1:
        str2 = rand_str()
    assert math.isclose(overlap(str1, str2), 0.0)


def test_build_vocab():
    def test():
        gold_vocab = {}
        i = 0
        tokens = []
        for _ in range(random.randint(10, 20)):
            text = rand_str()
            if text in gold_vocab:
                continue
            gold_vocab[text] = i
            i += 1
            tokens.append(text)
            if random.choice([True, False]):
                tokens.append(random.choice(tokens))
        vocab = build_vocab(tokens)
        assert vocab == gold_vocab

    for _ in range(TRIALS):
        test()


def test_cooccurrence():
    def test():
        s1 = rand_str()
        s2 = rand_str()
        kwargs = {rand_str(): rand_str() for _ in range(random.randint(0, 10))}
        res = cooccurrence(s1, s2, **kwargs)
        assert res == 1

    for _ in range(TRIALS):
        test()


def test_norm_token():
    def test():
        gold = rand_str()
        token = []
        for char in gold:
            if random.choice([True, False]):
                token.append(char.upper())
            else:
                token.append(char)
            while random.random() < 0.3:
                token.append(random.choice(string.punctuation))
        token = "".join(token)
        res = norm_token(token)
        assert res == gold

    for _ in range(TRIALS):
        test()


def test_norm_sentence():
    sentence = [rand_str() for _ in range(random.randint(1, 10))]
    with patch("text_rank.utils.norm_token") as norm_patch:
        norm_patch.side_effect = sentence
        res = norm_sentence(" ".join(sentence))
        for token in sentence:
            assert call(token) in norm_patch.call_args_list
        assert res == " ".join(sentence)
