import sys
import random
from unittest.mock import patch
import pytest
from text_rank.text_rank import (
    sum_edges,
    accumulate_score,
    text_rank_init_list,
    text_rank_init_matrix,
    text_rank_update_list,
    text_rank_update_matrix,
    text_rank_output_list,
    text_rank_output_matrix,
)


TRIALS = 100


def test_sum_edges():
    def test():
        edges = {}
        gold_total = random.randint(0, 100)
        total = gold_total
        i = 0
        while total > 0:
            weight = random.randint(1, 5)
            weight = min(weight, total)
            edges[i] = weight
            i += 1
            total -= weight
        score = sum_edges(edges)
        assert score == gold_total

    for _ in range(TRIALS):
        test()


def test_accumulate_score():
    pass

def test_init_list():
    pass

def test_init_matrix():
    pass

def test_update_list():
    pass

def test_update_matrix():
    pass

def test_output_list():
    pass

def test_output_matrix():
    pass
