import json
import argparse
from text_rank import DEMO_LOC
from text_rank.graph import sentence_graph, keyword_graph, AdjacencyList
from text_rank.text_rank import text_rank


def main():
    parser = argparse.ArgumentParser("Text Rank demo")
    parser.add_argument("--iters", "-i", type=int, default=40)
    parser.add_argument("--sents", "-s", type=int, default=3)
    parser.add_argument("--words", "-w", type=int, default=5)
    args = parser.parse_args()

    sentences = json.load(open(DEMO_LOC / "Automatic_Summarization-sents.json"))
    G = sentence_graph(sentences)
    sents1 = text_rank(G, args.iters)
    print("Adjacency Matrix based Text Rank for extractive summarization")
    for i, x in enumerate(sents1[: args.sents]):
        print(f" {i + 1}. {x[0]}")

    G = sentence_graph(sentences, GraphType=AdjacencyList)
    sents2 = text_rank(G, args.iters)
    print()
    print("Adjacency List based Text Rank for extractive summarization")
    for i, x in enumerate(sents2[: args.sents]):
        print(f" {i + 1}. {x[0]}")

    tokens = json.load(open(DEMO_LOC / "Automatic_Summarization-tokens.json"))
    G = keyword_graph(tokens)
    keywords1 = text_rank(G, args.iters)
    print()
    print("Adjacency Matrix based Text Rank for key-word extraction")
    for i, x in enumerate(keywords1[: args.words]):
        print(f" {i + 1}. {x[0]}")

    G2 = keyword_graph(tokens, GraphType=AdjacencyList)
    keywords2 = text_rank(G2, args.iters)
    print()
    print("Adjacency List based Text Rank for key-word extraction")
    for i, x in enumerate(keywords2[: args.words]):
        print(f" {i + 1}. {x[0]}")


if __name__ == "__main__":
    main()
