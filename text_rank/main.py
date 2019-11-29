import json
import argparse
from text_rank import DEMO_LOC
from text_rank.graph import sentence_graph, keyword_graph, AdjacencyList
from text_rank.text_rank import text_rank


def main():
    parser = argparse.ArgumentParser("Text Rank demo")
    parser.add_argument("--sents", "-s", type=int, default=3)
    parser.add_argument("--words", "-w", type=int, default=5)
    parser.add_argument("--iters", "-i", type=int, default=40)
    parser.add_argument("--convergence", "-c", type=float, default=0.0)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    sentences = json.load(open(DEMO_LOC / "Automatic_Summarization-sents.json"))

    G = sentence_graph(sentences)
    sents = text_rank(G, convergence=args.convergence, niter=args.iters, quiet=not args.verbose)
    print("Adjacency Matrix based Text Rank for extractive summarization")
    for i, x in enumerate(sents[: args.sents]):
        print(f" {i + 1}. {x[0]}")

    G = sentence_graph(sentences, GraphType=AdjacencyList)
    sents = text_rank(G, convergence=args.convergence, niter=args.iters, quiet=not args.verbose)
    print()
    print("Adjacency List based Text Rank for extractive summarization")
    for i, x in enumerate(sents[: args.sents]):
        print(f" {i + 1}. {x[0]}")

    tokens = json.load(open(DEMO_LOC / "Automatic_Summarization-tokens.json"))

    G = keyword_graph(tokens)
    keywords = text_rank(G, convergence=args.convergence, niter=args.iters, quiet=not args.verbose)
    print()
    print("Adjacency Matrix based Text Rank for key-word extraction")
    for i, x in enumerate(keywords[: args.words]):
        print(f" {i + 1}. {x[0]}")

    G = keyword_graph(tokens, GraphType=AdjacencyList)
    keywords = text_rank(G, convergence=args.convergence, niter=args.iters, quiet=not args.verbose)
    print()
    print("Adjacency List based Text Rank for key-word extraction")
    for i, x in enumerate(keywords[: args.words]):
        print(f" {i + 1}. {x[0]}")


if __name__ == "__main__":
    main()
