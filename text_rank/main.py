import json
import argparse
from text_rank import DEMO_LOC
from text_rank.graph import sentence_graph, keyword_graph
from text_rank.graph import sentence_graph_list, keyword_graph_list
from text_rank.text_rank import text_rank, text_rank_list

parser = argparse.ArgumentParser("Text Rank")
parser.add_argument("--iters", "-i", type=int, default=40)
parser.add_argument("--sents", "-s", type=int, default=3)
parser.add_argument("--words", "-w", type=int, default=5)
args = parser.parse_args()


sentences = json.load(open(DEMO_LOC / "Automatic_Summarization-sents.json"))
V, g = sentence_graph(sentences)
sents1 = text_rank(V, g, args.iters)
print([' '.join(x[0]) for x in sents1[:args.sents]])

V = sentence_graph_list(sentences)
sents2 = text_rank_list(V, args.iters)
print([' '.join(x[0].value) for x in sents2[:args.sents]])


tokens = json.load(open(DEMO_LOC / "Automatic_Summarization-tokens.json"))
V, g = keyword_graph(tokens)
keywords1 = text_rank(V, g, args.iters)
print([x[0] for x in keywords1[:args.words]])

V = keyword_graph_list(tokens)
keywords2 = text_rank_list(V, args.iters)
print([x[0].value for x in keywords2[:args.words]])
