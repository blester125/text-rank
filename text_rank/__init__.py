__version__ = "0.3.0"

from text_rank.graph import sentence_graph, keyword_graph, AdjacencyList, AdjacencyMatrix
from text_rank.text_rank import text_rank
from text_rank.keywords import keywords
from text_rank.summarize import summarize

from pathlib import Path

DEMO_LOC = Path(__file__).parent / "data"
