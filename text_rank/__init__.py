__version__ = "0.3.1"

# This is a new name to reference the text_rank as a modules instead of the function.
# This is used for mocking when testing while still allowing for `from text_rank import text_rank`
# by and end user
import text_rank.text_rank as text_rank_module
from text_rank.text_rank import text_rank
from text_rank.keywords import keywords
from text_rank.summarize import summarize
from text_rank.graph import sentence_graph, keyword_graph, AdjacencyList, AdjacencyMatrix

from pathlib import Path

DEMO_LOC = Path(__file__).parent / "data"
