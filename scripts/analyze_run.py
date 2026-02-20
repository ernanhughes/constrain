from constrain.data.memory import Memory
from constrain.analysis.high_dimensional_analysis import HighDimensionalAnalyzer
from constrain.config import get_config

memory = Memory(get_config().db_url)
run_id = "run_18cdc06e"

results = HighDimensionalAnalyzer.run(memory, run_id)
