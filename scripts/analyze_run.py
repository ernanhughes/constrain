from constrain.data.memory import Memory
from constrain.analysis.stage3.escalation_signal_discovery import HighDimensionalAnalyzer
from constrain.config import get_config

memory = Memory(get_config().db_url)
run_id = "run_18cdc06e"

results = HighDimensionalAnalyzer.run(memory, run_id)
