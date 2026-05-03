"""Run a full Act 1 with LLM strategy agent + MLflow tracing."""

import logging
import sys
import time


class _FlushingFileHandler(logging.FileHandler):
    """FileHandler that flushes after every record so logs are never lost."""

    def emit(self, record: logging.LogRecord) -> None:
        super().emit(record)
        self.flush()


class _FlushingStreamHandler(logging.StreamHandler):
    """StreamHandler that flushes after every record."""

    def emit(self, record: logging.LogRecord) -> None:
        super().emit(record)
        self.flush()


# Imports first — some libraries (MLflow, DSPy) call logging.basicConfig()
# during import, which can replace handlers.  We set up logging AFTER imports
# and use force=True to win the handler fight.
from sts_agent.tracing import setup_tracing
from sts_agent.strategy import StrategyAgent
from sts_agent.run import run_act1
from sts_agent.battle.mcts import MCTSPlanner

# Configure logging LAST so nothing can clobber our handlers.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[
        _FlushingFileHandler("/tmp/sts_llm_run.log", mode="w"),
        _FlushingStreamHandler(sys.stderr),
    ],
    force=True,
)

# MLflow tracing → sqlite in project dir
setup_tracing()

# LLM strategy agent with conservative sim budget for faster tool calls
agent = StrategyAgent(
    timeout_seconds=300,       # 5 min per card pick
    max_simulations=100,       # MCTS rollouts per tool call
    max_nodes=1000,            # MCTS node budget per tool call
)

planner = MCTSPlanner(
    simulations=1000,
    max_nodes=5000,
)

print("=== Starting LLM Act 1 run (seed=42) ===", file=sys.stderr)
logging.info("=== Starting LLM Act 1 run (seed=42) ===")
t0 = time.time()

result = run_act1(planner, seed=42, strategy_agent=agent)

elapsed = time.time() - t0
print(f"\n=== Run complete in {elapsed:.1f}s ===", file=sys.stderr)

# Summary
summary = (
    f"victory={result.victory}\n"
    f"floors_cleared={result.floors_cleared}/{result.total_floors}\n"
    f"final_hp={result.final_hp}/{result.max_hp}\n"
    f"damage_per_floor={result.damage_per_floor}\n"
    f"total_damage={result.damage_taken_total}\n"
    f"max_hp_gained={result.max_hp_gained_total}\n"
    f"cards_added={result.cards_added}\n"
    f"potions_gained={result.potions_gained}\n"
    f"encounter_types={result.encounter_types}\n"
)

print(summary)
with open("/tmp/sts_llm_result.txt", "w") as f:
    f.write(summary)
