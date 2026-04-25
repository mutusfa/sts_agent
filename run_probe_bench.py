"""Benchmark: probe-based SimStrategyAgent vs full-sim baseline."""
import logging, sys, time

fh = logging.FileHandler('/tmp/sts_probe_run.log', mode='w')
fh.setFormatter(logging.Formatter('%(message)s'))
logging.basicConfig(level=logging.INFO, handlers=[fh], force=True)

progress = open('/tmp/sts_probe_progress.txt', 'w')

def mark(msg):
    progress.write(f"[{time.time():.0f}] {msg}\n")
    progress.flush()

from sts_agent.strategy import SimStrategyAgent
from sts_agent.run import run_act1
from sts_agent.battle.mcts import MCTSPlanner

# Probe-based agent (fast: single act() per encounter per option)
agent = SimStrategyAgent(max_encounters_to_sim=3, sim_nodes=1000, sim_sims=1000)
planner = MCTSPlanner()

mark("RUN_START")
t0 = time.time()
result = run_act1(planner, seed=42, strategy_agent=agent)
elapsed = time.time() - t0
mark("RUN_END")

with open('/tmp/sts_probe_result.txt', 'w') as f:
    f.write(f"victory={result.victory}\n")
    f.write(f"floors_cleared={result.floors_cleared}/{result.total_floors}\n")
    f.write(f"final_hp={result.final_hp}/{result.max_hp}\n")
    f.write(f"damage_per_floor={result.damage_per_floor}\n")
    f.write(f"total_damage={result.damage_taken_total}\n")
    f.write(f"cards_added={result.cards_added}\n")
    f.write(f"potions_gained={result.potions_gained}\n")
    f.write(f"elapsed_seconds={elapsed:.1f}\n")

mark(f"DONE elapsed={elapsed:.1f}s")
progress.close()
print(f"Run complete in {elapsed:.1f}s! Results in /tmp/sts_probe_result.txt")
