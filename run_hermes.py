
import logging, sys, time

# Log to file for Hermes to read
fh = logging.FileHandler('/tmp/sts_hermes_run.log', mode='w')
fh.setFormatter(logging.Formatter('%(message)s'))
logging.basicConfig(level=logging.INFO, handlers=[fh], force=True)

# Also write progress to a separate marker file
progress = open('/tmp/sts_hermes_progress.txt', 'w')

def mark(msg):
    progress.write(f"[{time.time():.0f}] {msg}\n")
    progress.flush()

from sts_agent.strategy import SimStrategyAgent
from sts_agent.run import run_act1
from sts_agent.battle.mcts import MCTSPlanner

# Lower sim budget for faster run
agent = SimStrategyAgent(max_encounters_to_sim=3, sim_nodes=1000, sim_sims=1000)
planner = MCTSPlanner()

mark("RUN_START")
result = run_act1(planner, seed=42, strategy_agent=agent)
mark("RUN_END")

# Write final results to a separate file
with open('/tmp/sts_hermes_result.txt', 'w') as f:
    f.write(f"victory={result.victory}\n")
    f.write(f"floors_cleared={result.floors_cleared}/{result.total_floors}\n")
    f.write(f"final_hp={result.final_hp}/{result.max_hp}\n")
    f.write(f"damage_per_floor={result.damage_per_floor}\n")
    f.write(f"total_damage={result.damage_taken_total}\n")
    f.write(f"max_hp_gained={result.max_hp_gained_total}\n")
    f.write(f"cards_added={result.cards_added}\n")
    f.write(f"potions_gained={result.potions_gained}\n")
    f.write(f"encounter_types={result.encounter_types}\n")

mark("DONE")
progress.close()
print("Run complete! Results in /tmp/sts_hermes_result.txt")
