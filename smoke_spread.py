"""Quick smoke test: one card reward evaluation with spread output."""
import logging, sys
logging.basicConfig(level=logging.INFO, format='%(message)s', stream=sys.stdout)

from sts_agent.strategy import SimStrategyAgent
from sts_env.run.character import Character
from sts_env.run.scenarios import act1_encounters

char = Character.ironclad()
encounters = act1_encounters(42)

agent = SimStrategyAgent(max_encounters_to_sim=3, sim_nodes=1000, sim_sims=1000)

# Evaluate the first card reward (after floor 1)
upcoming = encounters[1:]  # floors 2, 3, 4
choices = ["Flex", "TrueStrike", "SecondWind"]

print("Evaluating card reward:", choices)
print("Upcoming encounters:", [(t, i) for t, i in upcoming[:3]])
print()

pick = agent.pick_card(char, choices, upcoming, seed=42000)
print(f"\nPicked: {pick}")
