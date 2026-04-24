"""Core protocols and utilities shared by all battle agents and planners.

Terminology
-----------
BattleAgent  – obs-in policy; receives only the read-only Observation and the
               list of legal actions returned by valid_actions().  Cannot look
               ahead or clone the environment.
BattlePlanner – receives the full Combat object so it can clone() and simulate
               future branches before committing to an action.

Runner helpers
--------------
run_agent   – drives a BattleAgent through a full combat (reset → done).
run_planner – same for a BattlePlanner.
Both return damage_taken at the end of combat.

Terminal scoring
----------------
terminal_score returns a penalty that is continuous even through death:
  alive  → damage_taken
  dead   → damage_taken * 2

The doubling ensures that any surviving branch (score ≤ start_hp) ranks
above any dying branch (score ≥ 2), while still ordering two dying branches
by how much overkill was taken.
"""

from __future__ import annotations

from typing import Protocol

from sts_env.combat import Action, Combat, Observation


class BattleAgent(Protocol):
    """Pure obs-in policy.  Cannot clone or look ahead."""

    def act(self, obs: Observation, valid_actions: list[Action]) -> Action:
        ...


class BattlePlanner(Protocol):
    """Full-access policy.  May clone the combat and simulate branches."""

    def act(self, combat: Combat) -> Action:
        ...


def terminal_score(combat: Combat) -> int:
    """Score a terminal combat state.  Lower is better (less HP lost)."""
    obs = combat.observe()
    multiplier = 2 if obs.player_dead else 1
    return combat.damage_taken * multiplier


def run_agent(agent: BattleAgent, combat: Combat) -> int:
    """Drive *agent* through one full combat starting from reset().

    Returns damage_taken at termination.
    """
    obs = combat.reset()
    while not obs.done:
        actions = combat.valid_actions()
        action = agent.act(obs, actions)
        obs, _reward, _info = combat.step(action)
    return combat.damage_taken


def run_planner(planner: BattlePlanner, combat: Combat) -> int:
    """Drive *planner* through one full combat starting from reset().

    Returns damage_taken at termination.
    """
    obs = combat.reset()
    while not obs.done:
        action = planner.act(combat)
        obs, _reward, _info = combat.step(action)
    return combat.damage_taken
