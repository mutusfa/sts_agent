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

import logging
from typing import Protocol

from sts_env.combat import Action, Combat, Observation
from sts_env.combat.state import ActionType

log = logging.getLogger(__name__)


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


# ---------------------------------------------------------------------------
# Format helpers
# ---------------------------------------------------------------------------

def _fmt_action(action: Action, hand: list[str]) -> str:
    """Compact human-readable action string."""
    if action.action_type == ActionType.END_TURN:
        return "END_TURN"
    card = hand[action.hand_index] if action.hand_index < len(hand) else "?"
    return f"{card}→E{action.target_index}"


def _fmt_obs(obs: Observation) -> str:
    """Single-line observation snapshot."""
    def _fmt_enemy_intent(enemy: object) -> str:
        intent_type = getattr(enemy, "intent_type", "NONE")
        base = getattr(enemy, "intent_damage", None)
        hits = getattr(enemy, "intent_hits", None)
        block_gain = getattr(enemy, "intent_block_gain", None)
        # Forward-compatible with env shape that exposes "base" and
        # "active-effects-applied" intent damage values separately.
        effective = getattr(enemy, "intent_damage_effective", None)

        if base is None and effective is None and hits is None and block_gain is None:
            return f"intent:{intent_type}"

        base_str = "?" if base is None else str(base)
        eff_str = base_str if effective is None else str(effective)
        hits_str = "?" if hits is None else str(hits)
        block_str = "?" if block_gain is None else str(block_gain)
        return (
            f"intent:{intent_type}(base:{base_str} eff:{eff_str}"
            f" hits:{hits_str} blk+:{block_str})"
        )

    enemies = " ".join(
        f"{e.name}({e.hp}/{e.max_hp} blk:{e.block} {_fmt_enemy_intent(e)})"
        for e in obs.enemies
    )
    hand = ",".join(obs.hand)
    return (
        f"T={obs.turn} hp={obs.player_hp}/{obs.player_max_hp} "
        f"blk={obs.player_block} nrg={obs.energy} hand=[{hand}] | {enemies}"
    )


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------

def run_agent(agent: BattleAgent, combat: Combat) -> int:
    """Drive *agent* through one full combat starting from reset().

    Returns damage_taken at termination.
    """
    obs = combat.reset()

    enemy_names = ", ".join(e.name for e in obs.enemies)
    log.info("START agent=%s enemies=[%s] player_hp=%d/%d hand=%s",
             type(agent).__name__, enemy_names,
             obs.player_hp, obs.player_max_hp, obs.hand)

    while not obs.done:
        actions = combat.valid_actions()
        action = agent.act(obs, actions)
        log.debug("T=%d action=%s | before: %s",
                  obs.turn, _fmt_action(action, obs.hand), _fmt_obs(obs))
        obs, _reward, _info = combat.step(action)

    outcome = "dead" if obs.player_dead else "won"
    log.info("END outcome=%s damage=%d turns=%d", outcome, combat.damage_taken, obs.turn)
    return combat.damage_taken


def run_planner(planner: BattlePlanner, combat: Combat) -> int:
    """Drive *planner* through one full combat starting from reset().

    Returns damage_taken at termination.
    """
    obs = combat.reset()

    enemy_names = ", ".join(e.name for e in obs.enemies)
    log.info("START planner=%s enemies=[%s] player_hp=%d/%d hand=%s",
             type(planner).__name__, enemy_names,
             obs.player_hp, obs.player_max_hp, obs.hand)

    while not obs.done:
        action = planner.act(combat)
        obs_before = combat.observe()
        log.debug("T=%d action=%s | before: %s",
                  obs_before.turn, _fmt_action(action, obs_before.hand), _fmt_obs(obs_before))
        obs, _reward, _info = combat.step(action)

    outcome = "dead" if obs.player_dead else "won"
    log.info("END outcome=%s damage=%d turns=%d", outcome, combat.damage_taken, obs.turn)
    return combat.damage_taken
