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
from sts_env.combat.card import Card
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

def _card_id(c) -> str:
    """Extract card_id from an observation hand item (dict or Card)."""
    return c["card_id"] if isinstance(c, dict) else c.card_id

def _fmt_action(action: Action, hand: list) -> str:
    """Compact human-readable action string."""
    if action.action_type == ActionType.END_TURN:
        return "END_TURN"
    if action.action_type == ActionType.CHOOSE_CARD:
        return f"CHOOSE:{action.choice_index}"
    if action.action_type == ActionType.SKIP_CHOICE:
        return "SKIP_CHOICE"
    card_obj = hand[action.hand_index] if action.hand_index < len(hand) else None
    card = card_obj["card_id"] if isinstance(card_obj, dict) else (card_obj.card_id if card_obj else "?")
    return f"{card}→E{action.target_index}"


_PLAYER_POWER_ABBREV: dict[str, str] = {
    "strength": "str",
    "vulnerable": "vuln",
    "weak": "weak",
    "frail": "frail",
}
_ENEMY_POWER_ABBREV: dict[str, str] = {
    "strength": "str",
    "vulnerable": "vuln",
    "weak": "weak",
    "curl_up": "curl",
    "angry": "angry",
}


def _fmt_powers(powers: dict, abbrev: dict[str, str]) -> str:
    """Return a compact string of nonzero power stacks, e.g. 'weak:2 vuln:1'."""
    parts = [
        f"{abbrev.get(k, k)}:{v}"
        for k, v in powers.items()
        if v and k in abbrev
    ]
    return " ".join(parts)


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

    def _fmt_enemy(e: object) -> str:
        hp = getattr(e, "hp", "?")
        max_hp = getattr(e, "max_hp", "?")
        blk = getattr(e, "block", 0)
        name = getattr(e, "name", "?")
        powers_str = _fmt_powers(getattr(e, "powers", {}), _ENEMY_POWER_ABBREV)
        powers_part = f" [{powers_str}]" if powers_str else ""
        return f"{name}({hp}/{max_hp} blk:{blk}{powers_part} {_fmt_enemy_intent(e)})"

    enemies = " ".join(_fmt_enemy(e) for e in obs.enemies)
    hand = ",".join(_card_id(c) for c in obs.hand)
    player_powers_str = _fmt_powers(obs.player_powers, _PLAYER_POWER_ABBREV)
    player_powers_part = f" [{player_powers_str}]" if player_powers_str else ""
    return (
        f"T={obs.turn} hp={obs.player_hp}/{obs.player_max_hp} "
        f"blk={obs.player_block} nrg={obs.energy}{player_powers_part} hand=[{hand}] | {enemies}"
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
    hand_str = ",".join(_card_id(c) for c in obs.hand)
    log.info("START agent=%s enemies=[%s] player_hp=%d/%d hand=%s",
             type(agent).__name__, enemy_names,
             obs.player_hp, obs.player_max_hp, hand_str)

    prev_turn = -1
    while not obs.done:
        if obs.turn != prev_turn:
            log.debug("--- Turn %d ---", obs.turn)
            prev_turn = obs.turn
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
    hand_str = ",".join(_card_id(c) for c in obs.hand)
    log.info("START planner=%s enemies=[%s] player_hp=%d/%d hand=%s",
             type(planner).__name__, enemy_names,
             obs.player_hp, obs.player_max_hp, hand_str)

    prev_turn = -1
    while not obs.done:
        curr_turn = combat.observe().turn
        if curr_turn != prev_turn:
            log.debug("--- Turn %d ---", curr_turn)
            prev_turn = curr_turn
        action = planner.act(combat)
        obs_before = combat.observe()
        log.debug("T=%d action=%s | before: %s",
                  obs_before.turn, _fmt_action(action, obs_before.hand), _fmt_obs(obs_before))
        obs, _reward, _info = combat.step(action)

    outcome = "dead" if obs.player_dead else "won"
    log.info("END outcome=%s damage=%d turns=%d", outcome, combat.damage_taken, obs.turn)
    return combat.damage_taken
