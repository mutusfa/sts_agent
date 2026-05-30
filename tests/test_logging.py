"""Tests for debug logging in battle runners and planners (TDD — fails until instrumented)."""

from __future__ import annotations

import logging

import pytest

from sts_env.combat import Combat
from sts_env.combat.encounters import cultist
from sts_env.combat.player_state import PlayerState
from sts_env.combat.card import Card
from sts_env.combat.state import EnemyObs, Observation

from sts_agent.battle import RandomAgent, TreeSearchPlanner, run_agent, run_planner
from sts_agent.battle.base import _fmt_obs


class _FirstActionPlanner:
    """Minimal BattlePlanner stub: always picks the first legal action.

    Fast stand-in for tests that exercise run_planner() logging, not planner
    quality.  Guaranteed to terminate (valid_actions always includes END_TURN).
    """

    def act(self, combat: Combat) -> object:
        return combat.valid_actions()[0]


def _make_obs(
    *,
    player_powers: dict | None = None,
    enemy_powers: dict | None = None,
) -> Observation:
    """Build a minimal Observation with controllable power dicts."""
    enemy = EnemyObs(
        name="Cultist",
        hp=50,
        max_hp=50,
        block=0,
        powers=enemy_powers or {"strength": 0, "vulnerable": 0, "weak": 0, "curl_up": 0, "angry": 0},
        intent_type="ATTACK",
        intent_damage=6,
        intent_damage_effective=6,
        intent_hits=1,
        intent_block_gain=0,
    )
    return Observation(
        player_hp=80,
        player_max_hp=80,
        player_block=0,
        player_powers=player_powers or {"strength": 0, "vulnerable": 0, "weak": 0, "frail": 0},
        energy=3,
        hand=[Card("Strike")],
        draw_pile={},
        discard_pile={},
        exhaust_pile={},
        enemies=[enemy],
        done=False,
        player_dead=False,
        turn=0,
        potions=[],
        max_potion_slots=3,
    )


@pytest.fixture()
def cultist_combat() -> Combat:
    return cultist(0, PlayerState.ironclad_starter())


def test_run_planner_emits_info_and_debug(cultist_combat: Combat, caplog: pytest.LogCaptureFixture) -> None:
    """run_planner must emit at least one INFO and at least one DEBUG record."""
    with caplog.at_level(logging.DEBUG, logger="sts_agent.battle"):
        run_planner(_FirstActionPlanner(), cultist_combat)

    levels = {r.levelno for r in caplog.records}
    assert logging.INFO in levels, "Expected at least one INFO log from run_planner"
    assert logging.DEBUG in levels, "Expected at least one DEBUG log from run_planner"


def test_run_agent_emits_info(cultist_combat: Combat, caplog: pytest.LogCaptureFixture) -> None:
    """run_agent must emit at least one INFO record."""
    with caplog.at_level(logging.INFO, logger="sts_agent.battle"):
        run_agent(RandomAgent(seed=0), cultist_combat)

    levels = {r.levelno for r in caplog.records}
    assert logging.INFO in levels, "Expected at least one INFO log from run_agent"


def test_run_planner_info_mentions_enemy(cultist_combat: Combat, caplog: pytest.LogCaptureFixture) -> None:
    """The INFO start message must mention the enemy name."""
    with caplog.at_level(logging.INFO, logger="sts_agent.battle"):
        run_planner(_FirstActionPlanner(), cultist_combat)

    info_msgs = [r.message for r in caplog.records if r.levelno == logging.INFO]
    assert any("Cultist" in m for m in info_msgs), (
        f"No INFO message mentions 'Cultist'. Got: {info_msgs}"
    )


def test_run_planner_info_mentions_outcome(cultist_combat: Combat, caplog: pytest.LogCaptureFixture) -> None:
    """The INFO end message must mention 'won' or 'dead'."""
    with caplog.at_level(logging.INFO, logger="sts_agent.battle"):
        run_planner(_FirstActionPlanner(), cultist_combat)

    info_msgs = [r.message for r in caplog.records if r.levelno == logging.INFO]
    assert any("won" in m or "dead" in m for m in info_msgs), (
        f"No INFO message mentions outcome. Got: {info_msgs}"
    )


@pytest.mark.slow
def test_tree_planner_emits_debug_with_action(cultist_combat: Combat, caplog: pytest.LogCaptureFixture) -> None:
    """TreeSearchPlanner.act must emit a DEBUG record mentioning 'nodes'."""
    planner = TreeSearchPlanner()

    with caplog.at_level(logging.DEBUG, logger="sts_agent.battle"):
        planner.act(cultist_combat)

    debug_msgs = [r.message for r in caplog.records if r.levelno == logging.DEBUG]
    assert any("node" in m.lower() for m in debug_msgs), (
        f"No DEBUG message from TreeSearchPlanner mentions 'node'. Got: {debug_msgs}"
    )


def test_random_agent_emits_debug(cultist_combat: Combat, caplog: pytest.LogCaptureFixture) -> None:
    """RandomAgent.act must emit at least one DEBUG record."""
    obs = cultist_combat.observe()
    agent = RandomAgent(seed=0)

    with caplog.at_level(logging.DEBUG, logger="sts_agent.battle"):
        agent.act(obs, cultist_combat.valid_actions())

    debug_msgs = [r.message for r in caplog.records if r.levelno == logging.DEBUG]
    assert debug_msgs, "Expected at least one DEBUG log from RandomAgent.act"


def test_debug_logs_contain_per_step_info(cultist_combat: Combat, caplog: pytest.LogCaptureFixture) -> None:
    """Per-step DEBUG messages should mention 'T=' and 'hp='."""
    with caplog.at_level(logging.DEBUG, logger="sts_agent.battle"):
        run_planner(_FirstActionPlanner(), cultist_combat)

    debug_msgs = [r.message for r in caplog.records if r.levelno == logging.DEBUG]
    assert any("T=" in m and "hp=" in m for m in debug_msgs), (
        f"No DEBUG message contains per-step info. Got first 5: {debug_msgs[:5]}"
    )


def test_debug_logs_include_enemy_intent_numbers(
    cultist_combat: Combat, caplog: pytest.LogCaptureFixture
) -> None:
    """Per-step DEBUG messages should include enemy intent base/effective damage."""
    with caplog.at_level(logging.DEBUG, logger="sts_agent.battle"):
        run_planner(_FirstActionPlanner(), cultist_combat)

    debug_msgs = [r.message for r in caplog.records if r.levelno == logging.DEBUG]
    assert any("intent:ATTACK(" in m for m in debug_msgs), (
        f"No DEBUG message contains intent damage details. Got first 5: {debug_msgs[:5]}"
    )


def test_run_planner_emits_turn_dividers(cultist_combat: Combat, caplog: pytest.LogCaptureFixture) -> None:
    """run_planner must emit '--- Turn N ---' debug lines at the start of each turn."""
    with caplog.at_level(logging.DEBUG, logger="sts_agent.battle"):
        run_planner(_FirstActionPlanner(), cultist_combat)

    debug_msgs = [r.message for r in caplog.records if r.levelno == logging.DEBUG]
    assert any("--- Turn 0 ---" in m for m in debug_msgs), (
        f"Expected '--- Turn 0 ---' in debug output. Got: {debug_msgs[:10]}"
    )
    assert any("--- Turn 1 ---" in m for m in debug_msgs), (
        f"Expected '--- Turn 1 ---' in debug output (multi-turn fight). Got: {debug_msgs[:10]}"
    )


def test_run_agent_emits_turn_dividers(cultist_combat: Combat, caplog: pytest.LogCaptureFixture) -> None:
    """run_agent must emit '--- Turn N ---' debug lines at the start of each turn."""
    with caplog.at_level(logging.DEBUG, logger="sts_agent.battle"):
        run_agent(RandomAgent(seed=0), cultist_combat)

    debug_msgs = [r.message for r in caplog.records if r.levelno == logging.DEBUG]
    assert any("--- Turn 0 ---" in m for m in debug_msgs), (
        f"Expected '--- Turn 0 ---' in debug output. Got: {debug_msgs[:10]}"
    )


def test_fmt_obs_shows_nonzero_player_powers() -> None:
    """_fmt_obs must include nonzero player powers so effects are visible in logs."""
    obs = _make_obs(player_powers={"strength": 0, "vulnerable": 0, "weak": 2, "frail": 0})
    result = _fmt_obs(obs)
    assert "weak:2" in result, f"Expected 'weak:2' in obs string, got: {result}"


def test_fmt_obs_omits_zero_player_powers() -> None:
    """_fmt_obs must not clutter the line with zero-value player powers."""
    obs = _make_obs(player_powers={"strength": 0, "vulnerable": 0, "weak": 0, "frail": 0})
    result = _fmt_obs(obs)
    assert "weak" not in result, f"'weak' should be absent when zero, got: {result}"
    assert "vulnerable" not in result, f"'vulnerable' should be absent when zero, got: {result}"


def test_fmt_obs_shows_multiple_player_powers() -> None:
    """_fmt_obs includes all nonzero player powers (e.g. weak + frail)."""
    obs = _make_obs(player_powers={"strength": 2, "vulnerable": 1, "weak": 0, "frail": 3})
    result = _fmt_obs(obs)
    assert "str:2" in result, f"Expected 'str:2', got: {result}"
    assert "vuln:1" in result, f"Expected 'vuln:1', got: {result}"
    assert "frail:3" in result, f"Expected 'frail:3', got: {result}"


def test_fmt_obs_shows_nonzero_enemy_powers() -> None:
    """_fmt_obs must include nonzero enemy powers (e.g. vulnerable stacks)."""
    obs = _make_obs(enemy_powers={"strength": 0, "vulnerable": 2, "weak": 0, "curl_up": 0, "angry": 0})
    result = _fmt_obs(obs)
    assert "vuln:2" in result, f"Expected 'vuln:2' in obs string, got: {result}"


def test_fmt_obs_omits_zero_enemy_powers() -> None:
    """_fmt_obs must not show zero-value enemy powers."""
    obs = _make_obs(enemy_powers={"strength": 0, "vulnerable": 0, "weak": 0, "curl_up": 0, "angry": 0})
    result = _fmt_obs(obs)
    # Powers block should not appear for the enemy
    assert "vuln:0" not in result, f"'vuln:0' should be absent, got: {result}"
    assert "angry:0" not in result, f"'angry:0' should be absent, got: {result}"
