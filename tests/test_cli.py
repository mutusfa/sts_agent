"""Tests for the CLI — written before implementation (TDD).

All tests invoke `sts_agent.cli.run(argv)` directly so no subprocess is needed.
"""

from __future__ import annotations

import logging
import sys

import pytest

import sts_agent.cli as cli


# ---------------------------------------------------------------------------
# Basic summary output
# ---------------------------------------------------------------------------

def test_cli_tree_vs_cultist_prints_summary(capsys: pytest.CaptureFixture) -> None:
    """CLI invocation prints a one-line summary with expected fields."""
    cli.run(["battle", "--encounter", "cultist", "--seed", "0", "--agent", "random"])
    out = capsys.readouterr().out
    assert "agent=random" in out
    assert "encounter=cultist" in out
    assert "seed=0" in out
    assert "damage=" in out
    assert "result=" in out


@pytest.mark.slow
def test_cli_tree_vs_cultist_result_won(capsys: pytest.CaptureFixture) -> None:
    """TreeSearch vs 12-HP Cultist should win with 0 damage."""
    cli.run(["battle", "--encounter", "cultist", "--seed", "0"])
    out = capsys.readouterr().out
    assert "result=won" in out


def test_cli_all_single_enemy_encounters_run(capsys: pytest.CaptureFixture) -> None:
    """CLI must accept all single-enemy encounters without error."""
    for encounter in ("cultist", "jaw_worm", "acid_slime_m"):
        cli.run(["battle", "--encounter", encounter, "--seed", "0", "--agent", "random"])
        out = capsys.readouterr().out
        assert f"encounter={encounter}" in out, f"Summary missing encounter={encounter}"
        assert "result=" in out


def test_cli_all_multi_enemy_encounters_run(capsys: pytest.CaptureFixture) -> None:
    """CLI must accept all multi-enemy encounters without error.

    Uses --agent random because the tree search state space grows too large for
    encounters like gremlin_gang (4 enemies) to complete in a reasonable time.
    """
    for encounter in ("small_slimes", "two_louses", "gremlin_gang"):
        try:
            cli.run(["battle", "--encounter", encounter, "--seed", "0", "--agent", "random"])
        except SystemExit as e:
            assert e.code == 1, f"Unexpected exit code {e.code} for {encounter}"
        out = capsys.readouterr().out
        assert f"encounter={encounter}" in out, f"Summary missing encounter={encounter}"
        assert "result=" in out


@pytest.mark.slow
def test_cli_tree_vs_jaw_worm_completes(capsys: pytest.CaptureFixture) -> None:
    """Tree search completes a full jaw_worm combat (~18k nodes/turn, ~9 turns)."""
    cli.run(["battle", "--encounter", "jaw_worm", "--seed", "0"])
    out = capsys.readouterr().out
    assert "encounter=jaw_worm" in out
    assert "result=" in out


def test_cli_supports_random_agent(capsys: pytest.CaptureFixture) -> None:
    """--agent random runs and emits a summary line."""
    cli.run(["battle", "--encounter", "cultist", "--seed", "0", "--agent", "random"])
    out = capsys.readouterr().out
    assert "agent=random" in out
    assert "damage=" in out
    assert "result=" in out


def test_cli_encounters_match_module() -> None:
    """_ENCOUNTERS must exactly mirror the public functions in sts_env.combat.encounters."""
    import inspect
    from sts_env.combat import encounters

    module_fns = {name for name, _ in inspect.getmembers(encounters, inspect.isfunction)}
    assert set(cli._ENCOUNTERS.keys()) == module_fns


def test_cli_unknown_encounter_exits_nonzero() -> None:
    """--encounter Bogus must exit non-zero (SystemExit with code 2)."""
    with pytest.raises(SystemExit) as exc_info:
        cli.run(["battle", "--encounter", "Bogus"])
    assert exc_info.value.code != 0


def test_cli_unknown_encounter_suggests_closest(capsys: pytest.CaptureFixture) -> None:
    """Invalid --encounter should print 'Did you mean' suggestions to stderr."""
    with pytest.raises(SystemExit) as exc_info:
        cli.run(["battle", "--encounter", "gremlin_nog"])
    assert exc_info.value.code != 0
    err = capsys.readouterr().err
    assert "Did you mean" in err
    assert "gremlin_nob" in err  # closest match to "gremlin_nog"


def test_cli_exit_code_zero_on_win(capsys: pytest.CaptureFixture) -> None:
    """Exit code 0 when player wins (MCTS vs 12-HP cultist should reliably win)."""
    try:
        cli.run(["battle", "--encounter", "cultist", "--seed", "0",
                 "--agent", "mcts", "--simulations", "50"])
    except SystemExit as e:
        pytest.fail(f"Unexpected SystemExit({e.code}) on a winning run")


# ---------------------------------------------------------------------------
# Verbose / debug logging
# ---------------------------------------------------------------------------

def test_cli_verbose_emits_info_logs(capsys: pytest.CaptureFixture, caplog: pytest.LogCaptureFixture) -> None:
    """-v enables INFO logging from sts_agent.battle."""
    with caplog.at_level(logging.INFO, logger="sts_agent.battle"):
        cli.run(["battle", "--encounter", "cultist", "--seed", "0", "--agent", "random", "-v"])
    levels = {r.levelno for r in caplog.records}
    assert logging.INFO in levels, "Expected INFO records with -v"


def test_cli_very_verbose_emits_debug_logs(capsys: pytest.CaptureFixture, caplog: pytest.LogCaptureFixture) -> None:
    """-vv enables DEBUG logging from sts_agent.battle."""
    with caplog.at_level(logging.DEBUG, logger="sts_agent.battle"):
        cli.run(["battle", "--encounter", "cultist", "--seed", "0", "--agent", "random", "-vv"])
    levels = {r.levelno for r in caplog.records}
    assert logging.DEBUG in levels, "Expected DEBUG records with -vv"


# ---------------------------------------------------------------------------
# Node budget
# ---------------------------------------------------------------------------

def test_cli_budget_exceeded_exits_with_code_2(capsys: pytest.CaptureFixture) -> None:
    """--max-nodes 1 should exit cleanly with code 2, not raise a traceback."""
    with pytest.raises(SystemExit) as exc_info:
        cli.run(["battle", "--encounter", "cultist", "--seed", "0", "--max-nodes", "1"])
    assert exc_info.value.code == 2


def test_cli_budget_exceeded_prints_error(capsys: pytest.CaptureFixture) -> None:
    """--max-nodes 1 should print a friendly error message to stderr."""
    with pytest.raises(SystemExit):
        cli.run(["battle", "--encounter", "cultist", "--seed", "0", "--max-nodes", "1"])
    err = capsys.readouterr().err
    assert "budget" in err.lower() or "node" in err.lower(), (
        f"Expected budget/node mention in stderr. Got: {err!r}"
    )


# ---------------------------------------------------------------------------
# MCTS agent
# ---------------------------------------------------------------------------


def test_cli_mcts_agent_prints_summary(capsys: pytest.CaptureFixture) -> None:
    """--agent mcts completes and the summary line contains PV stats."""
    cli.run(["battle", "--encounter", "cultist", "--seed", "0",
             "--agent", "mcts", "--simulations", "50"])
    out = capsys.readouterr().out
    assert "agent=mcts" in out
    assert "encounter=cultist" in out
    assert "damage=" in out
    assert "result=" in out
    assert "dmg=" in out
    assert "pv_depth=" in out


def test_cli_non_mcts_agent_has_no_mcts_stats(capsys: pytest.CaptureFixture) -> None:
    """random agent summary must NOT contain MCTS-only PV stats."""
    cli.run(["battle", "--encounter", "cultist", "--seed", "0", "--agent", "random"])
    out = capsys.readouterr().out
    assert "dmg=" not in out
    assert "pv_depth=" not in out


# ---------------------------------------------------------------------------
# Potions
# ---------------------------------------------------------------------------


def test_cli_potions_accepted_and_run_completes(capsys: pytest.CaptureFixture) -> None:
    """--potions BlockPotion is accepted and the battle runs to completion."""
    cli.run([
        "battle", "--encounter", "cultist", "--seed", "0",
        "--agent", "mcts", "--simulations", "50",
        "--potions", "BlockPotion",
    ])
    out = capsys.readouterr().out
    assert "agent=mcts" in out
    assert "encounter=cultist" in out
    assert "damage=" in out
    assert "result=" in out


def test_cli_potions_empty_list_is_noop(capsys: pytest.CaptureFixture) -> None:
    """Omitting --potions runs the same as an empty list (no change to combat)."""
    cli.run(["battle", "--encounter", "cultist", "--seed", "0", "--agent", "random"])
    out = capsys.readouterr().out
    assert "result=" in out


# ---------------------------------------------------------------------------
# Hermes-style runner tests (run_hermes.py)
# ---------------------------------------------------------------------------


def test_hermes_cli_accepts_agent_sim(capsys: pytest.CaptureFixture) -> None:
    """--agent sim should parse successfully."""
    import run_hermes
    from unittest.mock import patch, MagicMock

    mock_result = MagicMock(
        victory=True,
        floors_cleared=8,
        total_floors=8,
        final_hp=70,
        max_hp=80,
        damage_taken_total=10,
        max_hp_gained_total=0,
        damage_per_floor=[1, 2, 3, 4, 5, 6, 7, 8],
        encounter_types=["easy", "easy", "hard", "rest", "elite", "hard", "rest", "boss"],
        cards_added=["Anger", "Bash", "IronWave"],
        potions_gained=["BlockPotion"]
    )

    with patch("run_hermes.run_act1", return_value=mock_result):
        run_hermes.run(["--agent", "sim", "--seed", "42"])
    out = capsys.readouterr().out
    assert "agent=sim" in out


def test_hermes_cli_accepts_agent_llm(capsys: pytest.CaptureFixture) -> None:
    """--agent llm should parse successfully."""
    import run_hermes
    from unittest.mock import patch, MagicMock

    mock_result = MagicMock(
        victory=True,
        floors_cleared=8,
        total_floors=8,
        final_hp=70,
        max_hp=80,
        damage_taken_total=10,
        max_hp_gained_total=0,
        damage_per_floor=[1, 2, 3, 4, 5, 6, 7, 8],
        encounter_types=["easy", "easy", "hard", "rest", "elite", "hard", "rest", "boss"],
        cards_added=["Anger", "Bash", "IronWave"],
        potions_gained=["BlockPotion"]
    )

    with patch("run_hermes.run_act1", return_value=mock_result):
        run_hermes.run(["--agent", "llm", "--seed", "42"])
    out = capsys.readouterr().out
    assert "agent=llm" in out


def test_hermes_cli_accepts_agent_none(capsys: pytest.CaptureFixture) -> None:
    """--agent none should parse successfully."""
    import run_hermes
    from unittest.mock import patch, MagicMock

    mock_result = MagicMock(
        victory=True,
        floors_cleared=8,
        total_floors=8,
        final_hp=70,
        max_hp=80,
        damage_taken_total=10,
        max_hp_gained_total=0,
        damage_per_floor=[1, 2, 3, 4, 5, 6, 7, 8],
        encounter_types=["easy", "easy", "hard", "rest", "elite", "hard", "rest", "boss"],
        cards_added=["Anger", "Bash", "IronWave"],
        potions_gained=["BlockPotion"]
    )

    with patch("run_hermes.run_act1", return_value=mock_result):
        run_hermes.run(["--agent", "none", "--seed", "42"])
    out = capsys.readouterr().out
    assert "agent=none" in out


def test_hermes_cli_invalid_agent_fails(capsys: pytest.CaptureFixture) -> None:
    """--agent invalid should exit with non-zero code."""
    import run_hermes
    with pytest.raises(SystemExit):
        run_hermes.run(["--agent", "invalid", "--seed", "42"])


def test_hermes_cli_sim_agent_constructs_sim_strategy_agent() -> None:
    """--agent sim should construct SimStrategyAgent and pass it to run_act1."""
    import run_hermes
    from unittest.mock import patch, MagicMock

    mock_run_act1 = MagicMock(return_value=MagicMock(
        victory=False,
        floors_cleared=0,
        total_floors=8,
        final_hp=80,
        max_hp=80,
        damage_taken_total=0,
        max_hp_gained_total=0,
        damage_per_floor=[],
        encounter_types=[],
        cards_added=[],
        potions_gained=[]
    ))

    with patch("run_hermes.run_act1", mock_run_act1):
        run_hermes.run(["--agent", "sim", "--seed", "42"])

    # Verify run_act1 was called with a SimStrategyAgent
    call_args = mock_run_act1.call_args
    strategy_agent = call_args.kwargs.get("strategy_agent")
    from sts_agent.strategy import SimStrategyAgent
    assert isinstance(strategy_agent, SimStrategyAgent)


def test_hermes_cli_llm_agent_constructs_llm_strategy_agent() -> None:
    """--agent llm should construct StrategyAgent and pass it to run_act1."""
    import run_hermes
    from unittest.mock import patch, MagicMock

    mock_run_act1 = MagicMock(return_value=MagicMock(
        victory=False,
        floors_cleared=0,
        total_floors=8,
        final_hp=80,
        max_hp=80,
        damage_taken_total=0,
        max_hp_gained_total=0,
        damage_per_floor=[],
        encounter_types=[],
        cards_added=[],
        potions_gained=[]
    ))

    with patch("run_hermes.run_act1", mock_run_act1):
        run_hermes.run(["--agent", "llm", "--seed", "42"])

    # Verify run_act1 was called with a StrategyAgent
    call_args = mock_run_act1.call_args
    strategy_agent = call_args.kwargs.get("strategy_agent")
    from sts_agent.strategy import StrategyAgent
    assert isinstance(strategy_agent, StrategyAgent)


def test_hermes_cli_none_agent_passes_none_to_run_act1() -> None:
    """--agent none should pass None to run_act1."""
    import run_hermes
    from unittest.mock import patch, MagicMock

    mock_run_act1 = MagicMock(return_value=MagicMock(
        victory=False,
        floors_cleared=0,
        total_floors=8,
        final_hp=80,
        max_hp=80,
        damage_taken_total=0,
        max_hp_gained_total=0,
        damage_per_floor=[],
        encounter_types=[],
        cards_added=[],
        potions_gained=[]
    ))

    with patch("run_hermes.run_act1", mock_run_act1):
        run_hermes.run(["--agent", "none", "--seed", "42"])

    # Verify run_act1 was called with None
    call_args = mock_run_act1.call_args
    strategy_agent = call_args.kwargs.get("strategy_agent")
    assert strategy_agent is None


def test_hermes_cli_writes_progress_file(capsys: pytest.CaptureFixture, tmp_path: pytest.TempPathFactory) -> None:
    """Script should write progress markers to /tmp/sts_hermes_progress.txt."""
    import run_hermes
    from unittest.mock import patch, MagicMock

    mock_run_act1 = MagicMock(return_value=MagicMock(
        victory=False,
        floors_cleared=0,
        total_floors=8,
        final_hp=80,
        max_hp=80,
        damage_taken_total=0,
        max_hp_gained_total=0,
        damage_per_floor=[],
        encounter_types=[],
        cards_added=[],
        potions_gained=[]
    ))

    with patch("run_hermes.run_act1", mock_run_act1):
        run_hermes.run(["--agent", "none", "--seed", "42"])

    # Check progress file was written
    import os
    progress_file = "/tmp/sts_hermes_progress.txt"
    assert os.path.exists(progress_file), "Progress file should exist"

    content = open(progress_file).read()
    assert "RUN_START" in content
    assert "RUN_END" in content


def test_hermes_cli_writes_result_file(capsys: pytest.CaptureFixture) -> None:
    """Script should write results to /tmp/sts_hermes_result.txt."""
    import run_hermes
    from unittest.mock import patch, MagicMock

    mock_result = MagicMock(
        victory=True,
        floors_cleared=8,
        total_floors=8,
        final_hp=70,
        max_hp=80,
        damage_taken_total=10,
        max_hp_gained_total=0,
        damage_per_floor=[1, 2, 3, 4, 5, 6, 7, 8],
        encounter_types=["easy", "easy", "hard", "rest", "elite", "hard", "rest", "boss"],
        cards_added=["Anger", "Bash", "IronWave"],
        potions_gained=["BlockPotion"]
    )
    mock_run_act1 = MagicMock(return_value=mock_result)

    with patch("run_hermes.run_act1", mock_run_act1):
        run_hermes.run(["--agent", "none", "--seed", "42"])

    # Check result file was written with expected fields
    import os
    result_file = "/tmp/sts_hermes_result.txt"
    assert os.path.exists(result_file), "Result file should exist"

    content = open(result_file).read()
    assert "victory=True" in content
    assert "floors_cleared=8" in content
    assert "final_hp=70" in content
    assert "total_damage=10" in content
    assert "cards_added=" in content


