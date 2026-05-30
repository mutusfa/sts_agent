"""Command-line interface for sts-agent.

Usage
-----
    sts-agent [battle] --encounter cultist --seed 0 --agent tree -vv
    python -m sts_agent.cli --encounter jaw_worm -vv      # 'battle' is the default subcommand

Subcommands
-----------
battle  Run a single combat and print a one-line summary.
        Can be omitted; it is the default when the first argument starts with '-'.

Encounters
----------
All public functions in sts_env.combat.encounters are available automatically.

Verbosity
---------
(none)  WARNING level — only the summary line and errors.
-v      INFO level  — combat start/end messages.
-vv     DEBUG level — per-step action traces + planner search details.
-vvv    DEBUG level for third-party libraries (mlflow, urllib3, etc.).
        ``sts_agent`` stays at DEBUG from ``-vv``; external libs stay at
        WARNING until ``-vvv``.

Exit codes
----------
0   Player won.
1   Player died.
2   Node budget exceeded or argument error.
"""

from __future__ import annotations

import argparse
import inspect
import logging
import sys

from sts_env.combat import encounters
from sts_env.combat.player_state import PlayerState

from .battle import MCTSPlanner, RandomAgent, TreeSearchPlanner, run_agent, run_planner
from .battle.tree_search import SearchBudgetExceeded
from .logging_config import configure_logging

_ENCOUNTERS = {
    name: fn
    for name, fn in inspect.getmembers(encounters, inspect.isfunction)
}

_SUBCOMMANDS = ("battle",)

def _configure_logging(verbosity: int) -> None:
    """Set log levels for ``sts_agent`` and third-party libraries.

    Attaches handlers to the ``sts_agent`` package logger (and root at ``-vvv``)
    rather than configuring root alone, so pytest's ``caplog`` fixture is not
    disturbed at normal verbosity.
    """
    formatter = logging.Formatter("%(levelname)s %(name)s: %(message)s")
    pkg_handler = logging.StreamHandler(sys.stderr)
    pkg_handler.setFormatter(formatter)

    root_handlers: list[logging.Handler] | None = None
    if verbosity >= 3:
        root_handler = logging.StreamHandler(sys.stderr)
        root_handler.setFormatter(formatter)
        root_handlers = [root_handler]

    configure_logging(
        verbosity,
        handlers=[pkg_handler],
        root_handlers=root_handlers,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sts-agent",
        description="Slay the Spire agent runner.",
    )
    sub = parser.add_subparsers(dest="subcommand", required=True)

    battle = sub.add_parser("battle", help="Run a single combat.")
    battle.add_argument(
        "--encounter",
        required=True,
        metavar="{" + ",".join(_ENCOUNTERS) + "}",
        help="Encounter to run (see sts_env.combat.encounters).",
    )
    battle.add_argument("--seed", type=int, default=0, help="RNG seed (default: 0).")
    battle.add_argument(
        "--agent",
        choices=("random", "tree", "mcts"),
        default="tree",
        help="Agent to use (default: tree).",
    )
    battle.add_argument(
        "--simulations",
        type=int,
        default=1000,
        help="MCTSPlanner simulation budget per act() (mcts agent only, default: 1000).",
    )
    battle.add_argument(
        "--player-hp",
        type=int,
        default=80,
        dest="player_hp",
        help="Starting player HP (default: 80).",
    )
    battle.add_argument(
        "--max-nodes",
        type=int,
        default=None,
        dest="max_nodes",
        help="Node expansion budget (tree and mcts agents; default: unlimited).",
    )
    battle.add_argument(
        "--potions",
        nargs="*",
        default=[],
        metavar="POTION_ID",
        help="Potions to start with (e.g. BlockPotion EnergyPotion).",
    )
    battle.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (-v=INFO, -vv=DEBUG, -vvv=external DEBUG).",
    )

    return parser


def run(argv: list[str] | None = None) -> None:
    """Parse *argv* (defaults to sys.argv[1:]) and execute the requested subcommand.

    If the first positional argument is not a known subcommand, 'battle' is
    prepended automatically so that ``sts-agent --encounter cultist`` works.
    """
    if argv is None:
        argv = sys.argv[1:]
    # Inject the default subcommand when the user omits it.
    if not argv or argv[0] not in _SUBCOMMANDS:
        argv = ["battle", *argv]

    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.subcommand == "battle" and args.encounter not in _ENCOUNTERS:
        from rapidfuzz import process

        matches = process.extract(args.encounter, list(_ENCOUNTERS), limit=3)
        suggestions = ", ".join(m[0] for m in matches if m[1] >= 50)
        msg = f"invalid choice: {args.encounter!r}"
        if suggestions:
            msg += f"\nDid you mean one of: {suggestions}?"
        parser.error(msg)

    _configure_logging(args.verbose)

    if args.subcommand == "battle":
        _run_battle(args)


def _run_battle(args: argparse.Namespace) -> None:
    character = PlayerState(
        player_hp=args.player_hp,
        player_max_hp=args.player_hp,
        potions=list(args.potions),
    )
    combat = _ENCOUNTERS[args.encounter](args.seed, character)

    mcts_planner: MCTSPlanner | None = None
    try:
        if args.agent == "tree":
            planner = TreeSearchPlanner(max_nodes=args.max_nodes)
            damage = run_planner(planner, combat)
        elif args.agent == "mcts":
            mcts_planner = MCTSPlanner(
                simulations=args.simulations,
                max_nodes=args.max_nodes,
                seed=args.seed,
            )
            damage = run_planner(mcts_planner, combat)
        else:
            agent = RandomAgent(seed=args.seed)
            damage = run_agent(agent, combat)
    except SearchBudgetExceeded as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    obs = combat.observe()
    result = "dead" if obs.player_dead else "won"

    summary = (
        f"agent={args.agent} encounter={args.encounter} seed={args.seed} "
        f"damage={damage} turns={obs.turn} result={result}"
    )
    if mcts_planner is not None:
        import math as _math
        stats = mcts_planner.last_stats
        pv_n = stats["pv_n"]
        pv_death_rate = stats["pv_deaths"] / pv_n if pv_n > 0 else float("nan")
        pv_mean = stats["pv_mean"]
        pv_expected_dmg = (
            min(pv_mean, float(obs.player_max_hp))
            if pv_n > 0 and _math.isfinite(pv_mean)
            else float("nan")
        )
        summary += (
            f" dmg={pv_expected_dmg:.0f}±{stats['pv_std']:.0f}"
            f" ({pv_death_rate:.0%} die)"
            f" pv_depth={int(stats['pv_depth'])}"
        )

    print(summary)

    if obs.player_dead:
        raise SystemExit(1)


if __name__ == "__main__":
    run()
