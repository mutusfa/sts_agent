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

from sts_env.combat import Combat, encounters

from .battle import MCTSPlanner, RandomAgent, TreeSearchPlanner, run_agent, run_planner
from .battle.tree_search import SearchBudgetExceeded

_ENCOUNTERS = {
    name: fn
    for name, fn in inspect.getmembers(encounters, inspect.isfunction)
}

_SUBCOMMANDS = ("battle",)

_LOG_LEVELS = {
    0: logging.WARNING,
    1: logging.INFO,
    2: logging.DEBUG,
}


def _configure_logging(level: int) -> None:
    """Set the log level for the sts_agent package and add a console handler if needed.

    Attaches to the ``sts_agent`` package logger rather than the root logger so
    that pytest's ``caplog`` fixture (which injects into root) is never disturbed.
    """
    pkg = logging.getLogger("sts_agent")
    pkg.setLevel(level)
    if not pkg.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter("%(levelname)s %(name)s: %(message)s")
        )
        pkg.addHandler(handler)


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
        help="Increase verbosity (-v=INFO, -vv=DEBUG).",
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

    # Configure logging level based on -v count (capped at 2).
    verbosity = min(args.verbose, 2)
    level = _LOG_LEVELS[verbosity]
    _configure_logging(level)

    if args.subcommand == "battle":
        _run_battle(args)


def _run_battle(args: argparse.Namespace) -> None:
    combat = _ENCOUNTERS[args.encounter](args.seed, player_hp=args.player_hp)
    if args.potions:
        # Rebuild with potions using the same deck/enemies resolved by the factory.
        combat = Combat(
            deck=combat._deck,  # type: ignore[union-attr]
            enemies=combat._enemy_names,  # type: ignore[union-attr]
            seed=args.seed,
            player_hp=args.player_hp,
            potions=args.potions,
        )

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
        stats = mcts_planner.last_stats
        summary += (
            f" mean={stats['mean']:.1f} std={stats['std']:.1f} max={stats['max']:.1f}"
        )

    print(summary)

    if obs.player_dead:
        raise SystemExit(1)


if __name__ == "__main__":
    run()
