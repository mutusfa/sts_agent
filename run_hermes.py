"""Hermes-style runner with configurable strategy agent via CLI.

Usage
-----
    python run_hermes_cli.py --agent sim --seed 42
    python run_hermes_cli.py --agent llm --seed 42
    python run_hermes_cli.py --agent none --seed 42

This script mimics the behavior of run_hermes.py but allows you to choose
the strategy agent (sim/llm/none) via command-line arguments.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import mlflow

from sts_agent.battle.mcts import MCTSPlanner
from sts_agent.run import run_act1, RunResult
from sts_agent.strategy import SimStrategyAgent, StrategyAgent
from sts_agent.tracing import setup_tracing

# Log to file for Hermes to read
LOG_FILE = Path("/tmp/sts_hermes_run.log")
PROGRESS_FILE = Path("/tmp/sts_hermes_progress.txt")
RESULT_FILE = Path("/tmp/sts_hermes_result.txt")


def _setup_logging() -> None:
    """Configure logging to file."""
    fh = logging.FileHandler(LOG_FILE, mode="w")
    fh.setFormatter(logging.Formatter("%(message)s"))
    logging.basicConfig(level=logging.INFO, handlers=[fh], force=True)


def _open_progress_file():
    """Open the progress file for writing."""
    return open(PROGRESS_FILE, "w")


def _mark(progress, msg: str) -> None:
    """Write a progress marker with timestamp."""
    progress.write(f"[{time.time():.0f}] {msg}\n")
    progress.flush()


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        prog="run_hermes_cli",
        description="Hermes-style runner with configurable strategy agent.",
    )
    parser.add_argument(
        "--agent",
        choices=("sim", "llm", "none"),
        default="sim",
        help="Strategy agent to use (default: sim).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed (default: 42).",
    )
    return parser


def _build_strategy_agent(agent_type: str):
    """Build a strategy agent based on the CLI argument.

    Args:
        agent_type: One of "sim", "llm", or "none".

    Returns:
        SimStrategyAgent for "sim", StrategyAgent for "llm", or None for "none".
    """
    if agent_type == "sim":
        # Lower sim budget for faster run (same as run_hermes.py)
        return SimStrategyAgent(max_encounters_to_sim=3, sim_nodes=1000, sim_sims=1000)
    elif agent_type == "llm":
        return StrategyAgent()
    else:  # agent_type == "none"
        return None


def _write_result(result: RunResult) -> None:
    """Write the final result to the result file."""
    with open(RESULT_FILE, "w") as f:
        f.write(f"victory={result.victory}\n")
        f.write(f"floors_cleared={result.floors_cleared}/{result.total_floors}\n")
        f.write(f"final_hp={result.final_hp}/{result.max_hp}\n")
        f.write(f"damage_per_floor={result.damage_per_floor}\n")
        f.write(f"total_damage={result.damage_taken_total}\n")
        f.write(f"max_hp_gained={result.max_hp_gained_total}\n")
        f.write(f"cards_added={result.cards_added}\n")
        f.write(f"potions_gained={result.potions_gained}\n")
        f.write(f"encounter_types={result.encounter_types}\n")


def _log_run_to_mlflow(result: RunResult, agent_type: str, seed: int) -> None:
    """Log run result metrics and params to the active MLflow run."""
    mlflow.log_params({
        "agent": agent_type,
        "seed": str(seed),
    })
    mlflow.log_metrics({
        "victory": float(result.victory),
        "floors_cleared": float(result.floors_cleared),
        "total_floors": float(result.total_floors),
        "final_hp": float(result.final_hp),
        "max_hp": float(result.max_hp),
        "total_damage": float(result.damage_taken_total),
        "max_hp_gained": float(result.max_hp_gained_total),
    })


def run(argv: list[str] | None = None) -> None:
    """Parse arguments and run the Hermes-style runner.

    Args:
        argv: Command-line arguments (defaults to sys.argv[1:]).
    """
    if argv is None:
        argv = sys.argv[1:]

    parser = _build_parser()
    args = parser.parse_args(argv)

    # Setup logging and tracing
    _setup_logging()
    setup_tracing()
    progress = _open_progress_file()

    # Build the strategy agent based on CLI choice
    strategy_agent = _build_strategy_agent(args.agent)
    planner = MCTSPlanner()

    # Run the act inside an MLflow run
    _mark(progress, "RUN_START")
    with mlflow.start_run(run_name=f"seed={args.seed}"):
        result = run_act1(planner, seed=args.seed, strategy_agent=strategy_agent)
        _log_run_to_mlflow(result, args.agent, args.seed)
    _mark(progress, "RUN_END")

    # Write final results
    _write_result(result)
    _mark(progress, "DONE")
    progress.close()

    # Print summary to stdout
    print(f"agent={args.agent} seed={args.seed} victory={result.victory} "
          f"floors_cleared={result.floors_cleared}/{result.total_floors} "
          f"final_hp={result.final_hp}/{result.max_hp}")
    print(f"Run complete! Results in {RESULT_FILE}")


if __name__ == "__main__":
    run()
