#!/usr/bin/env python3
"""Parallel smoke test — run many Act 1 sim-agent runs and summarize data collection."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import mlflow

from sts_agent.battle.mcts import MCTSPlanner
from sts_agent.run import run_act1
from sts_agent.strategy import SimStrategyAgent
from sts_agent.tracing import setup_tracing

DEFAULT_SIMS = 1000
DEFAULT_NODES = 1000


@dataclass(frozen=True)
class SimBudgets:
    probe_nodes: int
    probe_sims: int
    combat_nodes: int
    combat_sims: int


def resolve_sim_budgets(
    *,
    sim_nodes: int | None,
    sim_sims: int | None,
    probe_nodes: int | None,
    probe_sims: int | None,
    combat_nodes: int | None,
    combat_sims: int | None,
) -> SimBudgets:
    """Resolve probe vs combat MCTS budgets.

    ``--sim-sims`` / ``--sim-nodes`` set both layers unless overridden by
    ``--probe-*`` or ``--combat-*``.
    """
    shared_nodes = sim_nodes if sim_nodes is not None else DEFAULT_NODES
    shared_sims = sim_sims if sim_sims is not None else DEFAULT_SIMS
    return SimBudgets(
        probe_nodes=probe_nodes if probe_nodes is not None else shared_nodes,
        probe_sims=probe_sims if probe_sims is not None else shared_sims,
        combat_nodes=combat_nodes if combat_nodes is not None else shared_nodes,
        combat_sims=combat_sims if combat_sims is not None else shared_sims,
    )


@dataclass
class SmokeResult:
    seed: int
    victory: bool
    floors_cleared: int
    total_floors: int
    error: str | None = None
    probe_records: int = 0
    record_type_counts: dict[str, int] | None = None
    run_id: str | None = None


def _run_one(
    seed: int,
    *,
    budgets: SimBudgets,
    max_encounters: int,
    decision_timeout: int,
    probe_dir: Path | None,
    experiment: str,
    use_map: bool,
) -> SmokeResult:
    setup_tracing(experiment_name=experiment)
    jsonl = probe_dir / f"seed_{seed}.jsonl" if probe_dir else None
    agent = SimStrategyAgent(
        max_encounters_to_sim=max_encounters,
        sim_nodes=budgets.probe_nodes,
        sim_sims=budgets.probe_sims,
        timeout_seconds=decision_timeout,
        seed=seed,
        probe_jsonl=jsonl,
    )
    planner = MCTSPlanner(
        simulations=budgets.combat_sims,
        max_nodes=budgets.combat_nodes,
    )
    try:
        with mlflow.start_run(run_name=f"smoke seed={seed}"):
            mlflow.log_params(
                {
                    "agent": "sim",
                    "seed": str(seed),
                    "smoke_test": "true",
                    "probe_sims": str(budgets.probe_sims),
                    "probe_nodes": str(budgets.probe_nodes),
                    "combat_sims": str(budgets.combat_sims),
                    "combat_nodes": str(budgets.combat_nodes),
                }
            )
            result = run_act1(
                planner, seed=seed, strategy_agent=agent, use_map=use_map
            )
            mlflow.log_metrics(
                {
                    "victory": float(result.victory),
                    "floors_cleared": float(result.floors_cleared),
                }
            )
            active = mlflow.active_run()
            run_id = active.info.run_id if active else None
        return SmokeResult(
            seed=seed,
            victory=result.victory,
            floors_cleared=result.floors_cleared,
            total_floors=result.total_floors,
            probe_records=len(agent.probe_collector.records),
            record_type_counts=agent.probe_collector.count_by_record_type(),
            run_id=run_id,
        )
    except Exception as exc:
        return SmokeResult(
            seed=seed,
            victory=False,
            floors_cleared=0,
            total_floors=0,
            error=f"{type(exc).__name__}: {exc}",
        )


def _run_one_packed(packed: tuple) -> SmokeResult:
    (
        seed,
        budgets,
        max_encounters,
        decision_timeout,
        probe_dir,
        experiment,
        use_map,
    ) = packed
    return _run_one(
        seed,
        budgets=budgets,
        max_encounters=max_encounters,
        decision_timeout=decision_timeout,
        probe_dir=probe_dir,
        experiment=experiment,
        use_map=use_map,
    )


def _count_pick_card_traces(experiment_name: str) -> int:
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        return 0
    client = mlflow.MlflowClient()
    traces = client.search_traces(locations=[exp.experiment_id], max_results=500)
    count = 0
    for trace in traces:
        for span in trace.data.spans:
            if span.name == "pick_card":
                count += 1
    return count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Parallel Act 1 smoke test")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument(
        "--sim-nodes",
        type=int,
        default=10000,
        help="MCTS node budget for probes and combat",
    )
    parser.add_argument(
        "--sim-sims",
        type=int,
        default=1000,
        help="MCTS simulation budget for probes and combat",
    )
    parser.add_argument(
        "--probe-nodes",
        type=int,
        default=None,
        help="Probe node budget (overrides --sim-nodes for strategy layer)",
    )
    parser.add_argument(
        "--probe-sims",
        type=int,
        default=100,
        help="Probe simulation budget (overrides --sim-sims for strategy layer)",
    )
    parser.add_argument(
        "--combat-nodes",
        type=int,
        default=None,
        help="Combat node budget (overrides --sim-nodes for battle planner)",
    )
    parser.add_argument(
        "--combat-sims",
        type=int,
        default=None,
        help="Combat simulation budget (overrides --sim-sims for battle planner)",
    )
    parser.add_argument("--max-encounters", type=int, default=3)
    parser.add_argument(
        "--decision-timeout",
        type=int,
        default=600,
        help="Wall-clock seconds per strategy decision before heuristic fallback",
    )
    parser.add_argument(
        "--probe-dir",
        type=Path,
        default=Path("smoke_probes"),
        help="Directory for per-run probe JSONL (default: smoke_probes/)",
    )
    parser.add_argument(
        "--linear",
        action="store_true",
        help="Use fixed 8-floor linear scenario instead of branching map.",
    )
    parser.add_argument(
        "--experiment",
        default="sts-agent-smoke",
        help="MLflow experiment name",
    )
    args = parser.parse_args(argv)
    use_map = not args.linear
    budgets = resolve_sim_budgets(
        sim_nodes=args.sim_nodes,
        sim_sims=args.sim_sims,
        probe_nodes=args.probe_nodes,
        probe_sims=args.probe_sims,
        combat_nodes=args.combat_nodes,
        combat_sims=args.combat_sims,
    )

    setup_tracing(experiment_name=args.experiment)
    args.probe_dir.mkdir(parents=True, exist_ok=True)

    seeds = [args.seed_start + i for i in range(args.count)]
    t0 = time.time()

    tasks = [
        (
            seed,
            budgets,
            args.max_encounters,
            args.decision_timeout,
            args.probe_dir,
            args.experiment,
            use_map,
        )
        for seed in seeds
    ]

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=args.workers) as pool:
        results = pool.map(_run_one_packed, tasks)

    elapsed = time.time() - t0
    completed = [r for r in results if r.error is None]
    errors = [r for r in results if r.error is not None]
    victories = sum(1 for r in completed if r.victory)
    total_probes = sum(r.probe_records for r in completed)
    pick_traces = _count_pick_card_traces(args.experiment)
    merged_counts: Counter[str] = Counter()
    for r in completed:
        if r.record_type_counts:
            merged_counts.update(r.record_type_counts)

    summary = {
        "total_runs": args.count,
        "completed": len(completed),
        "errors": len(errors),
        "victory_rate": victories / len(completed) if completed else 0.0,
        "victories": victories,
        "total_probe_records": total_probes,
        "record_type_counts": dict(merged_counts),
        "pick_card_traces": pick_traces,
        "elapsed_seconds": round(elapsed, 1),
        "error_samples": [r.error for r in errors[:5]],
        "probe_nodes": budgets.probe_nodes,
        "probe_sims": budgets.probe_sims,
        "combat_nodes": budgets.combat_nodes,
        "combat_sims": budgets.combat_sims,
    }

    print(json.dumps(summary, indent=2))
    out_path = args.probe_dir / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Summary written to {out_path}")

    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
