# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
just test              # Run all tests (quiet)
just test-v            # Run all tests (verbose)
just run <ARGS>        # Run pytest with custom args

# Single test file or pattern
python -m pytest tests/test_cli.py -v
python -m pytest -k "test_tree_search" -v
python -m pytest --run-slow   # Include @pytest.mark.slow tests
```

Run a single battle via CLI:
```bash
sts-agent --encounter cultist --seed 0 --agent tree
sts-agent --encounter jaw_worm --seed 0 --agent mcts --simulations 1000
sts-agent battle --encounter small_slimes --seed 42 --agent random -vv
```

## Concerns

sts_agent only concerns itself with decisions an agent playing a game needs to make. It does not concern itself with the game itself. Things like which cards should I pick are agent's concerns, while what are the available options for picking are environment's concerns.

## Architecture

This is a Slay the Spire AI agent that plays full Act 1 runs. There are two distinct layers of decision-making:

**Battle layer** (`src/sts_agent/battle/`): In-combat card play. Three implementations:
- `RandomAgent` — uniform random over legal actions
- `TreeSearchPlanner` — exhaustive minimax DFS with alpha-pruning and transposition table
- `MCTSPlanner` — open-loop UCT (stochastic transitions absorbed into value estimates, not branched)

Both `TreeSearchPlanner` and `MCTSPlanner` clone the `Combat` object to simulate branches without side-effects. Terminal scoring: `damage_taken` if alive, `damage_taken * 2` if dead (ensures surviving branches dominate).

**Strategy layer** (`src/sts_agent/strategy/`): Run-level decisions (route, card rewards, rest, shop, relics). Three implementations:
- `BaseStrategyAgent` — random valid choices, serves as the base class
- `SimStrategyAgent` — uses MCTS probes to evaluate card choices and route options
- `StrategyAgent` (LLM) — `dspy.ReAct` with tools `simulate_upcoming` and `try_card`; needs `GLM_API_KEY` in `.env`

**Run orchestrator** (`src/sts_agent/run.py`): Chains combats into full runs. `run_act1()` walks a generated Act 1 map (15 floors), calling the battle planner for each combat and the strategy agent for all other decisions. MLflow spans are emitted per-floor. Returns a `RunResult` dataclass.

**CLI** (`src/sts_agent/cli.py`): Entry point `sts-agent`. Exit codes: 0 win, 1 died, 2 node budget exceeded.

## Key Design Decisions

**Protocol-based battle agents**: `BattleAgent` and `BattlePlanner` are `Protocol` classes (structural typing), not base classes. `BattleAgent.act(obs, actions)` gets no environment access; `BattlePlanner.act(combat)` gets the full clonable `Combat`.

**Open-loop MCTS**: The MCTS does not branch on stochastic events (shuffles, enemy intents). Each simulation reseeds independently, so variance appears in edge statistics rather than tree structure. This keeps the tree manageable.

**Transposition table (tree search)**: State key normalizes hand/discard/exhaust as sorted sets (order-invariant), but preserves draw pile order. Memoizes terminal scores to prune duplicate subtrees.

**Strategy agent injection**: `run_act1(planner, seed, strategy_agent=None)` defaults to `BaseStrategyAgent(seed=seed)` if `None`. Swap implementations without changing run logic.

**MLflow tracing**: `setup_tracing()` in `tracing.py` enables DSPy autolog and writes to `sqlite:///traces.db`. `run_act1` and floor spans are decorated/wrapped with `@mlflow.trace`.

## Testing Notes

- `conftest.py` provides `ironclad_vs_cultist` and `ironclad_vs_jaw_worm` fixtures.
- Long tests are marked `@pytest.mark.slow` and skipped by default (use `--run-slow`).
- Tests are deterministic: same seed → same action sequence.
- `pytest-explicit` plugin enforces the `--run-slow` / `--run-all` opt-in for slow tests.
