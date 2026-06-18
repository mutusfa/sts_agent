"""Tests for SimStrategyAgent shop decisions."""

from __future__ import annotations

from sts_env.run.character import Character
from sts_env.run.encounter_queue import EncounterQueue
from sts_env.run.rng_streams import RunRNG
from sts_env.run.shop import ShopInventory, generate_shop

from sts_agent.strategy.sim_agent import SimStrategyAgent


class TestSimStrategyAgentShop:
    def test_falls_back_to_heuristic_without_encounter_tracking(self):
        agent = SimStrategyAgent(sim_nodes=100, sim_sims=100, seed=0)
        ch = Character.ironclad()
        ch.gold = 500
        ch.floor = 3
        gold_before = ch.gold
        inv = generate_shop(RunRNG(0), ch.floor, ch)

        agent.shop(inv, ch)

        assert ch.gold < gold_before

    def test_probe_shop_runs_with_encounter_tracking(self):
        agent = SimStrategyAgent(sim_nodes=100, sim_sims=100, seed=0)
        ch = Character.ironclad()
        ch.gold = 500
        ch.floor = 5
        inv = generate_shop(RunRNG(1), ch.floor, ch)
        queue = EncounterQueue(RunRNG(0xBEEF))
        agent.set_encounter_tracking(queue, [], [])

        agent.shop(inv, ch)

        # Shop visit completes without error; may leave if probes find no value.
        assert ch.gold <= 500

    def test_executes_buy_when_card_clearly_helps(self, monkeypatch):
        """Force probe scores so buying is strictly better than leaving."""
        agent = SimStrategyAgent(sim_nodes=50, sim_sims=50, seed=0)
        ch = Character.ironclad()
        ch.gold = 200
        ch.floor = 3
        inv = ShopInventory(
            cards=[("Inflame", 50)],
            potions=[],
            relics=[None, None, None],
            remove_cost=75,
        )
        queue = EncounterQueue(RunRNG(0xBEEF))
        agent.set_encounter_tracking(queue, [], [])

        from sts_agent.strategy import shop_eval as shop_eval_mod
        from sts_agent.strategy import sim_agent as sim_agent_mod

        def fake_evaluate(action, character, inventory, encounters, seed, **kwargs):
            if action == "leave":
                return shop_eval_mod.ShopScore(
                    action="leave",
                    total_expected_damage=100.0,
                    total_survival_rate=0.5,
                    worst_max_score=200.0,
                )
            return shop_eval_mod.ShopScore(
                action=action,
                total_expected_damage=10.0,
                total_survival_rate=1.0,
                worst_max_score=50.0,
                price=50,
            )

        monkeypatch.setattr(sim_agent_mod, "evaluate_shop_option", fake_evaluate)
        monkeypatch.setattr(
            sim_agent_mod,
            "evaluate_shop_baseline",
            lambda character, encounters, seed, **kwargs: shop_eval_mod.ShopScore(
                action="leave",
                total_expected_damage=100.0,
                total_survival_rate=0.5,
                worst_max_score=200.0,
            ),
        )

        agent.shop(inv, ch)

        assert "Inflame" in ch.deck
        assert ch.gold < 200
