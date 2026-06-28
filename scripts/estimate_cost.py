"""Rough cost estimator for exp1 runs.

Usage:
    conda run -n SkillSimulate python scripts/estimate_cost.py --config configs/exp1_wikipedia_smoke.yaml \
        --conditions 5 --rounds 20 --population 15 --threads 10 \
        --reflection-interval 20 --no-skill-compile
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def load_model_rates(models_path: str):
    with open(models_path) as f:
        data = yaml.safe_load(f)
    return {m["name"]: (m["cost_per_1k_input"], m["cost_per_1k_output"]) for m in data["models"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/exp1_wikipedia_smoke.yaml")
    parser.add_argument("--models", default="configs/models.yaml")
    parser.add_argument("--conditions", type=int, default=13, help="Number of conditions to run")
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--population", type=int, default=30)
    parser.add_argument("--threads", type=int, default=5, help="Sim threads per cell")
    parser.add_argument("--turns-per-agent-per-round", type=float, default=1.0)
    parser.add_argument("--reflection-interval", type=int, default=10)
    parser.add_argument("--disable-reflection", action="store_true")
    parser.add_argument("--no-skill-compile", action="store_true", help="Reuse existing skills")
    parser.add_argument("--clusters", type=int, default=4)
    parser.add_argument("--compile-calls-per-cluster", type=int, default=3)
    parser.add_argument("--planner-input-tokens", type=int, default=1200)
    parser.add_argument("--planner-output-tokens", type=int, default=200)
    parser.add_argument("--reflection-input-tokens", type=int, default=2500)
    parser.add_argument("--reflection-output-tokens", type=int, default=400)
    parser.add_argument("--compile-input-tokens", type=int, default=8000)
    parser.add_argument("--compile-output-tokens", type=int, default=1500)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    sim_model = cfg.get("models", ["deepseek-v4-flash"])[0]
    compile_model = cfg.get("compile_model", sim_model)
    rates = load_model_rates(args.models)
    sim_in, sim_out = rates.get(sim_model, (0.0, 0.0))
    comp_in, comp_out = rates.get(compile_model, (0.0, 0.0))

    n_conditions = args.conditions
    n_rounds = args.rounds
    n_agents = args.population
    turns = args.turns_per_agent_per_round

    planner_calls = n_conditions * n_rounds * n_agents * turns
    planner_cost = planner_calls * (
        args.planner_input_tokens / 1000 * sim_in +
        args.planner_output_tokens / 1000 * sim_out
    )

    if args.disable_reflection:
        reflection_calls = 0
        reflection_cost = 0.0
    else:
        reflection_calls = n_conditions * n_agents * max(1, n_rounds // args.reflection_interval)
        reflection_cost = reflection_calls * (
            args.reflection_input_tokens / 1000 * sim_in +
            args.reflection_output_tokens / 1000 * sim_out
        )

    if args.no_skill_compile:
        compile_cost = 0.0
        compile_calls = 0
    else:
        compile_calls = args.clusters * args.compile_calls_per_cluster
        compile_cost = compile_calls * (
            args.compile_input_tokens / 1000 * comp_in +
            args.compile_output_tokens / 1000 * comp_out
        )

    total = planner_cost + reflection_cost + compile_cost

    print(f"Config: {args.config}")
    print(f"Conditions: {n_conditions}, Rounds: {n_rounds}, Population: {n_agents}, Threads: {args.threads}")
    print(f"Simulation model: {sim_model} (${sim_in:.2f}/1k in, ${sim_out:.2f}/1k out)")
    print(f"Compile model: {compile_model} (${comp_in:.2f}/1k in, ${comp_out:.2f}/1k out)")
    print("")
    print(f"Planner calls:        {int(planner_calls):,}  -> ${planner_cost:,.2f}")
    print(f"Reflection calls:     {int(reflection_calls):,}  -> ${reflection_cost:,.2f}")
    print(f"Skill compile calls:  {int(compile_calls):,}  -> ${compile_cost:,.2f}")
    print(f"TOTAL ESTIMATED COST: ${total:,.2f}")
    print("")
    print("NOTE: Token counts are rough assumptions. If your actual DeepSeek pricing is lower "
          "than the rates in configs/models.yaml, scale the total proportionally.")


if __name__ == "__main__":
    main()
