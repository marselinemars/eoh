import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from statistics import mean, pstdev


OPERATORS = ["e1", "e2", "m1", "m2", "m3"]


def read_jsonl(path: Path):
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def discover_seed_roots(compare_root: Path):
    seed_dirs = sorted([p for p in compare_root.glob("seed_*") if p.is_dir()])
    if seed_dirs:
        return [(p.name, p) for p in seed_dirs]
    return [("seed_single", compare_root)]


def mode_results_dir(seed_root: Path, mode: str):
    candidate = seed_root / mode / "results"
    if candidate.exists():
        return candidate
    flat = seed_root / "results"
    if mode == "baseline" and flat.exists():
        return flat
    return candidate


def to_float_or_none(value):
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def by_gen_last(rows):
    out = {}
    for row in rows:
        gen = row.get("gen")
        if gen is None:
            continue
        try:
            out[int(gen)] = row
        except Exception:
            continue
    return out


def by_gen_list(rows):
    out = {}
    for row in rows:
        gen = row.get("gen")
        if gen is None:
            continue
        try:
            out.setdefault(int(gen), []).append(row)
        except Exception:
            continue
    return out


def flatten_obs(obs_row):
    out = {}
    obs = obs_row.get("observation", {}) if isinstance(obs_row, dict) else {}
    if not isinstance(obs, dict):
        return out
    out["obs_population_size"] = obs.get("population_size")
    out["obs_best_fitness"] = to_float_or_none(obs.get("best_fitness"))
    out["obs_delta_best"] = to_float_or_none(obs.get("delta_best"))
    out["obs_improve_rate_k"] = to_float_or_none(obs.get("improve_rate_k"))
    out["obs_stagnation_len"] = obs.get("stagnation_len")
    out["obs_invalid_rate_k"] = to_float_or_none(obs.get("invalid_rate_k"))
    out["obs_diversity_score"] = to_float_or_none(obs.get("diversity_score"))
    budget = obs.get("budget", {}) if isinstance(obs.get("budget"), dict) else {}
    out["obs_budget_instances"] = budget.get("instances")
    out["obs_budget_holdout_instances"] = budget.get("holdout_instances")
    top = obs.get("top_heuristics", [])
    if isinstance(top, list):
        out["obs_top_heuristics_n"] = len(top)
        complexities = [to_float_or_none(h.get("complexity")) for h in top if isinstance(h, dict)]
        complexities = [c for c in complexities if c is not None]
        out["obs_top_avg_complexity"] = (sum(complexities) / len(complexities)) if complexities else None
    else:
        out["obs_top_heuristics_n"] = 0
        out["obs_top_avg_complexity"] = None
    return out


def flatten_diag(diag_row):
    out = {}
    if not isinstance(diag_row, dict):
        return out
    out["diag_label_primary"] = diag_row.get("diagnosis_label")
    out["diag_summary"] = diag_row.get("summary")
    factors = diag_row.get("factors", {}) if isinstance(diag_row.get("factors"), dict) else {}
    for key in [
        "exploration_need",
        "exploitation_need",
        "diversity_need",
        "invalid_risk",
        "overfit_risk",
        "confidence",
    ]:
        out[f"diag_{key}"] = to_float_or_none(factors.get(key))
    labels = diag_row.get("diagnosis_labels", [])
    if isinstance(labels, list):
        out["diag_labels"] = "|".join(str(x) for x in labels)
    evidence = diag_row.get("evidence", [])
    out["diag_evidence_n"] = len(evidence) if isinstance(evidence, list) else 0
    return out


def flatten_plan(plan_row):
    out = {}
    if not isinstance(plan_row, dict):
        return out
    out["plan_diagnosis_used"] = plan_row.get("diagnosis_used")
    op_probs = plan_row.get("op_probs", {}) if isinstance(plan_row.get("op_probs"), dict) else {}
    for op in OPERATORS:
        out[f"plan_op_{op}"] = to_float_or_none(op_probs.get(op))
    parent_mix = plan_row.get("parent_mix", {}) if isinstance(plan_row.get("parent_mix"), dict) else {}
    out["plan_parent_elite"] = to_float_or_none(parent_mix.get("elite"))
    out["plan_parent_diverse"] = to_float_or_none(parent_mix.get("diverse"))
    out["plan_parent_random"] = to_float_or_none(parent_mix.get("random"))
    eval_plan = plan_row.get("evaluation_plan", {}) if isinstance(plan_row.get("evaluation_plan"), dict) else {}
    out["plan_eval_instances"] = eval_plan.get("instances")
    out["plan_eval_holdout_instances"] = eval_plan.get("holdout_instances")
    prompt_modifiers = plan_row.get("prompt_modifiers", [])
    if isinstance(prompt_modifiers, list):
        out["plan_prompt_modifiers"] = " | ".join(str(x) for x in prompt_modifiers)
    rationale = plan_row.get("rationale", [])
    out["plan_rationale_n"] = len(rationale) if isinstance(rationale, list) else 0
    return out


def flatten_critic(critic_row):
    out = {}
    if not isinstance(critic_row, dict):
        return out
    out["critic_verdict"] = critic_row.get("verdict")
    out["critic_chosen_operator"] = critic_row.get("chosen_operator")
    reasons = critic_row.get("reasons", [])
    if isinstance(reasons, list):
        out["critic_reasons"] = " | ".join(str(x) for x in reasons)
        out["critic_reasons_n"] = len(reasons)
    final_plan = critic_row.get("final_plan", {}) if isinstance(critic_row.get("final_plan"), dict) else {}
    final_op = final_plan.get("op_probs", {}) if isinstance(final_plan.get("op_probs"), dict) else {}
    for op in OPERATORS:
        out[f"final_op_{op}"] = to_float_or_none(final_op.get(op))
    final_parent = final_plan.get("parent_mix", {}) if isinstance(final_plan.get("parent_mix"), dict) else {}
    out["final_parent_elite"] = to_float_or_none(final_parent.get("elite"))
    out["final_parent_diverse"] = to_float_or_none(final_parent.get("diverse"))
    out["final_parent_random"] = to_float_or_none(final_parent.get("random"))
    final_eval = final_plan.get("evaluation_plan", {}) if isinstance(final_plan.get("evaluation_plan"), dict) else {}
    out["final_eval_instances"] = final_eval.get("instances")
    out["final_eval_holdout_instances"] = final_eval.get("holdout_instances")
    final_modifiers = final_plan.get("prompt_modifiers", [])
    if isinstance(final_modifiers, list):
        out["final_prompt_modifiers"] = " | ".join(str(x) for x in final_modifiers)
    return out


def summarize_operator_events_for_gen(op_events):
    out = {}
    if not op_events:
        return out
    out["op_event_n"] = len(op_events)
    executed = [r for r in op_events if r.get("executed", True)]
    out["op_event_executed_n"] = len(executed)
    invalid_rates = [to_float_or_none(r.get("invalid_rate")) for r in executed]
    invalid_rates = [x for x in invalid_rates if x is not None]
    deltas = [to_float_or_none(r.get("delta_best")) for r in executed]
    deltas = [x for x in deltas if x is not None]
    out["op_event_invalid_rate_mean"] = mean(invalid_rates) if invalid_rates else None
    out["op_event_delta_best_mean"] = mean(deltas) if deltas else None
    out["op_event_operators"] = "|".join(str(r.get("operator")) for r in executed)
    return out


def build_mode_timeline(seed_name, mode, results_dir):
    run_rows = read_jsonl(results_dir / "run_log.jsonl")
    op_rows = read_jsonl(results_dir / "operator_events.jsonl")
    run_by_gen = by_gen_last(run_rows)
    op_by_gen = by_gen_list(op_rows)
    all_gens = sorted(set(run_by_gen.keys()) | set(op_by_gen.keys()))

    timelines = []
    for gen in all_gens:
        run = run_by_gen.get(gen, {})
        row = {
            "seed": seed_name,
            "mode": mode,
            "gen": gen,
            "best_fitness": to_float_or_none(run.get("best_fitness")),
            "train_fitness": to_float_or_none(run.get("train_fitness")),
            "holdout_fitness": to_float_or_none(run.get("holdout_fitness")),
            "fitness_gap": to_float_or_none(run.get("fitness_gap")),
            "invalid_rate": to_float_or_none(run.get("invalid_rate")),
            "no_improve_gens": run.get("no_improve_gens"),
            "chosen_operator": run.get("chosen_operator"),
            "diagnosis_label": run.get("diagnosis_label"),
            "overfit_risk": run.get("overfit_risk"),
        }
        row.update(summarize_operator_events_for_gen(op_by_gen.get(gen, [])))
        timelines.append(row)
    return timelines


def build_routed_agentic_timeline(seed_name, routed_results_dir):
    run_rows = read_jsonl(routed_results_dir / "run_log.jsonl")
    op_rows = read_jsonl(routed_results_dir / "operator_events.jsonl")
    obs_rows = read_jsonl(routed_results_dir / "agent_observation.jsonl")
    diag_rows = read_jsonl(routed_results_dir / "agent_diagnosis.jsonl")
    plan_rows = read_jsonl(routed_results_dir / "agent_plan.jsonl")
    critic_rows = read_jsonl(routed_results_dir / "agent_critic.jsonl")

    run_by_gen = by_gen_last(run_rows)
    op_by_gen = by_gen_list(op_rows)
    obs_by_gen = by_gen_last(obs_rows)
    diag_by_gen = by_gen_last(diag_rows)
    plan_by_gen = by_gen_last(plan_rows)
    critic_by_gen = by_gen_last(critic_rows)

    all_gens = sorted(
        set(run_by_gen.keys())
        | set(op_by_gen.keys())
        | set(obs_by_gen.keys())
        | set(diag_by_gen.keys())
        | set(plan_by_gen.keys())
        | set(critic_by_gen.keys())
    )

    rows = []
    for gen in all_gens:
        run = run_by_gen.get(gen, {})
        diag = diag_by_gen.get(gen, {})
        plan = plan_by_gen.get(gen, {})
        critic = critic_by_gen.get(gen, {})
        row = {
            "seed": seed_name,
            "mode": "routed",
            "gen": gen,
            "best_fitness": to_float_or_none(run.get("best_fitness")),
            "train_fitness": to_float_or_none(run.get("train_fitness")),
            "holdout_fitness": to_float_or_none(run.get("holdout_fitness")),
            "fitness_gap": to_float_or_none(run.get("fitness_gap")),
            "run_invalid_rate": to_float_or_none(run.get("invalid_rate")),
            "run_no_improve_gens": run.get("no_improve_gens"),
            "run_chosen_operator": run.get("chosen_operator"),
            "run_diagnosis_label": run.get("diagnosis_label"),
            "run_train_delta": to_float_or_none(run.get("train_delta")),
            "run_overfit_risk": run.get("overfit_risk"),
        }
        row.update(summarize_operator_events_for_gen(op_by_gen.get(gen, [])))
        row.update(flatten_obs(obs_by_gen.get(gen, {})))
        row.update(flatten_diag(diag))
        row.update(flatten_plan(plan))
        row.update(flatten_critic(critic))

        final_probs = {op: row.get(f"final_op_{op}") for op in OPERATORS}
        valid_probs = {k: v for k, v in final_probs.items() if v is not None}
        if valid_probs:
            top_op = max(valid_probs.items(), key=lambda x: x[1])[0]
            row["final_top_operator"] = top_op
            row["chosen_matches_final_top"] = int(top_op == row.get("run_chosen_operator"))
        else:
            row["final_top_operator"] = None
            row["chosen_matches_final_top"] = None
        rows.append(row)
    return rows


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as f:
            f.write("")
        return
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def build_long_prob_rows(agent_rows):
    rows = []
    for row in agent_rows:
        for stage in ["plan", "final"]:
            for op in OPERATORS:
                key = f"{stage}_op_{op}"
                prob = row.get(key)
                if prob is None:
                    continue
                rows.append(
                    {
                        "seed": row["seed"],
                        "gen": row["gen"],
                        "stage": "planner" if stage == "plan" else "critic_final",
                        "operator": op,
                        "prob": prob,
                        "chosen_operator": row.get("run_chosen_operator"),
                    }
                )
    return rows


def build_prompt_modifier_rows(agent_rows):
    rows = []
    for row in agent_rows:
        for stage_key, stage_name in [("plan_prompt_modifiers", "planner"), ("final_prompt_modifiers", "critic_final")]:
            mods = row.get(stage_key)
            if not mods:
                continue
            parts = [p.strip() for p in str(mods).split("|") if p.strip()]
            for rank, text in enumerate(parts, start=1):
                rows.append(
                    {
                        "seed": row["seed"],
                        "gen": row["gen"],
                        "stage": stage_name,
                        "modifier_rank": rank,
                        "modifier": text,
                    }
                )
    return rows


def build_critic_reason_rows(agent_rows):
    rows = []
    for row in agent_rows:
        reasons = row.get("critic_reasons")
        if not reasons:
            continue
        parts = [p.strip() for p in str(reasons).split("|") if p.strip()]
        for rank, text in enumerate(parts, start=1):
            rows.append(
                {
                    "seed": row["seed"],
                    "gen": row["gen"],
                    "reason_rank": rank,
                    "reason": text,
                }
            )
    return rows


def final_mode_fitness(mode_rows):
    if not mode_rows:
        return None
    by_seed = {}
    for row in mode_rows:
        seed = row["seed"]
        by_seed.setdefault(seed, [])
        by_seed[seed].append(row)
    out = {}
    for seed, rows in by_seed.items():
        rows = sorted(rows, key=lambda r: r.get("gen", 0))
        last = rows[-1]
        fit = last.get("train_fitness")
        if fit is None:
            fit = last.get("best_fitness")
        out[seed] = to_float_or_none(fit)
    return out


def build_summary(all_timeline_rows, agent_rows):
    baseline_rows = [r for r in all_timeline_rows if r.get("mode") == "baseline"]
    routed_rows = [r for r in all_timeline_rows if r.get("mode") == "routed"]
    base_final = final_mode_fitness(baseline_rows) or {}
    routed_final = final_mode_fitness(routed_rows) or {}

    seeds = sorted(set(base_final.keys()) | set(routed_final.keys()))
    by_seed_rows = []
    improvements = []
    wins = 0
    for seed in seeds:
        b = base_final.get(seed)
        r = routed_final.get(seed)
        delta = None
        if b is not None and r is not None:
            delta = b - r
            improvements.append(delta)
            if delta > 0:
                wins += 1
        by_seed_rows.append(
            {
                "seed": seed,
                "baseline_final_fitness": b,
                "routed_final_fitness": r,
                "fitness_improvement_baseline_minus_routed": delta,
                "routed_better": int(delta > 0) if delta is not None else None,
            }
        )

    chosen_ops = [r.get("run_chosen_operator") for r in agent_rows if r.get("run_chosen_operator")]
    diag_labels = [r.get("diag_label_primary") for r in agent_rows if r.get("diag_label_primary")]
    matches = [r.get("chosen_matches_final_top") for r in agent_rows if r.get("chosen_matches_final_top") is not None]
    matches = [int(x) for x in matches]

    overall = {
        "n_seeds": len(seeds),
        "n_agentic_generations": len(agent_rows),
        "baseline_final_mean": mean([v for v in base_final.values() if v is not None]) if base_final else None,
        "baseline_final_std": pstdev([v for v in base_final.values() if v is not None]) if len(base_final) > 1 else 0.0 if len(base_final) == 1 else None,
        "routed_final_mean": mean([v for v in routed_final.values() if v is not None]) if routed_final else None,
        "routed_final_std": pstdev([v for v in routed_final.values() if v is not None]) if len(routed_final) > 1 else 0.0 if len(routed_final) == 1 else None,
        "improvement_mean_baseline_minus_routed": mean(improvements) if improvements else None,
        "improvement_std_baseline_minus_routed": pstdev(improvements) if len(improvements) > 1 else 0.0 if len(improvements) == 1 else None,
        "routed_win_rate": (wins / len(improvements)) if improvements else None,
        "chosen_operator_counts": dict(Counter(chosen_ops)),
        "diagnosis_label_counts": dict(Counter(diag_labels)),
        "chosen_matches_final_top_rate": (sum(matches) / len(matches)) if matches else None,
    }
    return by_seed_rows, overall


def build_trace_text(agent_rows):
    lines = []
    for seed in sorted(set(r["seed"] for r in agent_rows)):
        lines.append(f"# {seed}")
        seed_rows = sorted([r for r in agent_rows if r["seed"] == seed], key=lambda x: x["gen"])
        for r in seed_rows:
            lines.append(
                "gen={gen} diag={diag} exp={exp} expl={expl} inv={inv} "
                "plan[e2={pe2},m2={pm2}] final[e2={fe2},m2={fm2}] chosen={op} "
                "delta={delta} fit={fit} verdict={verdict}".format(
                    gen=r.get("gen"),
                    diag=r.get("diag_label_primary"),
                    exp=r.get("diag_exploration_need"),
                    expl=r.get("diag_exploitation_need"),
                    inv=r.get("diag_invalid_risk"),
                    pe2=r.get("plan_op_e2"),
                    pm2=r.get("plan_op_m2"),
                    fe2=r.get("final_op_e2"),
                    fm2=r.get("final_op_m2"),
                    op=r.get("run_chosen_operator"),
                    delta=r.get("run_train_delta"),
                    fit=r.get("train_fitness"),
                    verdict=r.get("critic_verdict"),
                )
            )
        lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Aggregate EOH baseline/routed + agent logs for analysis.")
    parser.add_argument("--compare-root", type=str, default="./compare_runs", help="Root folder containing seed_* outputs.")
    parser.add_argument("--out", type=str, default=None, help="Output folder (default: <compare-root>/analysis_agentic).")
    args = parser.parse_args()

    compare_root = Path(args.compare_root).resolve()
    out_root = Path(args.out).resolve() if args.out else (compare_root / "analysis_agentic")
    out_root.mkdir(parents=True, exist_ok=True)

    all_timeline_rows = []
    routed_agent_rows = []
    for seed_name, seed_root in discover_seed_roots(compare_root):
        base_dir = mode_results_dir(seed_root, "baseline")
        routed_dir = mode_results_dir(seed_root, "routed")
        all_timeline_rows.extend(build_mode_timeline(seed_name, "baseline", base_dir))
        all_timeline_rows.extend(build_mode_timeline(seed_name, "routed", routed_dir))
        routed_agent_rows.extend(build_routed_agentic_timeline(seed_name, routed_dir))

    prob_long_rows = build_long_prob_rows(routed_agent_rows)
    prompt_rows = build_prompt_modifier_rows(routed_agent_rows)
    reason_rows = build_critic_reason_rows(routed_agent_rows)
    by_seed_rows, overall = build_summary(all_timeline_rows, routed_agent_rows)
    trace_text = build_trace_text(routed_agent_rows)

    write_csv(out_root / "timeline_all_generations.csv", all_timeline_rows)
    write_csv(out_root / "routed_agentic_steps.csv", routed_agent_rows)
    write_csv(out_root / "routed_op_probs_long.csv", prob_long_rows)
    write_csv(out_root / "routed_prompt_modifiers_long.csv", prompt_rows)
    write_csv(out_root / "routed_critic_reasons_long.csv", reason_rows)
    write_csv(out_root / "summary_by_seed.csv", by_seed_rows)
    write_json(out_root / "summary_overall.json", overall)
    (out_root / "routed_agentic_trace.txt").write_text(trace_text, encoding="utf-8")

    print(f"compare_root: {compare_root}")
    print(f"out_root: {out_root}")
    print(f"timeline rows: {len(all_timeline_rows)}")
    print(f"routed agent rows: {len(routed_agent_rows)}")
    print(f"summary seeds: {len(by_seed_rows)}")
    print("wrote:")
    for name in [
        "timeline_all_generations.csv",
        "routed_agentic_steps.csv",
        "routed_op_probs_long.csv",
        "routed_prompt_modifiers_long.csv",
        "routed_critic_reasons_long.csv",
        "summary_by_seed.csv",
        "summary_overall.json",
        "routed_agentic_trace.txt",
    ]:
        print(" -", out_root / name)


if __name__ == "__main__":
    main()
