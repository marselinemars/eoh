import argparse
import json
import os
import random
import sys
from pathlib import Path


def ensure_eoh_src_on_path():
    cwd = Path.cwd().resolve()
    candidates = [cwd, cwd.parent, cwd.parent.parent]
    for base in candidates:
        src = base / "eoh" / "src"
        if src.exists():
            if str(src) not in sys.path:
                sys.path.insert(0, str(src))
            return
    raise RuntimeError(f"Could not locate eoh/src from cwd={cwd}")


def build_dummy_observation(gen, base_best=0.0140):
    jitter = (random.random() - 0.5) * 0.0003
    best = max(0.0100, base_best - 0.00005 * min(gen, 6) + jitter)
    stagnation_len = max(0, gen - 5)
    improve_rate = 0.5 if gen < 4 else (0.1 if stagnation_len > 0 else 0.3)
    invalid_rate = 0.05 if gen < 8 else 0.15
    diversity = max(0.2, 0.75 - 0.04 * gen)

    op_stats = {
        "e1": {"calls": 2, "success_rate": 0.20, "mean_delta": -0.0001, "invalid_rate": 0.35},
        "e2": {"calls": 8, "success_rate": 0.62, "mean_delta": 0.0009, "invalid_rate": 0.08},
        "m1": {"calls": 5, "success_rate": 0.40, "mean_delta": 0.0004, "invalid_rate": 0.18},
        "m2": {"calls": 6, "success_rate": 0.52, "mean_delta": 0.0005, "invalid_rate": 0.05},
        "m3": {"calls": 4, "success_rate": 0.34, "mean_delta": 0.0002, "invalid_rate": 0.04},
    }
    if stagnation_len >= 3:
        op_stats["m2"]["mean_delta"] = 0.0002
        op_stats["e2"]["mean_delta"] = 0.0011

    return {
        "gen": int(gen + 1),
        "population_size": 10,
        "best_fitness": float(best),
        "best_history": [float(best + 0.0003), float(best + 0.0002), float(best + 0.0001), float(best)],
        "delta_best": 0.0001 if gen % 3 == 0 else 0.0,
        "improve_rate_k": float(improve_rate),
        "stagnation_len": int(stagnation_len),
        "invalid_rate_k": float(invalid_rate),
        "diversity_score": float(diversity),
        "op_stats_k": op_stats,
        "last_used_ops": ["e2", "m2", "e2", "m1", "e2"][-5:],
        "top_heuristics": [
            {"id": "h1", "fitness": float(best), "summary": "balanced scoring", "complexity": 220.0},
            {"id": "h2", "fitness": float(best + 0.0001), "summary": "slightly conservative", "complexity": 240.0},
        ],
        "budget": {"instances": 256, "holdout_instances": 0},
        "notes": "dry_test",
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Dry-test controller JSON reliability (no EOH evaluation).")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--api-endpoint", default=os.getenv("EOH_LLM_API_ENDPOINT", "xxx"))
    parser.add_argument("--api-key", default=os.getenv("EOH_LLM_API_KEY", "xxx"))
    parser.add_argument("--model", default=os.getenv("EOH_LLM_MODEL", "xxx"))
    parser.add_argument("--use-local", action="store_true", default=os.getenv("EOH_LLM_USE_LOCAL", "1") == "1")
    parser.add_argument("--local-url", default=os.getenv("EOH_LLM_LOCAL_URL", "http://127.0.0.1:18000/completions"))
    parser.add_argument("--use-critic", action="store_true", help="Enable Critic stage in controller.")
    parser.add_argument("--mock", action="store_true", help="Use mock LLM responses (offline sanity).")
    parser.add_argument("--mock-invalid-rate", type=float, default=0.05)
    parser.add_argument("--output", default="compare_runs/controller_json_dry_test_summary.json")
    return parser.parse_args()


class MockLLM:
    def __init__(self, invalid_rate=0.05):
        self.invalid_rate = max(0.0, min(1.0, float(invalid_rate)))

    def _maybe_invalid(self):
        return random.random() < self.invalid_rate

    def get_response(self, prompt_content, request_mode="code", tool_name=None, json_schema=None, stop=None):
        if self._maybe_invalid():
            return "invalid json text"
        if isinstance(prompt_content, str) and "Infer the search state using ONLY the telemetry" in prompt_content:
            return json.dumps(
                {
                    "summary": "stagnation_len high with low diversity_score.",
                    "factors": {
                        "exploration_need": 0.72,
                        "exploitation_need": 0.28,
                        "diversity_need": 0.63,
                        "invalid_risk": 0.11,
                        "overfit_risk": 0.20,
                        "confidence": 0.86,
                    },
                    "diagnosis_labels": ["STAGNATION", "LOW_DIVERSITY"],
                    "evidence": [
                        "stagnation_len=4 and improve_rate_k=0.1 indicate plateau.",
                        "diversity_score=0.35 and invalid_rate_k=0.08 suggest safe structured exploration.",
                        "Top mean_delta operator is e2 from op_stats_k.e2.mean_delta=0.0011.",
                    ],
                }
            )
        if isinstance(prompt_content, str) and "Create a generation plan grounded in telemetry" in prompt_content:
            return json.dumps(
                {
                    "diagnosis_used": "stagnation with low invalid risk",
                    "op_probs": {"e1": 0.02, "e2": 0.54, "m1": 0.16, "m2": 0.20, "m3": 0.08},
                    "parent_mix": {"elite": 0.50, "diverse": 0.30, "random": 0.20},
                    "prompt_modifiers": ["Keep scoring simple.", "Avoid many constants."],
                    "evaluation_plan": {"instances": 256, "holdout_instances": 0},
                    "rationale": [
                        "stagnation_len=4 and diversity_score=0.35 support structured exploration.",
                        "op_stats_k.e2.mean_delta=0.0011 is strongest while invalid_rate_k=0.08 remains low.",
                    ],
                }
            )
        return json.dumps(
            {
                "verdict": "approve",
                "reasons": ["No guardrail violation."],
                "final_plan": {
                    "op_probs": {"e1": 0.02, "e2": 0.54, "m1": 0.16, "m2": 0.20, "m3": 0.08},
                    "parent_mix": {"elite": 0.50, "diverse": 0.30, "random": 0.20},
                    "prompt_modifiers": ["Keep scoring simple."],
                    "evaluation_plan": {"instances": 256, "holdout_instances": 0},
                },
            }
        )


def main():
    args = parse_args()
    ensure_eoh_src_on_path()
    from eoh.methods.eoh.agentic_controller import AgenticController

    if args.mock:
        controller = object.__new__(AgenticController)
        controller.debug_mode = False
        controller.use_critic_agent = bool(args.use_critic)
        controller.max_json_retries = max(1, int(os.getenv("EOH_CONTROLLER_JSON_RETRIES", "3")))
        controller.interface_llm = MockLLM(invalid_rate=args.mock_invalid_rate)
    else:
        controller = AgenticController(
            api_endpoint=args.api_endpoint,
            api_key=args.api_key,
            model_llm=args.model,
            llm_use_local=bool(args.use_local),
            llm_local_url=args.local_url,
            use_critic_agent=bool(args.use_critic),
            debug_mode=False,
        )

    stats = {
        "diagnoser": {"ok": 0, "llm_success": 0, "fallback": 0},
        "planner": {"ok": 0, "llm_success": 0, "fallback": 0},
    }
    stage_order = [("diagnoser", "diagnoser"), ("planner", "planner")]
    if args.use_critic:
        stats["critic"] = {"ok": 0, "llm_success": 0, "fallback": 0}
        stage_order.append(("critic", "critic"))
    failures = []

    for i in range(max(1, int(args.iters))):
        obs = build_dummy_observation(i)
        try:
            out = controller.run(obs)
        except Exception as exc:
            failures.append({"iter": i + 1, "error": str(exc)})
            continue

        for stage_key, stage_name in stage_order:
            dbg = out.get("debug", {}).get(stage_key, {})
            llm = dbg.get("llm", {}) if isinstance(dbg, dict) else {}
            flags = dbg.get("flags", {}) if isinstance(dbg, dict) else {}
            ok = bool(llm.get("parse_ok", False) and llm.get("validation_ok", False))
            stats[stage_name]["ok"] += 1 if ok else 0
            stats[stage_name]["llm_success"] += 1 if bool(llm.get("llm_success", False)) else 0
            stats[stage_name]["fallback"] += 1 if bool(flags.get("fully_fallback", False)) else 0

    total = max(1, int(args.iters))
    summary = {
        "iters": total,
        "threshold": args.threshold,
        "stages": {},
        "failures": failures,
    }
    overall_pass = True
    for stage in [stage for _, stage in stage_order]:
        ok_rate = stats[stage]["ok"] / float(total)
        llm_success_rate = stats[stage]["llm_success"] / float(total)
        fallback_rate = stats[stage]["fallback"] / float(total)
        summary["stages"][stage] = {
            "ok_rate": ok_rate,
            "llm_success_rate": llm_success_rate,
            "fallback_rate": fallback_rate,
            "ok_count": stats[stage]["ok"],
            "llm_success_count": stats[stage]["llm_success"],
            "fallback_count": stats[stage]["fallback"],
        }
        if ok_rate < float(args.threshold):
            overall_pass = False

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    if not overall_pass:
        print("Dry test failed threshold.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
