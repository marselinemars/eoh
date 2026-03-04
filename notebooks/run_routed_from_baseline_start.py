import hashlib
import json
import os
import random
import shutil
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np

from hpc_llm_setup import (
    config_from_env,
    ensure_eoh_src_on_path,
    start_hpc_bridge,
    stop_hpc_bridge,
    test_bridge,
)


def _ts() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")


def _make_logger(status_log_path: Path):
    status_log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(message: str, **fields):
        record = {"time": _ts(), "message": message}
        record.update(fields)
        print(f"[{record['time']}] {message}", flush=True)
        with status_log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    return log


def _parse_seed_dirs(compare_root: Path):
    seed_dirs = sorted([p for p in compare_root.glob("seed_*") if p.is_dir()])
    seeds = []
    for d in seed_dirs:
        try:
            seeds.append(int(d.name.split("seed_", 1)[1]))
        except Exception:
            continue
    return seeds


def _parse_compare_seeds(compare_root: Path):
    raw = os.getenv("EOH_COMPARE_SEEDS", "").strip()
    if raw:
        seeds = [int(s.strip()) for s in raw.split(",") if s.strip()]
        return seeds
    auto = _parse_seed_dirs(compare_root)
    if auto:
        return auto
    return [int(os.getenv("EOH_SEED", "2024"))]


def _seed_root(compare_root: Path, seed: int) -> Path:
    candidate = compare_root / f"seed_{seed}"
    if candidate.exists():
        return candidate
    return compare_root


def _code_hash(code: str):
    return hashlib.sha1(code.encode("utf-8")).hexdigest()[:12]


def _build_continue_population(seed_root: Path, log):
    baseline_pop0 = seed_root / "baseline" / "results" / "pops" / "population_generation_0.json"
    shared_seed = seed_root / "shared_initial_population.json"
    continue_pop = seed_root / "routed_continue_population_from_baseline0.json"

    if not baseline_pop0.exists():
        raise RuntimeError(f"Missing baseline generation-0 population: {baseline_pop0}")

    with baseline_pop0.open("r", encoding="utf-8") as f:
        baseline_data = json.load(f)
    if not isinstance(baseline_data, list) or len(baseline_data) == 0:
        raise RuntimeError(f"Invalid baseline generation-0 population file: {baseline_pop0}")

    # If baseline pop is full, reuse it as continue population.
    if isinstance(baseline_data[0], dict) and "code" in baseline_data[0]:
        with continue_pop.open("w", encoding="utf-8") as f:
            json.dump(baseline_data, f, indent=2)
        log("continue population built from full baseline pop0", continue_path=str(continue_pop))
        return continue_pop

    if not shared_seed.exists():
        raise RuntimeError(
            f"Baseline pop0 is compact (no code) and shared seed file is missing: {shared_seed}"
        )

    with shared_seed.open("r", encoding="utf-8") as f:
        shared_seed_data = json.load(f)
    if not isinstance(shared_seed_data, list) or len(shared_seed_data) == 0:
        raise RuntimeError(f"Invalid shared seed file: {shared_seed}")

    by_hash = {}
    for item in shared_seed_data:
        if not isinstance(item, dict):
            continue
        code = item.get("code")
        algorithm = item.get("algorithm")
        if isinstance(code, str) and isinstance(algorithm, str):
            by_hash[_code_hash(code)] = {"code": code, "algorithm": algorithm}

    full_pop = []
    missing = 0
    for row in baseline_data:
        if not isinstance(row, dict):
            continue
        h = row.get("code_hash")
        obj = row.get("objective")
        seed_item = by_hash.get(h)
        if seed_item is None:
            missing += 1
            continue
        full_pop.append(
            {
                "algorithm": seed_item["algorithm"],
                "code": seed_item["code"],
                "objective": obj,
                "other_inf": None,
            }
        )

    if missing > 0:
        raise RuntimeError(
            f"Failed to reconstruct {missing} individuals from baseline pop0 using shared seed file."
        )
    if len(full_pop) == 0:
        raise RuntimeError("Reconstructed continue population is empty.")

    with continue_pop.open("w", encoding="utf-8") as f:
        json.dump(full_pop, f, indent=2)
    log("continue population reconstructed", continue_path=str(continue_pop), n_individuals=len(full_pop))
    return continue_pop


def _run_routed_from_continue(seed_root: Path, bridge_url: str, model_id: str, settings: dict, continue_path: Path, log):
    from eoh import eoh
    from eoh.utils.getParas import Paras

    out_name = os.getenv("EOH_ROUTED_ONLY_OUT_NAME", "routed")
    routed_out = seed_root / out_name
    if routed_out.exists():
        shutil.rmtree(routed_out)
    routed_out.mkdir(parents=True, exist_ok=True)

    random.seed(settings["seed"])
    np.random.seed(settings["seed"])

    paras = Paras()
    paras.set_paras(
        method="eoh",
        problem=settings["problem"],
        llm_use_local=True,
        llm_local_url=bridge_url,
        llm_model=model_id,
        ec_pop_size=settings["pop_size"],
        ec_n_pop=settings["generations"],
        exp_n_proc=settings["n_proc"],
        exp_output_path=str(routed_out),
        exp_debug_mode=False,
        eval_instances_per_gen=settings["eval_instances_per_gen"],
        holdout_instances=settings["holdout_instances"],
        holdout_eval_interval=settings["holdout_eval_interval"],
        route_improvement_epsilon=settings["route_improvement_epsilon"],
        route_warmup_gens=settings["route_warmup_gens"],
        route_e1_cooldown=settings["route_e1_cooldown"],
        route_e2_recent_k=settings["route_e2_recent_k"],
        route_use_probabilistic=settings["route_use_probabilistic"],
        exp_use_continue=True,
        exp_continue_id=0,
        exp_continue_path=str(continue_path),
        eoh_mode="routed",
        log_full_population=False,
    )
    if settings.get("disable_numba", False):
        paras.eva_numba_decorator = False

    log(
        "starting routed-only from baseline start",
        seed=settings["seed"],
        routed_out=str(routed_out),
        continue_path=str(continue_path),
    )
    runner = eoh.EVOL(paras)
    runner.run()
    log(
        "finished routed-only from baseline start",
        seed=settings["seed"],
        run_log=str(routed_out / "results" / "run_log.jsonl"),
    )


def main():
    project_root, eoh_src = ensure_eoh_src_on_path()
    compare_root = Path(os.getenv("EOH_COMPARE_OUT", "./compare_runs")).resolve()
    compare_root.mkdir(parents=True, exist_ok=True)

    settings = {
        "problem": os.getenv("EOH_PROBLEM", "bp_online"),
        "pop_size": int(os.getenv("EOH_POP_SIZE", "8")),
        "generations": int(os.getenv("EOH_N_GENERATIONS", "10")),
        "n_proc": int(os.getenv("EOH_N_PROC", "1")),
        "eval_instances_per_gen": int(os.getenv("EOH_EVAL_INSTANCES_PER_GEN", "256")),
        "holdout_instances": int(os.getenv("EOH_HOLDOUT_INSTANCES", "64")),
        "holdout_eval_interval": int(os.getenv("EOH_HOLDOUT_EVAL_INTERVAL", "1")),
        "route_improvement_epsilon": float(os.getenv("EOH_ROUTE_IMPROVEMENT_EPS", "1e-12")),
        "route_warmup_gens": int(os.getenv("EOH_ROUTE_WARMUP_GENS", "2")),
        "route_e1_cooldown": int(os.getenv("EOH_ROUTE_E1_COOLDOWN", "3")),
        "route_e2_recent_k": int(os.getenv("EOH_ROUTE_E2_RECENT_K", "3")),
        "route_use_probabilistic": os.getenv("EOH_ROUTE_USE_PROBABILISTIC", "1") == "1",
        "disable_numba": os.getenv("EOH_DISABLE_NUMBA", "1") == "1",
    }

    status_log = compare_root / "runner_status_routed_only.jsonl"
    if status_log.exists():
        status_log.unlink()
    log = _make_logger(status_log)

    seeds = _parse_compare_seeds(compare_root)
    log("routed-only runner started", project_root=str(project_root), eoh_src=str(eoh_src), compare_root=str(compare_root))
    log("settings", **settings, seeds=seeds)

    cfg = config_from_env()
    log(
        "hpc config",
        hpc_base=cfg.base_url,
        bridge_port=cfg.port,
        requested_model=cfg.model,
        api_key_present=bool(cfg.api_key),
    )

    server = None
    try:
        log("starting hpc bridge")
        server, _thread, bridge_url, model_id = start_hpc_bridge(cfg)
        log("hpc bridge ready", bridge_url=bridge_url, resolved_model=model_id)
        status, payload = test_bridge(bridge_url)
        log("bridge probe", bridge_test_status=status, bridge_test_payload=payload)

        for seed in seeds:
            seed_root = _seed_root(compare_root, seed)
            seed_settings = dict(settings)
            seed_settings["seed"] = int(seed)
            log("seed start", seed=seed, seed_root=str(seed_root))
            continue_path = _build_continue_population(seed_root, log)
            _run_routed_from_continue(seed_root, bridge_url, model_id, seed_settings, continue_path, log)
            log("seed done", seed=seed)

        log("routed-only runner completed successfully")
    except Exception as exc:
        log("routed-only runner failed", error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        stop_hpc_bridge(server)
        log("Bridge stopped")


if __name__ == "__main__":
    main()
