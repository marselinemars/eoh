import json
import os
import random
import shutil
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np

from hpc_llm_setup import (
    config_from_env,
    ensure_eoh_src_on_path,
    resolve_model_id,
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


def _tail_run_log(run_log_path: Path, stop_event: threading.Event, log, experiment: str):
    offset = 0
    while not stop_event.is_set():
        if run_log_path.exists():
            with run_log_path.open("r", encoding="utf-8") as f:
                f.seek(offset)
                while True:
                    line = f.readline()
                    if not line:
                        break
                    offset = f.tell()
                    try:
                        rec = json.loads(line)
                        log(
                            f"{experiment} progress: gen={rec.get('gen')} best={rec.get('best_fitness')} op={rec.get('chosen_operator')} "
                            f"mem_mode={rec.get('memory_mode')} mem_written={rec.get('memory_entries_written_so_far')} mem_reads={rec.get('memory_retrieval_events_so_far')}",
                            experiment=experiment,
                            event="generation_progress",
                            gen=rec.get("gen"),
                            best_fitness=rec.get("best_fitness"),
                            chosen_operator=rec.get("chosen_operator"),
                            memory_mode=rec.get("memory_mode"),
                            memory_entries_written_so_far=rec.get("memory_entries_written_so_far"),
                            memory_retrieval_events_so_far=rec.get("memory_retrieval_events_so_far"),
                        )
                    except Exception:
                        log(f"{experiment} progress raw: {line.strip()}", experiment=experiment, event="generation_progress_raw")
        time.sleep(2)


def _heartbeat(mode_root: Path, run_log_path: Path, stop_event: threading.Event, log, experiment: str):
    started = time.time()
    while not stop_event.is_set():
        elapsed = int(time.time() - started)
        run_log_lines = 0
        if run_log_path.exists():
            with run_log_path.open("r", encoding="utf-8") as f:
                run_log_lines = sum(1 for _ in f)
        log(
            f"{experiment} heartbeat: elapsed={elapsed}s run_log_lines={run_log_lines}",
            experiment=experiment,
            event="heartbeat",
            elapsed_s=elapsed,
            run_log_lines=run_log_lines,
            mode_root=str(mode_root),
        )
        time.sleep(20)


def _prepare_shared_seed(seed_root: Path, bridge_url: str, model_id: str, settings: dict, log):
    from eoh import eoh
    from eoh.utils.getParas import Paras

    seed_build_out = seed_root / "_shared_seed_build"
    shared_seed_path = seed_root / "shared_initial_population.json"
    seed_build_out.mkdir(parents=True, exist_ok=True)

    paras = Paras()
    paras.set_paras(
        method="eoh",
        problem=settings["problem"],
        llm_use_local=True,
        llm_local_url=bridge_url,
        llm_model=model_id,
        ec_pop_size=settings["pop_size"],
        ec_n_pop=0,
        exp_n_proc=settings["n_proc"],
        exp_output_path=str(seed_build_out),
        exp_debug_mode=False,
        eval_parallel_instances=settings["eval_parallel_instances"],
        eval_instances_per_gen=settings["eval_instances_per_gen"],
        holdout_instances=settings["holdout_instances"],
        holdout_eval_interval=settings["holdout_eval_interval"],
        eoh_mode="baseline",
        log_full_population=True,
        use_experience_memory=False,
    )
    if settings.get("disable_numba", False):
        paras.eva_numba_decorator = False

    runner = eoh.EVOL(paras)
    runner.run()

    pop0_path = seed_build_out / "results" / "pops" / "population_generation_0.json"
    if not pop0_path.exists():
        raise RuntimeError(f"Shared-seed build failed: missing {pop0_path}")

    with pop0_path.open("r", encoding="utf-8") as f:
        pop0 = json.load(f)

    seeds = []
    for ind in pop0:
        if not isinstance(ind, dict):
            continue
        code = ind.get("code")
        algorithm = ind.get("algorithm")
        if isinstance(code, str) and isinstance(algorithm, str):
            seeds.append({"algorithm": algorithm, "code": code})
    if len(seeds) == 0:
        raise RuntimeError("Shared-seed build failed: no valid seed algorithms found.")

    with shared_seed_path.open("w", encoding="utf-8") as f:
        json.dump(seeds, f, indent=2)
    log("shared initial population ready", event="shared_seed_prepare_done", shared_seed_path=str(shared_seed_path), seed_count=len(seeds))
    return shared_seed_path


def _experiment_configs(memory_store_path: Path):
    return [
        {
            "name": "A_vanilla_eoh",
            "mode": "baseline",
            "use_shared_seed": True,
            "paras": {
                "use_experience_memory": False,
                "memory_mode": "off",
                "memory_read_enabled": False,
                "memory_write_enabled": False,
                "memory_read_only": False,
            },
        },
        {
            "name": "B_memory_write_only",
            "mode": "baseline",
            "use_shared_seed": True,
            "paras": {
                "use_experience_memory": True,
                "memory_mode": "retrieve_for_generation",
                "memory_store_path": str(memory_store_path),
                "memory_read_enabled": False,
                "memory_write_enabled": True,
                "memory_read_only": False,
                "memory_reset_on_start": False,
            },
        },
        {
            "name": "C_memory_retrieval",
            "mode": "baseline",
            "use_shared_seed": True,
            "paras": {
                "use_experience_memory": True,
                "memory_mode": "retrieve_for_generation",
                "memory_store_path": str(memory_store_path),
                "memory_read_enabled": True,
                "memory_write_enabled": False,
                "memory_read_only": True,
                "memory_reset_on_start": False,
            },
        },
        {
            "name": "D_memory_retrieval_plus_seed",
            "mode": "baseline",
            "use_shared_seed": False,
            "paras": {
                "use_experience_memory": True,
                "memory_mode": "retrieve_plus_seed",
                "memory_store_path": str(memory_store_path),
                "memory_read_enabled": True,
                "memory_write_enabled": False,
                "memory_read_only": True,
                "memory_seed_top_n": int(os.getenv("EOH_MEMORY_SEED_TOP_N", "2")),
                "memory_reset_on_start": False,
            },
        },
    ]


def _selected_experiment_names():
    raw = os.getenv("EOH_MEMORY_EXPERIMENTS", "").strip()
    if not raw:
        return None
    names = []
    for token in raw.split(","):
        token = token.strip()
        if token:
            names.append(token)
    return set(names) if names else None


def _clean_previous_outputs(mode_root: Path):
    if mode_root.exists():
        shutil.rmtree(mode_root)
    mode_root.mkdir(parents=True, exist_ok=True)


def run_once(
    experiment: dict,
    output_path: str,
    bridge_url: str,
    model_id: str,
    settings: dict,
    log,
    shared_seed_path: Path | None = None,
):
    from eoh import eoh
    from eoh.utils.getParas import Paras

    mode_root = Path(output_path).resolve()
    _clean_previous_outputs(mode_root)
    run_log_path = mode_root / "results" / "run_log.jsonl"
    stop_event = threading.Event()
    monitor = threading.Thread(target=_tail_run_log, args=(run_log_path, stop_event, log, experiment["name"]), daemon=True)
    heart = threading.Thread(target=_heartbeat, args=(mode_root, run_log_path, stop_event, log, experiment["name"]), daemon=True)
    monitor.start()
    heart.start()

    random.seed(settings["seed"])
    np.random.seed(settings["seed"])

    try:
        paras = Paras()
        base_kwargs = dict(
            method="eoh",
            problem=settings["problem"],
            llm_use_local=True,
            llm_local_url=bridge_url,
            llm_model=model_id,
            ec_pop_size=settings["pop_size"],
            ec_n_pop=settings["generations"],
            exp_n_proc=settings["n_proc"],
            exp_output_path=str(mode_root),
            exp_debug_mode=False,
            eval_parallel_instances=settings["eval_parallel_instances"],
            eval_instances_per_gen=settings["eval_instances_per_gen"],
            holdout_instances=settings["holdout_instances"],
            holdout_eval_interval=settings["holdout_eval_interval"],
            eoh_mode=experiment["mode"],
            exp_use_seed=bool(shared_seed_path),
            exp_seed_path=str(shared_seed_path) if shared_seed_path else "./seeds/seeds.json",
            log_full_population=False,
            memory_top_k=settings["memory_top_k"],
            memory_max_entries=settings["memory_max_entries"],
            memory_min_score=settings["memory_min_score"],
            memory_include_failures=settings["memory_include_failures"],
        )
        base_kwargs.update(experiment["paras"])
        paras.set_paras(**base_kwargs)
        if settings.get("disable_numba", False):
            paras.eva_numba_decorator = False

        log(
            f"starting {experiment['name']}",
            experiment=experiment["name"],
            output_path=str(mode_root),
            shared_seed_path=str(shared_seed_path) if shared_seed_path else None,
            memory_store_path=experiment["paras"].get("memory_store_path"),
            memory_mode=experiment["paras"].get("memory_mode"),
            memory_read_enabled=experiment["paras"].get("memory_read_enabled"),
            memory_write_enabled=experiment["paras"].get("memory_write_enabled"),
        )

        runner = eoh.EVOL(paras)
        runner.run()
        log(f"finished {experiment['name']}", experiment=experiment["name"], event="experiment_finished")
    finally:
        stop_event.set()
        monitor.join(timeout=3)
        heart.join(timeout=3)


def _collect_summary(experiment_root: Path) -> dict:
    run_log_path = experiment_root / "results" / "run_log.jsonl"
    memory_summary_path = experiment_root / "results" / "memory_usage_summary.json"
    rows = []
    if run_log_path.exists():
        with run_log_path.open("r", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
    final_best = rows[-1].get("best_fitness") if rows else None
    payload = {
        "experiment_root": str(experiment_root),
        "final_best_fitness": final_best,
        "n_generations_logged": len(rows),
    }
    if memory_summary_path.exists():
        with memory_summary_path.open("r", encoding="utf-8") as f:
            payload["memory_usage"] = json.load(f)
    return payload


def main():
    project_root, eoh_src = ensure_eoh_src_on_path()
    settings = {
        "problem": os.getenv("EOH_PROBLEM", "bp_online"),
        "pop_size": int(os.getenv("EOH_POP_SIZE", "8")),
        "generations": int(os.getenv("EOH_N_GENERATIONS", "10")),
        "n_proc": int(os.getenv("EOH_N_PROC", str(min(24, os.cpu_count() or 1)))),
        "eval_parallel_instances": int(os.getenv("EOH_EVAL_PARALLEL_INSTANCES", "1")),
        "eval_instances_per_gen": int(os.getenv("EOH_EVAL_INSTANCES_PER_GEN", "256")),
        "holdout_instances": int(os.getenv("EOH_HOLDOUT_INSTANCES", "64")),
        "holdout_eval_interval": int(os.getenv("EOH_HOLDOUT_EVAL_INTERVAL", "1")),
        "seed": int(os.getenv("EOH_SEED", "2024")),
        "share_initial_population": os.getenv("EOH_SHARE_INITIAL_POP", "1") == "1",
        "disable_numba": os.getenv("EOH_DISABLE_NUMBA", "1") == "1",
        "memory_top_k": int(os.getenv("EOH_MEMORY_TOP_K", "3")),
        "memory_max_entries": int(os.getenv("EOH_MEMORY_MAX_ENTRIES", "5000")),
        "memory_min_score": float(os.getenv("EOH_MEMORY_MIN_SCORE", "0.25")),
        "memory_include_failures": os.getenv("EOH_MEMORY_INCLUDE_FAILURES", "1") == "1",
    }

    out_root = Path(os.getenv("EOH_MEMORY_COMPARE_OUT", "./compare_runs/experience_memory")).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    status_log_path = out_root / "runner_status.jsonl"
    if status_log_path.exists():
        status_log_path.unlink()
    log = _make_logger(status_log_path)

    memory_store_path = Path(os.getenv("EOH_MEMORY_STORE_PATH", str(out_root / "shared_memory" / "experience_memory.jsonl"))).resolve()
    if os.getenv("EOH_MEMORY_RESET_STORE", "1") == "1":
        if memory_store_path.exists():
            memory_store_path.unlink()

    summary_path = out_root / "memory_experiment_summary.json"
    cfg = config_from_env()
    log("memory experiments started", project_root=str(project_root), eoh_src=str(eoh_src), out_root=str(out_root), memory_store_path=str(memory_store_path))

    server = None
    try:
        resolved_model = resolve_model_id(cfg)
        log("resolved model", requested_model=cfg.model, resolved_model=resolved_model)
        server, _thread, bridge_url, model_id = start_hpc_bridge(cfg)
        log("bridge ready", bridge_url=bridge_url, model_id=model_id)
        status, payload = test_bridge(bridge_url)
        log("bridge probe complete", bridge_test_status=status, bridge_test_payload=payload)

        shared_seed_path = None
        if settings["share_initial_population"]:
            shared_seed_path = _prepare_shared_seed(out_root, bridge_url, model_id, settings, log)

        selected = _selected_experiment_names()
        all_experiments = _experiment_configs(memory_store_path)
        if selected is not None:
            all_experiments = [exp for exp in all_experiments if exp["name"] in selected]
            if len(all_experiments) == 0:
                raise RuntimeError(f"No matching experiments for EOH_MEMORY_EXPERIMENTS={sorted(selected)}")

        experiment_summaries = []
        for experiment in all_experiments:
            exp_root = out_root / experiment["name"]
            exp_shared_seed = shared_seed_path if experiment.get("use_shared_seed", False) else None
            run_once(
                experiment=experiment,
                output_path=str(exp_root),
                bridge_url=bridge_url,
                model_id=model_id,
                settings=settings,
                log=log,
                shared_seed_path=exp_shared_seed,
            )
            experiment_summaries.append(_collect_summary(exp_root))

        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "time": _ts(),
                    "settings": settings,
                    "memory_store_path": str(memory_store_path),
                    "experiments": experiment_summaries,
                },
                f,
                indent=2,
            )
        log("memory experiments completed", summary_path=str(summary_path))
    except Exception as exc:
        log("memory experiments failed", error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        stop_hpc_bridge(server)
        log("bridge stopped")


if __name__ == "__main__":
    main()
