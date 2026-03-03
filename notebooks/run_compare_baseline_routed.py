import os
import random
import json
import time
import traceback
import threading
from pathlib import Path
from datetime import datetime

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


def _parse_compare_seeds():
    raw = os.getenv("EOH_COMPARE_SEEDS", "").strip()
    if not raw:
        return [int(os.getenv("EOH_SEED", "2024"))], False
    seeds = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        seeds.append(int(token))
    if not seeds:
        seeds = [int(os.getenv("EOH_SEED", "2024"))]
    return seeds, True


def _tail_run_log(run_log_path: Path, stop_event: threading.Event, log, mode: str, run_tag: str):
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
                            f"{run_tag} {mode} progress: gen={rec.get('gen')} train={rec.get('train_fitness', rec.get('best_fitness'))} "
                            f"holdout={rec.get('holdout_fitness')} gap={rec.get('fitness_gap')} "
                            f"op={rec.get('chosen_operator')} label={rec.get('diagnosis_label')}",
                            run_tag=run_tag,
                            mode=mode,
                            event="generation_progress",
                            gen=rec.get("gen"),
                            train_fitness=rec.get("train_fitness", rec.get("best_fitness")),
                            holdout_fitness=rec.get("holdout_fitness"),
                            fitness_gap=rec.get("fitness_gap"),
                            chosen_operator=rec.get("chosen_operator"),
                            diagnosis_label=rec.get("diagnosis_label"),
                        )
                    except Exception:
                        log(
                            f"{run_tag} {mode} progress raw: {line.strip()}",
                            run_tag=run_tag,
                            mode=mode,
                            event="generation_progress_raw",
                        )
        time.sleep(2)


def _tail_operator_log(operator_log_path: Path, stop_event: threading.Event, log, mode: str, run_tag: str):
    offset = 0
    while not stop_event.is_set():
        if operator_log_path.exists():
            with operator_log_path.open("r", encoding="utf-8") as f:
                f.seek(offset)
                while True:
                    line = f.readline()
                    if not line:
                        break
                    offset = f.tell()
                    try:
                        rec = json.loads(line)
                        log(
                            f"{run_tag} {mode} operator: gen={rec.get('gen')} op={rec.get('operator')} "
                            f"invalid_rate={rec.get('invalid_rate')} delta={rec.get('delta_best')}",
                            run_tag=run_tag,
                            mode=mode,
                            event="operator_progress",
                            gen=rec.get("gen"),
                            operator=rec.get("operator"),
                            invalid_rate=rec.get("invalid_rate"),
                            delta_best=rec.get("delta_best"),
                            diagnosis_label=rec.get("diagnosis_label"),
                            n_offspring=rec.get("n_offspring"),
                            n_valid=rec.get("n_valid"),
                            n_invalid=rec.get("n_invalid"),
                        )
                    except Exception:
                        log(
                            f"{run_tag} {mode} operator raw: {line.strip()}",
                            run_tag=run_tag,
                            mode=mode,
                            event="operator_progress_raw",
                        )
        time.sleep(2)


def _heartbeat(
    mode_root: Path,
    run_log_path: Path,
    operator_log_path: Path,
    pop0_path: Path,
    stop_event: threading.Event,
    log,
    mode: str,
    run_tag: str,
):
    started = time.time()
    while not stop_event.is_set():
        elapsed = int(time.time() - started)
        run_log_lines = 0
        operator_log_lines = 0
        if run_log_path.exists():
            with run_log_path.open("r", encoding="utf-8") as f:
                run_log_lines = sum(1 for _ in f)
        if operator_log_path.exists():
            with operator_log_path.open("r", encoding="utf-8") as f:
                operator_log_lines = sum(1 for _ in f)
        log(
            f"{run_tag} {mode} heartbeat: elapsed={elapsed}s pop0_exists={pop0_path.exists()} "
            f"run_log_lines={run_log_lines} operator_log_lines={operator_log_lines}",
            run_tag=run_tag,
            mode=mode,
            event="heartbeat",
            elapsed_s=elapsed,
            pop0_exists=pop0_path.exists(),
            run_log_lines=run_log_lines,
            operator_log_lines=operator_log_lines,
        )
        time.sleep(20)


def run_once(mode: str, output_path: str, bridge_url: str, model_id: str, settings: dict, log, run_tag: str):
    from eoh import eoh
    from eoh.utils.getParas import Paras

    mode_root = Path(output_path).resolve()
    mode_root.mkdir(parents=True, exist_ok=True)
    run_log_path = mode_root / "results" / "run_log.jsonl"
    operator_log_path = mode_root / "results" / "operator_events.jsonl"
    pop0_path = mode_root / "results" / "pops" / "population_generation_0.json"
    llm_io_dir = mode_root / "results" / "llm_io"
    llm_io_path = llm_io_dir / "llm_interactions.jsonl"
    parse_events_path = llm_io_dir / "parse_events.jsonl"
    # Remove stale files so progress is unambiguous.
    if run_log_path.exists():
        run_log_path.unlink()
    if operator_log_path.exists():
        operator_log_path.unlink()
    if pop0_path.exists():
        pop0_path.unlink()
    if llm_io_path.exists():
        llm_io_path.unlink()
    if parse_events_path.exists():
        parse_events_path.unlink()

    if settings.get("log_llm_io", True):
        os.environ["EOH_LOG_LLM_IO"] = "1"
        os.environ["EOH_LOG_PARSE_EVENTS"] = "1"
        os.environ["EOH_LLM_IO_DIR"] = str(llm_io_dir)
    else:
        os.environ["EOH_LOG_LLM_IO"] = "0"
        os.environ["EOH_LOG_PARSE_EVENTS"] = "0"

    log(
        f"{run_tag} starting mode={mode}",
        run_tag=run_tag,
        mode=mode,
        output_path=str(mode_root),
        run_log_path=str(run_log_path),
        operator_log_path=str(operator_log_path),
        population0_path=str(pop0_path),
        llm_io_path=str(llm_io_path),
        parse_events_path=str(parse_events_path),
    )

    stop_event = threading.Event()
    monitor = threading.Thread(
        target=_tail_run_log,
        args=(run_log_path, stop_event, log, mode, run_tag),
        daemon=True,
    )
    op_monitor = threading.Thread(
        target=_tail_operator_log,
        args=(operator_log_path, stop_event, log, mode, run_tag),
        daemon=True,
    )
    heart = threading.Thread(
        target=_heartbeat,
        args=(mode_root, run_log_path, operator_log_path, pop0_path, stop_event, log, mode, run_tag),
        daemon=True,
    )
    monitor.start()
    op_monitor.start()
    heart.start()

    random.seed(settings["seed"])
    np.random.seed(settings["seed"])

    try:
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
            exp_output_path=output_path,
            exp_debug_mode=False,
            eval_instances_per_gen=settings["eval_instances_per_gen"],
            holdout_instances=settings["holdout_instances"],
            holdout_eval_interval=settings["holdout_eval_interval"],
            eoh_mode=mode,
            log_full_population=False,
        )
        if settings.get("disable_numba", False):
            paras.eva_numba_decorator = False
        runner = eoh.EVOL(paras)
        runner.run()
        log(f"{run_tag} finished mode={mode}", run_tag=run_tag, mode=mode, event="mode_finished")
    finally:
        stop_event.set()
        monitor.join(timeout=3)
        op_monitor.join(timeout=3)
        heart.join(timeout=3)


def main():
    project_root, eoh_src = ensure_eoh_src_on_path()
    compare_seeds, from_compare_env = _parse_compare_seeds()
    settings = {
        "problem": os.getenv("EOH_PROBLEM", "bp_online"),
        "pop_size": int(os.getenv("EOH_POP_SIZE", "8")),
        "generations": int(os.getenv("EOH_N_GENERATIONS", "10")),
        "n_proc": int(os.getenv("EOH_N_PROC", "1")),
        "eval_instances_per_gen": int(os.getenv("EOH_EVAL_INSTANCES_PER_GEN", "256")),
        "holdout_instances": int(os.getenv("EOH_HOLDOUT_INSTANCES", "64")),
        "holdout_eval_interval": int(os.getenv("EOH_HOLDOUT_EVAL_INTERVAL", "1")),
        "disable_numba": os.getenv("EOH_DISABLE_NUMBA", "1") == "1",
        "log_llm_io": os.getenv("EOH_LOG_LLM_IO", "1") == "1",
    }

    out_root = Path(os.getenv("EOH_COMPARE_OUT", "./compare_runs")).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    status_log_path = out_root / "runner_status.jsonl"
    if status_log_path.exists():
        status_log_path.unlink()
    log = _make_logger(status_log_path)

    cfg = config_from_env()
    log("compare run started", project_root=str(project_root), eoh_src=str(eoh_src))
    log(
        "hpc config loaded",
        hpc_base=cfg.base_url,
        bridge_port=cfg.port,
        requested_model=cfg.model,
        api_key_present=bool(cfg.api_key),
    )
    log(
        "experiment settings",
        **settings,
        compare_seeds=compare_seeds,
        seeds_source="EOH_COMPARE_SEEDS" if from_compare_env else "EOH_SEED",
    )
    log("output root prepared", out_root=str(out_root))

    server = None
    try:
        log("starting hpc bridge")
        server, _thread, bridge_url, model_id = start_hpc_bridge(cfg)
        log("hpc bridge ready", bridge_url=bridge_url, resolved_model=model_id)
        status, payload = test_bridge(bridge_url)
        log("bridge probe complete", bridge_test_status=status, bridge_test_payload=payload)

        use_seed_subdirs = len(compare_seeds) > 1
        for seed in compare_seeds:
            run_tag = f"seed={seed}"
            seed_settings = dict(settings)
            seed_settings["seed"] = int(seed)
            seed_root = out_root / f"seed_{seed}" if use_seed_subdirs else out_root
            baseline_out = str(seed_root / "baseline")
            routed_out = str(seed_root / "routed")
            log(
                f"{run_tag} output paths prepared",
                run_tag=run_tag,
                seed=seed,
                seed_root=str(seed_root),
                baseline_out=baseline_out,
                routed_out=routed_out,
            )

            run_once("baseline", baseline_out, bridge_url, model_id, seed_settings, log, run_tag)
            log(
                f"{run_tag} baseline run log path",
                run_tag=run_tag,
                path=str(Path(baseline_out) / "results" / "run_log.jsonl"),
            )
            log(
                f"{run_tag} baseline operator log path",
                run_tag=run_tag,
                path=str(Path(baseline_out) / "results" / "operator_events.jsonl"),
            )

            run_once("routed", routed_out, bridge_url, model_id, seed_settings, log, run_tag)
            log(
                f"{run_tag} routed run log path",
                run_tag=run_tag,
                path=str(Path(routed_out) / "results" / "run_log.jsonl"),
            )
            log(
                f"{run_tag} routed operator log path",
                run_tag=run_tag,
                path=str(Path(routed_out) / "results" / "operator_events.jsonl"),
            )

        log("compare run completed successfully")
    except Exception as exc:
        log("compare run failed", error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        stop_hpc_bridge(server)
        log("Bridge stopped")


if __name__ == "__main__":
    main()
