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


def _tail_run_log(run_log_path: Path, stop_event: threading.Event, log, mode: str):
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
                            f"{mode} progress: gen={rec.get('gen')} best={rec.get('best_fitness')} "
                            f"op={rec.get('chosen_operator')} label={rec.get('diagnosis_label')}",
                            mode=mode,
                            event="generation_progress",
                            gen=rec.get("gen"),
                            best_fitness=rec.get("best_fitness"),
                            chosen_operator=rec.get("chosen_operator"),
                            diagnosis_label=rec.get("diagnosis_label"),
                        )
                    except Exception:
                        log(f"{mode} progress raw: {line.strip()}", mode=mode, event="generation_progress_raw")
        time.sleep(2)


def _heartbeat(mode_root: Path, run_log_path: Path, pop0_path: Path, stop_event: threading.Event, log, mode: str):
    started = time.time()
    while not stop_event.is_set():
        elapsed = int(time.time() - started)
        run_log_lines = 0
        if run_log_path.exists():
            with run_log_path.open("r", encoding="utf-8") as f:
                run_log_lines = sum(1 for _ in f)
        log(
            f"{mode} heartbeat: elapsed={elapsed}s pop0_exists={pop0_path.exists()} run_log_lines={run_log_lines}",
            mode=mode,
            event="heartbeat",
            elapsed_s=elapsed,
            pop0_exists=pop0_path.exists(),
            run_log_lines=run_log_lines,
        )
        time.sleep(20)


def run_once(mode: str, output_path: str, bridge_url: str, model_id: str, settings: dict, log):
    from eoh import eoh
    from eoh.utils.getParas import Paras

    mode_root = Path(output_path).resolve()
    mode_root.mkdir(parents=True, exist_ok=True)
    run_log_path = mode_root / "results" / "run_log.jsonl"
    pop0_path = mode_root / "results" / "pops" / "population_generation_0.json"
    log(
        f"starting mode={mode}",
        mode=mode,
        output_path=str(mode_root),
        run_log_path=str(run_log_path),
        population0_path=str(pop0_path),
    )

    stop_event = threading.Event()
    monitor = threading.Thread(target=_tail_run_log, args=(run_log_path, stop_event, log, mode), daemon=True)
    heart = threading.Thread(target=_heartbeat, args=(mode_root, run_log_path, pop0_path, stop_event, log, mode), daemon=True)
    monitor.start()
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
            eoh_mode=mode,
            log_full_population=False,
        )
        runner = eoh.EVOL(paras)
        runner.run()
        log(f"finished mode={mode}", mode=mode, event="mode_finished")
    finally:
        stop_event.set()
        monitor.join(timeout=3)
        heart.join(timeout=3)


def main():
    project_root, eoh_src = ensure_eoh_src_on_path()
    settings = {
        "problem": os.getenv("EOH_PROBLEM", "bp_online"),
        "pop_size": int(os.getenv("EOH_POP_SIZE", "8")),
        "generations": int(os.getenv("EOH_N_GENERATIONS", "10")),
        "n_proc": int(os.getenv("EOH_N_PROC", "1")),
        "eval_instances_per_gen": int(os.getenv("EOH_EVAL_INSTANCES_PER_GEN", "256")),
        "seed": int(os.getenv("EOH_SEED", "2024")),
    }

    out_root = Path(os.getenv("EOH_COMPARE_OUT", "./compare_runs")).resolve()
    baseline_out = str(out_root / "baseline")
    routed_out = str(out_root / "routed")
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
    log("experiment settings", **settings)
    log("output paths prepared", out_root=str(out_root), baseline_out=baseline_out, routed_out=routed_out)

    server = None
    try:
        log("starting hpc bridge")
        server, _thread, bridge_url, model_id = start_hpc_bridge(cfg)
        log("hpc bridge ready", bridge_url=bridge_url, resolved_model=model_id)
        status, payload = test_bridge(bridge_url)
        log("bridge probe complete", bridge_test_status=status, bridge_test_payload=payload)

        run_once("baseline", baseline_out, bridge_url, model_id, settings, log)
        log("baseline run log path", path=str(Path(baseline_out) / "results" / "run_log.jsonl"))

        run_once("routed", routed_out, bridge_url, model_id, settings, log)
        log("routed run log path", path=str(Path(routed_out) / "results" / "run_log.jsonl"))
        log("compare run completed successfully")
    except Exception as exc:
        log("compare run failed", error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        stop_hpc_bridge(server)
        log("Bridge stopped")


if __name__ == "__main__":
    main()
