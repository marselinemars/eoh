import os
import random
from pathlib import Path

import numpy as np

from hpc_llm_setup import (
    config_from_env,
    ensure_eoh_src_on_path,
    start_hpc_bridge,
    stop_hpc_bridge,
)


def run_once(mode: str, output_path: str, bridge_url: str, model_id: str, settings: dict):
    from eoh import eoh
    from eoh.utils.getParas import Paras

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
        exp_output_path=output_path,
        exp_debug_mode=False,
        eval_instances_per_gen=settings["eval_instances_per_gen"],
        eoh_mode=mode,
        log_full_population=False,
    )
    runner = eoh.EVOL(paras)
    runner.run()


def main():
    project_root, eoh_src = ensure_eoh_src_on_path()
    print("PROJECT_ROOT:", project_root)
    print("EOH_SRC:", eoh_src)

    cfg = config_from_env()
    print("HPC_BASE:", cfg.base_url)
    print("BRIDGE_PORT:", cfg.port)
    print("HPC_MODEL:", cfg.model)

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

    server = None
    try:
        server, _thread, bridge_url, model_id = start_hpc_bridge(cfg)
        print("BRIDGE_URL:", bridge_url)
        print("MODEL_ID:", model_id)
        print("SETTINGS:", settings)

        print("\n=== RUN BASELINE ===")
        run_once("baseline", baseline_out, bridge_url, model_id, settings)
        print("baseline log:", Path(baseline_out) / "results" / "run_log.jsonl")

        print("\n=== RUN ROUTED ===")
        run_once("routed", routed_out, bridge_url, model_id, settings)
        print("routed log:", Path(routed_out) / "results" / "run_log.jsonl")
    finally:
        stop_hpc_bridge(server)
        print("Bridge stopped")


if __name__ == "__main__":
    main()
