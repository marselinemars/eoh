import glob
import json
import os
from pathlib import Path

from hpc_llm_setup import (
    config_from_env,
    ensure_eoh_src_on_path,
    start_hpc_bridge,
    stop_hpc_bridge,
    test_bridge,
)


def main():
    project_root, eoh_src = ensure_eoh_src_on_path()
    print("PROJECT_ROOT:", project_root)
    print("EOH_SRC:", eoh_src)

    cfg = config_from_env()
    print("HPC_BASE:", cfg.base_url)
    print("BRIDGE_PORT:", cfg.port)
    print("HPC_MODEL:", cfg.model)

    server = None
    try:
        server, _thread, bridge_url, model_id = start_hpc_bridge(cfg)
        print("BRIDGE_URL:", bridge_url)
        print("MODEL_ID:", model_id)

        status, payload = test_bridge(bridge_url)
        print("BRIDGE_TEST_STATUS:", status)
        print("BRIDGE_TEST_PAYLOAD:", payload)

        from eoh import eoh
        from eoh.utils.getParas import Paras

        problem = os.getenv("EOH_PROBLEM", "bp_online")
        output_path = os.getenv("EOH_OUTPUT_PATH", "./")
        pop_size = int(os.getenv("EOH_POP_SIZE", "3"))
        n_generations = int(os.getenv("EOH_N_GENERATIONS", "2"))
        n_proc = int(os.getenv("EOH_N_PROC", "1"))

        paras = Paras()
        paras.set_paras(
            method="eoh",
            problem=problem,
            llm_use_local=True,
            llm_local_url=bridge_url,
            llm_model=model_id,
            ec_pop_size=pop_size,
            ec_n_pop=n_generations,
            exp_n_proc=n_proc,
            exp_output_path=output_path,
            exp_debug_mode=False,
        )

        runner = eoh.EVOL(paras)
        runner.run()

        pop_files = sorted(glob.glob("./results/pops/population_generation_*.json"))
        best_files = sorted(glob.glob("./results/pops_best/population_generation_*.json"))
        print("Population files:", pop_files)
        print("Best files:", best_files)
        if pop_files:
            latest = pop_files[-1]
            with open(latest, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            print("Latest pop file:", latest)
            print("Individuals:", len(data))
            if len(data) > 0:
                print("Sample objective:", data[0].get("objective"))
    finally:
        stop_hpc_bridge(server)
        print("Bridge stopped")


if __name__ == "__main__":
    main()
