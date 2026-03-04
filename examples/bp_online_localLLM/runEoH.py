from eoh import eoh
from eoh.utils.getParas import Paras
import os

# Parameter initilization #
paras = Paras() 

# Set parameters #
paras.set_paras(method = "eoh",    # ['ael','eoh']
                problem = "bp_online", #['tsp_construct','bp_online']
                llm_use_local = True, # set your LLM endpoint
                llm_local_url = "http://xxx/completions",   # set your url for local deployment
                ec_pop_size = int(os.getenv("EOH_POP_SIZE", "8")), # number of samples in each population
                ec_n_pop = int(os.getenv("EOH_N_GENERATIONS", "10")),  # number of populations
                exp_n_proc = int(os.getenv("EOH_N_PROC", str(min(24, os.cpu_count() or 1)))),  # multi-core parallel
                eval_parallel_instances = int(os.getenv("EOH_EVAL_PARALLEL_INSTANCES", "1")),
                eoh_mode = os.getenv("EOH_MODE", "routed"),
                route_controller_enabled = os.getenv("EOH_ROUTE_CONTROLLER_ENABLED", "1") == "1",
                route_controller_window = int(os.getenv("EOH_ROUTE_CONTROLLER_WINDOW", "5")),
                exp_debug_mode = False)

# initilization
evolution = eoh.EVOL(paras)

# run 
evolution.run()
