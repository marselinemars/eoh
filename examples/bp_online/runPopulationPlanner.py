from eoh import eoh
from eoh.utils.getParas import Paras


paras = Paras()
paras.set_paras(
    method="eoh",
    problem="bp_online",
    eoh_mode="population_planner",
    llm_api_endpoint="xxx",
    llm_api_key="xxx",
    llm_model="gpt-3.5-turbo",
    ec_pop_size=4,
    ec_n_pop=4,
    exp_n_proc=4,
    exp_debug_mode=False,
    planner_profile_train_instances=8,
    planner_profile_holdout_instances=8,
    planner_view_size=8,
)

evolution = eoh.EVOL(paras)
evolution.run()
