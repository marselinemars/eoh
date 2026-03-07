from eoh import eoh
from eoh.utils.getParas import Paras


paras = Paras()
paras.set_paras(
    method="eoh",
    problem="bp_online",
    eoh_mode="population_planner",
    llm_api_endpoint="http://vllm-nodeport.vllm-ns.svc.cluster.local:8000/v1",
    llm_api_key="my-key-ensia-2022-1030",
    llm_model="QuantTrio/Qwen3-VL-235B-A22B-Instruct-AWQ",
    ec_pop_size=4,
    ec_n_pop=4,
    exp_n_proc=1,
    exp_debug_mode=False,
    planner_profile_train_instances=8,
    planner_profile_holdout_instances=8,
    planner_view_size=8,
)

evolution = eoh.EVOL(paras)
evolution.run()
