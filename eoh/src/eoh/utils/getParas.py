DEFAULT_ENSIA_BASE_URL = "http://vllm-nodeport.vllm-ns.svc.cluster.local:8000/v1"
DEFAULT_ENSIA_API_KEY = "my-key-ensia-2022-1030"
DEFAULT_ENSIA_MODEL = "QuantTrio/Qwen3-VL-235B-A22B-Instruct-AWQ"


class Paras():
    def __init__(self):
        #####################
        ### General settings  ###
        #####################
        self.method = 'eoh'
        self.problem = 'tsp_construct'
        self.selection = None
        self.management = None

        #####################
        ###  EC settings  ###
        #####################
        self.ec_pop_size = 5  # number of algorithms in each population, default = 10
        self.ec_n_pop = 5 # number of populations, default = 10
        self.ec_operators = None # evolution operators: ['e1','e2','m1','m2'], default =  ['e1','e2','m1','m2']
        self.ec_m = 2  # number of parents for 'e1' and 'e2' operators, default = 2
        self.ec_operator_weights = None  # weights for operators, i.e., the probability of use the operator in each iteration, default = [1,1,1,1]
        
        #####################
        ### LLM settings  ###
        #####################
        self.llm_use_local = False  # if use local model
        self.llm_local_url = None  # your local server 'http://127.0.0.1:11012/completions'
        self.llm_api_endpoint = DEFAULT_ENSIA_BASE_URL # endpoint for remote LLM, e.g., api.deepseek.com
        self.llm_api_key = DEFAULT_ENSIA_API_KEY  # API key for remote LLM, e.g., sk-xxxx
        self.llm_model = DEFAULT_ENSIA_MODEL  # model type for remote LLM, e.g., deepseek-chat

        #####################
        ###  Exp settings  ###
        #####################
        self.exp_debug_mode = False  # if debug
        self.exp_output_path = "./"  # default folder for ael outputs
        self.exp_use_seed = False
        self.exp_seed_path = "./seeds/seeds.json"
        self.exp_use_continue = False
        self.exp_continue_id = 0
        self.exp_continue_path = "./results/pops/population_generation_0.json"
        self.exp_n_proc = 1
        
        #####################
        ###  Evaluation settings  ###
        #####################
        self.eva_timeout = 30
        self.eva_numba_decorator = False
        self.eval_instances_per_gen = None
        self.holdout_instances = 64
        self.holdout_eval_interval = 1
        self.planner_profile_train_instances = 8
        self.planner_profile_holdout_instances = 8

        #####################
        ###  Run mode / logging ###
        #####################
        self.eoh_mode = "baseline"  # baseline | routed | population_planner
        self.route_improvement_epsilon = 1e-12
        self.route_stagnation_k = 3
        self.route_invalid_rate_threshold = 0.5
        self.route_use_diversity = True
        self.route_recent_window = 3
        self.route_structural_plateau_eps = 1e-5
        self.route_warmup_gens = 2
        self.route_e1_cooldown = 3
        self.route_e2_recent_k = 3
        self.route_use_probabilistic = True
        self.log_full_population = False
        self.planner_view_size = 8


    def set_parallel(self):
        import multiprocessing
        num_processes = multiprocessing.cpu_count()
        if self.exp_n_proc == -1 or self.exp_n_proc > num_processes:
            self.exp_n_proc = num_processes
            print(f"Set the number of proc to {num_processes} .")
    
    def set_ec(self):    
        
        if self.management == None:
            if self.method in ['ael','eoh']:
                self.management = 'pop_greedy'
            elif self.method == 'ls':
                self.management = 'ls_greedy'
            elif self.method == 'sa':
                self.management = 'ls_sa'
        
        if self.selection == None:
            self.selection = 'prob_rank'
            
        
        if self.ec_operators == None:
            if self.method == 'eoh':
                self.ec_operators  = ['e1','e2','m1','m2']
            elif self.method == 'ael':
                self.ec_operators  = ['crossover','mutation']
            elif self.method == 'ls':
                self.ec_operators  = ['m1']
            elif self.method == 'sa':
                self.ec_operators  = ['m1']

        if self.ec_operator_weights == None:
            self.ec_operator_weights = [1 for _ in range(len(self.ec_operators))]
        elif len(self.ec_operator) != len(self.ec_operator_weights):
            print("Warning! Lengths of ec_operator_weights and ec_operator shoud be the same.")
            self.ec_operator_weights = [1 for _ in range(len(self.ec_operators))]
                    
        if self.method in ['ls','sa'] and self.ec_pop_size >1:
            self.ec_pop_size = 1
            self.exp_n_proc = 1
            print("> single-point-based, set pop size to 1. ")
            
    def set_evaluation(self):
        # Initialize evaluation settings
        if self.problem == 'bp_online':
            self.eva_timeout = 20
            self.eva_numba_decorator  = True
        elif self.problem == 'tsp_construct':
            self.eva_timeout = 20
                
    def set_paras(self, *args, **kwargs):
        
        # Map paras
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
              
        # Identify and set parallel 
        self.set_parallel()
        
        # Initialize method and ec settings
        self.set_ec()
        
        # Initialize evaluation settings
        self.set_evaluation()




if __name__ == "__main__":

    # Create an instance of the Paras class
    paras_instance = Paras()

    # Setting parameters using the set_paras method
    paras_instance.set_paras(llm_use_local=True, llm_local_url='http://example.com', ec_pop_size=8)

    # Accessing the updated parameters
    print(paras_instance.llm_use_local)  # Output: True
    print(paras_instance.llm_local_url)  # Output: http://example.com
    print(paras_instance.ec_pop_size)    # Output: 8
            
            
            
